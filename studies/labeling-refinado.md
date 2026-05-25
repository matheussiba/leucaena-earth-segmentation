# Labeling refinado: polígono + CHM + NDVI + IGNORE

> Mudança no preparo do treino para reduzir a "confusão" do modelo (excesso
> de falsos positivos de leucaena). Implementada em maio/2026 a pedido do
> professor.

## TL;DR

Antes (legado):

```
dentro do polígono  → label = 1  (leucaena)
fora do polígono    → label = 0  (background)
```

Depois (modo refinado, ativo quando `--lidar-dir` é passado):

```
fora do polígono                       → label = 255  (IGNORE — pulado no treino)
dentro do polígono e
   CHM ≥ 4.5 m  E  NDVI ≥ 0.3          → label = 1    (leucaena confirmada)
dentro do polígono mas falha           → label = 0    (background "ativo")
```

Mais três ajustes complementares:

- Overlap entre patches: **0.5 → 0.6** (60%).
- Data augmentation: **rotação contínua 0–360°** + **translação ±10%** +
  hflip/vflip, aleatórios **toda época** (não geram arquivos extras no
  disco — aplicados no `__getitem__` do dataloader).
- Tiles **sem LiDAR são pulados** quando `--lidar-dir` é passado (sem CHM
  não conseguimos refinar o label, então não treinamos com eles).

## Por que isso resolve o problema da confusão

O modelo estava chamando muita coisa de leucaena que não é. A raiz é o
label de antes:

- Um polígono anotado pelo especialista cobre uma **mancha** de leucaena,
  mas a mancha tem **clareiras**, **árvores baixas**, **pasto entre os
  indivíduos** e às vezes uma **torre/antena** dentro.
- Tudo isso virava `1` no label → a rede aprendeu a chamar pasto, solo,
  edificação e mato baixo de "leucaena", porque era isso que estava no
  treino.

A regra nova ataca os dois lados:

1. **Dentro do polígono**, só os pixels que parecem **árvore alta e
   verde** (CHM ≥ 4.5 m, NDVI ≥ 0.3) ficam como leucaena.  
   Pixels baixos (clareira, gramado) viram `0` — ensinam ATIVAMENTE a
   rede que aquilo NÃO é leucaena, mesmo estando "dentro" da área
   anotada.
2. **Fora do polígono** vira `IGNORE (255)` — não são exemplos confiáveis
   de "não-leucaena" (podem ter leucaena que o especialista esqueceu).
   Excluí-los do loss é mais honesto do que assumir background.

## A regra exata em pseudocódigo

```python
in_poly  = polygon_mask == 1
is_tall  = chm  >= LEUCAENA_CHM_MIN_M    # 4.5 m
is_veg   = ndvi >= LEUCAENA_NDVI_MIN     # 0.3

label = full(IGNORE_INDEX, shape=tile)   # 255 em tudo
label[in_poly] = 0                       # dentro do polígono começa como background
label[in_poly & is_tall & is_veg] = 1    # confirma leucaena
```

NDVI vem direto do tile RGBN:

```
NDVI = (NIR - RED) / (NIR + RED)
```

CHM vem da banda 1 do `lidar/<tile>.tif` gerado por
`prep-lidar-rasters.py`.

## Por que `IGNORE_INDEX = 255` é seguro

Já está montado em todo lugar do projeto:

- `train.py`:
  `nn.CrossEntropyLoss(ignore_index=general.IGNORE_INDEX, weight=...)`  
  → pixels com label 255 **não contam** no loss.
- `utils/trainer.py`:
  `MulticlassF1Score(num_classes=general.N_CLASSES, ignore_index=general.DISCARDED_CLASS)`  
  → pixels com label 255 **não contam** na métrica F1.
- `evaluation.py`: filtra `label != IGNORE_INDEX` antes do
  `classification_report`.

Ou seja: o modelo nunca vê esses pixels, nem no gradiente, nem nas
métricas. É como se eles não existissem para fins de aprendizado.

## Quando o tile NÃO tem LiDAR

Decisão (alinhada antes de implementar): **pula o tile inteiro**.

Motivação: a regra exige CHM. Sem CHM não temos como aplicá-la, e
voltar ao label antigo SÓ para esses tiles mistura dois regimes muito
diferentes no mesmo conjunto de treino — exatamente o tipo de coisa que
estava gerando a confusão.

O log avisa:

```
[SKIP] no LiDAR tile for <tile_name> (lidar_dir set: refined-label mode requires CHM).
```

Se você quiser usar tiles sem LiDAR, rode o script **sem** `--lidar-dir`
(volta ao modo legado, polígono inteiro = 1).

## Por que overlap 60% (e não 80%)

- Em 50% o passo entre patches é 128 px (PATCH_SIZE/2). Cada pixel de
  leucaena aparece em ~4 patches.
- Em 60% o passo é ~102 px. Cada pixel aparece em ~6 patches.
- Mais overlap = mais visões da mesma região = treino mais robusto.
- Acima disso (70–80%) o custo de armazenamento e tempo de prep cresce
  rápido sem ganho proporcional.

O número final de patches por tile cresce aproximadamente como
`(1/(1-overlap))² · n_anteriores`. Com 0.6 a contagem é ~1.5× a de 0.5.

## Por que a augmentation foi para o dataloader

O professor sugeriu "translação + rotação 0–360 aleatória". Duas formas
de implementar:

| Opção | Onde | Pro | Contra |
|---|---|---|---|
| Disco | `prep-patches-from-tiles.py` gera N variações por patch | Visualizável no QGIS | 4×–8× espaço; mesmo conjunto fixo toda época |
| Memória (RAM, on-the-fly) | `PatchFileDataset.__getitem__` aplica random toda época | 0 espaço extra; rotações diferentes a cada epoch | Só dá pra "ver" no Tensorboard, não no QGIS |

Escolhemos **memória**. Cada patch original é "esticado" para infinitas
variações ao longo das 300 épocas: angulações diferentes, deslocamentos
diferentes, sem inflar o dataset físico.

### Detalhes técnicos da affine

- Rotação: `[0, 360°)` uniforme. Pixels que saem da imagem original
  ficam **pretos** na entrada (fill=0) e **IGNORE** no label
  (fill=255). Como o loss ignora 255, esses cantos rotacionados não
  influenciam o gradiente.
- Translação: ±10% da largura/altura em cada eixo (≈ ±25 px num patch
  de 256). Mesmo tratamento de fill.
- Interpolação:
  - **Bilinear** para opt e LiDAR (continuum) → suave, sem aliasing.
  - **Nearest** para o label (índices discretos 0/1/255) → nunca cria
    "meio-leucaena".
- Flips hflip/vflip continuam (50% de probabilidade cada). São de
  graça (sem reamostragem).

Constantes em `conf/general.py`:

```python
AUG_ROTATION_DEG   = 360.0
AUG_TRANSLATE_FRAC = 0.10
```

Para desligar rotação ou translação, basta zerar uma delas e regenerar
`__pycache__`.

## Impacto esperado nas métricas

- **Precision** para leucaena: tende a **subir** (modelo erra menos
  pasto/clareira/torre como leucaena).
- **Recall** para leucaena: pode **descer ligeiramente** no início
  (alguns pixels que antes eram aprendidos como leucaena por estarem em
  polígono agora são "background ativo"). O overlap maior + augmentation
  compensam.
- **F1** geral: tende a **subir** porque o ganho em precision é maior
  que a perda em recall (assumindo que a maioria dos erros antigos era
  falso positivo).
- **Patches usáveis**: cai um pouco, porque alguns tiles inteiros são
  pulados (sem LiDAR) e dentro dos polígonos o número de pixels `1`
  diminui. Se ficar muito baixo, abaixar `--min-target-class` (default
  `0.01` → testar `0.005`).

## Manifesto e GeoJSON: novas colunas

`manifest.csv` (e `patch_footprints.geojson`) ganharam:

- `polygon_fraction`: fração de pixels do patch com `label != 255`,
  isto é, **dentro** de alguma área anotada. No modo legado isso é
  sempre `1.0`.
- `polygon_pct`: idem, em porcentagem (para conveniência em QGIS).
- `leucaena_fraction`: continua sendo a fração de pixels com
  `label == 1`. No modo refinado isso é estritamente menor ou igual a
  `polygon_fraction`.

Útil para inspecionar no QGIS: você pode estilizar o GeoJSON colorindo
por `polygon_pct` para ver visualmente quanto de cada patch é "área
útil" vs "ignorada".

## Como rodar

Pipeline novo (refinamento ativo):

```bash
docker compose run --rm app python prep-patches-from-tiles.py \
    --tiles-dir /data/rgbir \
    --lidar-dir /data/lidar \
    --masks /data/masks/leucaena.geojson \
    --out-dir /prepared/patches \
    --overlap 0.6
```

Sem `--lidar-dir` (modo legado, polígono inteiro = 1):

```bash
docker compose run --rm app python prep-patches-from-tiles.py \
    --tiles-dir /data/rgbir \
    --masks /data/masks/leucaena.geojson \
    --out-dir /prepared/patches \
    --overlap 0.6
```

O treino (`train.py`) não muda: ele já leva o `IGNORE_INDEX` no
`CrossEntropyLoss` e no F1.

## Onde isso vive no código

- Constantes: [`conf/general.py`](../conf/general.py)
  - `PATCH_OVERLAP = 0.6`
  - `LEUCAENA_CHM_MIN_M = 4.5`
  - `LEUCAENA_NDVI_MIN = 0.3`
  - `AUG_ROTATION_DEG = 360.0`
  - `AUG_TRANSLATE_FRAC = 0.10`
- Refinamento por tile:
  [`prep-patches-from-tiles.py`](../prep-patches-from-tiles.py)
  → `_compute_ndvi_from_tile`, `_read_full_lidar_band`,
  `_refine_label_with_chm_ndvi`, integração em `_process_tile`.
- Augmentation por patch:
  [`utils/dataloader.py`](../utils/dataloader.py)
  → `_augment_opt_lidar_label` chamada por
  `PatchFileDataset.__getitem__`.
- Manifesto: `manifest.csv` ganhou `polygon_fraction`.
