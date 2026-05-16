# Guia rápido do código — leucaena-earth-segmentation

Objetivo: te tirar do "perdido entre vários scripts" e te dar uma espinha dorsal
clara para entender / controlar cada etapa. Não é um manual exaustivo; é o
mapa para você abrir os arquivos certos na ordem certa, ler o que importa, e
saber o que mexer quando quiser.

---

## 0. Visão geral em uma figura

```
       ┌──────────────────────────────────────────────────┐
       │  1) Preparar patches (uma vez por dataset)        │
       │     prep-patches-from-tiles.py                    │
       │     entrada: tiles .tif (RGBN) + polygons.geojson │
       │     saída:   prepared/patches/{opt,lbl}/*.npy     │
       │              prepared/patches/manifest.csv        │
       └──────────────────────────────────────────────────┘
                          │
                          ▼
       ┌──────────────────────────────────────────────────┐
       │  2) Treinar a rede                                │
       │     train.py --patch-source file                  │
       │     usa:  conf/model_<e>.py  → models/resunet.py  │
       │           utils/dataloader.py (PatchFileDataset)  │
       │           utils/trainer.py (loop + early stop)    │
       │     loga: experiments/exp_<e>/logs/{metrics.csv,  │
       │           train_<e>.txt, training_config.json}    │
       │     salva: experiments/exp_<e>/models/model.pt    │
       └──────────────────────────────────────────────────┘
                          │
                          ▼
       ┌──────────────────────────────────────────────────┐
       │  3) Visualizar curvas e prever                    │
       │     utils/plot_training.py     (gráficos)         │
       │     predict-tile-preview.py    (PNG + F1, 1 tile) │
       │     predict-tiles.py           (TIF + VRT, AOI)   │
       └──────────────────────────────────────────────────┘
```

Existe ainda um caminho **legacy** (`prep-data.py` + `prediction.py` + `evaluation.py`)
herdado do projeto original *tree_fusion*. Funciona com **uma cena única**
(VRT/mosaico) e suporta LiDAR. Hoje, no dia a dia, você usa o caminho
**tile-based** acima. Tem uma seção dedicada ao legacy no final.

---

## 1. Parte 1 — Preparação de patches

### Ponto de entrada
- **Script:** `prep-patches-from-tiles.py`

### O que ele faz, em ordem
1. Lê `--tiles-dir` (default: `/data/rgbir`) e `--masks` (default: GeoJSON).
2. Para **cada tile**:
   - Abre com GDAL → pega bbox e CRS do raster.
   - Chama `rasterize_geojson_for_tile()` (em `utils/ops.py`) — **filtra polígonos**
     que intersectam o bbox (OGR `SetSpatialFilter`) e rasteriza só esses na
     grade do tile. Pixels dentro de polígono = `1`, fora = `0`.
   - Desliza janelas `256×256` com `overlap=0.5` sobre o label, e mantém só as
     janelas com **≥ `--min-target-class`** (1%) de pixels de leucaena.
   - Lê **apenas** essas janelas do GeoTIFF (não a tile inteira) e salva como
     `.npy` em `prepared/patches/opt/<id>.npy` (uint8 BGRN) e `lbl/<id>.npy`.
3. No fim, embaralha todos os patches e divide em **train / val / test** com
   `--test-split=0.2`, `--val-split=0.2`. Escreve `manifest.csv` com a coluna `split`.

### Arquivos relevantes (abra nesta ordem)
1. `prep-patches-from-tiles.py` — comece pela docstring no topo (linhas 1–40)
   e depois pelo `main()` no final (`if __name__ == "__main__"`).
2. `utils/ops.py` — função `rasterize_geojson_for_tile()`. É **o coração da
   ligação imagem ↔ máscara** (reprojeção CRS + filtro espacial + queima de
   polígono).
3. `conf/general.py` — `PATCH_SIZE`, `PATCH_OVERLAP`, `TEST_SPLIT`, `VAL_SPLIT`.
4. `conf/default.py` — `MIN_TRAIN_CLASS = 0.01` (1% mínimo de leucaena por patch).
5. `conf/paths.py` — `PATH_TILES_DIR`, `PATH_MASKS`, `PATH_PATCHES_DIR`. Lê do
   `.env` quando rodando no Docker.

### O que você pode controlar
- **Tiles incluídos:** o que estiver em `--tiles-dir`. Para limitar a 4 tiles,
  basta deixar só esses na pasta.
- **Tamanho do patch e sobreposição:** `conf/general.py` (`PATCH_SIZE`,
  `PATCH_OVERLAP`).
- **Fração mínima de leucaena por patch:** flag `--min-target-class` ou
  `conf/default.py` (`MIN_TRAIN_CLASS`).
- **Banda RGBN vs BGRN:** flag `--band-order` (interno é sempre BGRN).
- **Split:** flags `--test-split` e `--val-split`.

### Comando padrão (dentro do container)

```bash
python prep-patches-from-tiles.py --band-order RGBN
```

Saída em `prepared/patches/`:
```
opt/<patch_id>.npy   uint8 (256,256,4)
lbl/<patch_id>.npy   uint8 (256,256)
manifest.csv         patch_id,tile_name,...,leucaena_fraction,split
preparation.txt      log do run
```

---

## 2. Parte 2 — Treinamento

### Ponto de entrada
- **Script:** `train.py`

### O que ele faz, em ordem
1. Lê flags (`-e <id_experimento>`, `-b <batch>`, `--patch-source file`,
   `--cache-patches`, …).
2. Cria `experiments/exp_<e>/{logs,models,visual,predicted,results}`.
3. Redireciona `stdout`/`stderr` para uma classe `Tee` → tudo o que é
   `print()` aparece **no terminal e** em `logs/train_<e>.txt`.
4. **Importa o modelo** dinâmico: `conf/model_<e>.py` define `get_model()`.
   - `model_1.py` → `ResUnetOpt` (só óptico, é o que você está usando)
   - `model_2.py` → `ResUnet` early fusion (óptico + LiDAR concatenados)
   - `model_3.py` → `LateFusion` (dois encoders/decoders separados, fusão no fim)
5. **Cria os datasets** (training / validation):
   - `PatchFileDataset` em `utils/dataloader.py` quando `--patch-source file`.
   - Cada `__getitem__` devolve `((opt_tensor, lidar_tensor), label_tensor)`.
   - Para experimento 1, `lidar_tensor` é um tensor de zeros (placeholder; o
     forward de `ResUnetOpt` ignora ele).
   - Com `--cache-patches`, todos os `.npy` do split são carregados para RAM
     uma vez (≈ 2 GiB para ~6k patches).
6. **Loss e otimizador:**
   - `CrossEntropyLoss` com `class_weights=[0.3, 0.7]` (mais peso na classe
     leucaena, que é minoria).
   - `Adam` com `lr=1e-4`.
   - `ExponentialLR(gamma=0.995)` — LR cai 0,5% por época.
7. **`EarlyStop`** em `utils/trainer.py`: depois de `EARLY_STOP_MIN_EPOCHS`,
   se o **val loss** não melhorar por `EARLY_STOP_PATIENCE` épocas seguidas,
   o treino para. Salva `model.pt` toda vez que o val melhora.
8. **Loop principal** (em `train.py`, perto da linha 265):
   ```python
   for t in range(MAX_EPOCHS):
       train_loop(...)     # treina, devolve (loss, F1)
       val_loop(...)       # avalia, devolve (loss, F1)
       early_stop.testEpoch(...)
       metrics_logger.log_epoch(...)   # escreve linha em metrics.csv
       scheduler.step()
   ```
9. Salva sample em `experiments/exp_<e>/visual/sample_*.png` por época.

### Arquivos relevantes (abra nesta ordem)
1. **`train.py`** — top-down. Primeiro o `argparse` (até a linha ~115), depois
   o bloco `try:` com importação do modelo, datasets, loss e loop.
2. **`conf/model_1.py`** — escolhe o modelo (`ResUnetOpt`). É a "ponte"
   entre flag `-e` e arquitetura. Para criar um experimento novo, basta criar
   `conf/model_4.py`.
3. **`models/resunet.py`** — **espinha dorsal da rede**.
   - `ResUnetEncoder` → 4 níveis (32 → 64 → 128 → 256 canais).
   - `ResUnetDecoder` → mesmos níveis em reverso com `Upsample` e
     concatenação de skip connections.
   - `ResUnetClassifier` → conv 1×1 para `N_CLASSES` + `Softmax`.
   - `ResidualBlock` em `models/layers.py` é o tijolo básico (BN→ReLU→Conv×2 +
     atalho identidade).
4. **`utils/dataloader.py`** → `PatchFileDataset`:
   - `__init__` lê `manifest.csv` e filtra pelo `split`.
   - `__getitem__` carrega `.npy` (ou cache RAM), normaliza `/255`, aplica
     data augmentation (rotações 90° + flips).
5. **`utils/trainer.py`**:
   - `train_loop` / `val_loop`: barra de progresso `tqdm`, soma loss e F1
     por batch, devolve médias.
   - `EarlyStop.testEpoch`: lógica de paciência + checkpoint.
6. **`conf/general.py`** — hiperparâmetros globais (`MAX_EPOCHS`, `LEARNING_RATE`,
   `CLASSES_WEIGHTS`, `EARLY_STOP_*`, `N_CLASSES`, `PATCH_SIZE`…).

### O que você pode controlar
| Variável | Onde mudar | Efeito |
|---|---|---|
| Tamanho do batch | flag `-b` | mais batch → GPU mais cheia, menos batches por época |
| Augmentation on/off | flag `-a` | `False` desliga rotações/flips |
| Cache RAM | flag `--cache-patches` | leitura uma vez do disco |
| LR inicial | `conf/general.py:LEARNING_RATE` | passos do otimizador |
| Pesos das classes | `conf/general.py:CLASSES_WEIGHTS` | enfatizar leucaena |
| Patience | `conf/general.py:EARLY_STOP_PATIENCE` | quão "paciente" o early stop é |
| Máx épocas | `conf/general.py:MAX_EPOCHS` | teto |
| Arquitetura | `conf/model_<e>.py` + `models/resunet.py:depths` | mais/menos camadas e largura |

### Comando padrão

```bash
python train.py -e 1 -b 16 --patch-source file --cache-patches
```

### O que é "uma época"?
Uma passada por **todos** os patches do split `train` (3.951 no seu caso).
Com batch 16 isso dá ≈ 247 iterações. Em cada iteração: forward → loss →
backward → `optimizer.step()`.

---

## 3. Parte 3 — Logs, métricas e gráficos

### Arquivos gerados (em `experiments/exp_<e>/logs/`)
| Arquivo | Conteúdo |
|---|---|
| `metrics.csv` | uma linha por época (loss, F1, LR, melhor val, etc.) |
| `training_config.json` | hiperparâmetros do run |
| `train_<e>.txt` | cópia do que apareceu no terminal |

### Quem escreve isso
- **`utils/training_log.py` → `MetricsLogger`** — abre `metrics.csv`, escreve
  `training_config.json` e dá `flush()` por época.
- `train.py` chama `metrics_logger.log_epoch(...)` no fim de cada época.

### Como gerar gráficos
**Script:** `utils/plot_training.py`

```bash
python -m utils.plot_training -e 1                  # todas as épocas
python -m utils.plot_training -e 1 --upto-epoch 10  # só até a 10ª (parcial)
python -m utils.plot_training -e 1 --table          # só tabela no terminal
```

Salva `experiments/exp_<e>/logs/training_curves.png` com 3 subplots: loss
(train vs val), F1 (train vs val) e learning rate.

### O que olhar
- **val loss caindo + train e val próximos** → bom.
- **train loss caindo e val subindo** → overfitting; aumentar augmentation,
  diminuir modelo, mais dados.
- **F1 estagnado em ~0,5** → pode ser desbalanceamento ou bug nos dados.

---

## 4. Parte 4 — Predição visual em tile real

### Ponto de entrada
- **Script:** `predict-tile-preview.py` (na raiz)

### O que ele faz
1. Carrega `experiments/exp_<e>/models/model.pt`.
2. Lê uma `.tif` (4 bandas) e opcionalmente recorta o centro
   (`--max-side 4096`) por velocidade.
3. Desliza janelas 256×256 (overlap 0,5), faz forward na GPU, **acumula a
   probabilidade** em cada pixel (em sobreposição usa média).
4. `argmax` → classe 0/1.
5. Se `--masks` foi passado, rasteriza polígonos no mesmo bbox e calcula
   **acurácia** e **F1** (macro e por classe), usando o mesmo
   `rasterize_geojson_for_tile()` do prep.
6. Salva PNGs em `experiments/exp_<e>/predicted/`:
   - `preview_<tile>_rgb.png` — só o RGB stretch
   - `preview_<tile>_pred_class.png` — máscara 0/1
   - `preview_<tile>_pred_overlay.png` — RGB com leucaena vermelha
   - `preview_<tile>_triptych.png` — RGB | label | predição

### Comando

```bash
python predict-tile-preview.py -e 1 \
  --tile /data/rgbir/SF-23-Y-A-IV-2-SE-C.tif \
  --masks /data/masks/polygons.geojson \
  --band-order RGBN -b 16
```

Tile inteiro: `--max-side 0` (lento e exige mais RAM).

---

## 4.b Parte 4b — Predição em escala (tile-by-tile)

Quando o `predict-tile-preview.py` é só "uma olhada num tile", e o
`prediction.py` precisaria mosaicar o AOI inteiro em RAM, entra o
`predict-tiles.py`: itera tile-a-tile, grava `_pred.tif` + `_prob.tif`
georreferenciados num caminho local (fora do OneDrive), e monta um `.vrt`
no fim para o QGIS abrir como mosaico.

### Ponto de entrada
- **Script:** `predict-tiles.py`
- **Helpers:** `utils/inference.py` (`predict_tile_probability`,
  `read_tile_as_bgrn`, `read_lidar_as_array`, `write_class_geotiff`,
  `write_prob_geotiff`)

### Comando padrão

```bash
# default: todos os tiles, overlaps [0, 0.25, 0.5], prob em uint16 (2 bytes/px)
python predict-tiles.py -e 1

# preview rápido em 3 tiles, overlap único
python predict-tiles.py -e 1 --max-tiles 3 --overlap 0

# fusion com LiDAR; tiles sem LiDAR usam zeros + log
python predict-tiles.py -e 3 --lidar-dir /data/lidar
```

Detalhes técnicos (por que uint16, por que [0, 0.25, 0.5], como fica o
manifest, etc.) → `studies/predicao-em-escala.md`.

---

## 5. Parte 5 — LiDAR

### Como o LiDAR entra no projeto

O LiDAR **não** é um `.las` cru. O projeto espera que você **pré-processe**
seus pontos LiDAR em **rasters GeoTIFF** (CHM = Canopy Height Model, INTENSITY,
etc.), alinhados pixel-a-pixel ao óptico. Ferramentas comuns para isso: PDAL,
LAStools, CloudCompare, WhiteboxTools (mencionado em
`conf/paths.py:PATH_LIDAR`).

Resumo:

```
nuvem.las  ─►  ferramenta GIS  ─►  lidar.tif   (multi-band)
                                       │
                                       ▼
                              mesmo grid do óptico
```

### Caminho atual (legacy) — funciona

- **Script:** `prep-data.py` (cena única) carrega `optical.tif` **e** `lidar.tif`,
  rasteriza máscara, fatia em patches e gera `prepared/{opt_img,lidar_img,label_train,label_test}.npy`
  + arrays de índices `train_patches.npy` etc.
- **Treino:** `python train.py -e 2 -b 8` (sem `--patch-source file`) usa
  `TreeTrainDataSet`, que devolve `(opt, lidar)` reais.
- **Modelos com LiDAR:**
  - `conf/model_2.py` → `ResUnet` (early fusion: concatena bandas óptico+lidar).
  - `conf/model_3.py` → `LateFusion` (encoders separados, fusão no decoder).

### Caminho tile-based (atual) — **ainda sem LiDAR**

`prep-patches-from-tiles.py` hoje só grava `opt/` e `lbl/`. O `PatchFileDataset`
inventa um tensor de zeros para o LiDAR só para manter a "contratada"
`(opt, lidar)`. Por isso `model_1.py` funciona, `model_2.py` / `model_3.py`
não funcionariam direito com `--patch-source file`.

### O que falta para usar LiDAR no caminho tile-based
1. `prep-patches-from-tiles.py` precisaria ler um segundo raster por tile
   (`lidar/<stem>.tif`), recortar as mesmas janelas e salvar em
   `prepared/patches/lidar/<id>.npy`.
2. `PatchFileDataset` precisa carregar essa banda e devolver no lugar do
   tensor de zeros.
3. Treinar com `-e 2` ou `-e 3`.

Isso está planejado no `plans/04-tile-based-part2-predict-scale-lidar.md` mas
**não foi implementado** ainda. Hoje, se você quiser testar com LiDAR, use o
**caminho legacy** (`prep-data.py` + `train.py` sem `--patch-source file`).

---

## 6. Caminho legacy (cena única) — quando ele aparece?

Esses três scripts existem desde a base original e ainda funcionam:

| Script | Quando usar |
|---|---|
| `prep-data.py` | Você tem **uma** cena/VRT e quer rodar tudo num shot (suporta LiDAR). |
| `prediction.py` | Roda inferência **no full scene preparado**. Não funciona com tiles separados. |
| `evaluation.py` | Compara `pred.npy` com `label_test.npy` (saída do prep-data). |

Para o seu caso atual (várias tiles na AOI de Piracicaba) **use o caminho
tile-based** (parte 1 deste guia) + `predict-tile-preview.py`. O legacy
ainda serve se você fundir tudo num VRT ou no futuro pré-processar LiDAR.

---

## 7. Cheatsheet de "abrir os arquivos certos"

### Quero entender de onde vêm os patches
1. `prep-patches-from-tiles.py` (docstring + `main()`)
2. `utils/ops.py: rasterize_geojson_for_tile`
3. `conf/general.py: PATCH_SIZE, PATCH_OVERLAP`

### Quero entender como os dados chegam na GPU
1. `train.py` (bloco do `PatchFileDataset(...)`)
2. `utils/dataloader.py: PatchFileDataset.__getitem__`
3. `torch.utils.data.DataLoader` (PyTorch — não está no repo, é biblioteca)

### Quero entender a rede em si
1. `conf/model_1.py: get_model()`
2. `models/resunet.py: ResUnetOpt → Encoder → Decoder → Classifier`
3. `models/layers.py: ResidualBlock`

### Quero entender o loop de treino e a parada
1. `train.py` (loop `for t in range(MAX_EPOCHS):`)
2. `utils/trainer.py: train_loop, val_loop, EarlyStop`

### Quero entender métricas / gráficos
1. `utils/training_log.py: MetricsLogger`
2. `utils/plot_training.py`

### Quero ver na prática se a rede aprendeu
1. `predict-tile-preview.py`
2. Abrir os PNGs em `experiments/exp_<e>/predicted/`

---

## 8. Ordem prática para um run "do zero"

```bash
# 0. dentro do container
docker compose run --rm segmentation bash

# 1. gerar patches (uma vez por dataset)
python prep-patches-from-tiles.py --band-order RGBN

# 2. treinar (acompanhar no terminal + metrics.csv)
python train.py -e 1 -b 16 --patch-source file --cache-patches

# 3. plotar curvas
python -m utils.plot_training -e 1

# 4. ver predição num tile real (com métricas)
python predict-tile-preview.py -e 1 \
  --tile /data/rgbir/SF-23-Y-A-IV-2-SE-C.tif \
  --masks /data/masks/polygons.geojson \
  --band-order RGBN -b 16
```

---

## 9. Glossário rápido

- **Patch:** janela 256×256 da imagem, é o que a rede vê de cada vez.
- **Tile:** o GeoTIFF "grande" original (ex.: 27k×19k).
- **CRS:** sistema de coordenadas (4326 = lat/lon graus; 31983 = SIRGAS UTM 23S em metros).
- **Manifest:** o `manifest.csv` lista quais patches existem e em qual split estão.
- **Skip connection:** atalho entre encoder e decoder que preserva detalhes finos (U-Net).
- **Softmax:** transforma a saída de N canais em probabilidades que somam 1.
- **CrossEntropyLoss:** distância entre probabilidades preditas e o pixel verdadeiro.
- **F1:** média harmônica entre precision e recall (0–1, maior é melhor).
- **Early stop:** parar de treinar quando o val não melhora há X épocas.
- **Checkpoint (`model.pt`):** pesos salvos quando o val melhora.

---

## 10. O que ainda não está pronto (plans/)

Veja `plans/04-tile-based-part2-predict-scale-lidar.md`:
- ~~predição tile-by-tile sem mosaicar~~ → **feito** (`predict-tiles.py`,
  detalhes em `studies/predicao-em-escala.md`)
- ~~LiDAR no pipeline tile-based~~ → **feito** (`prep-lidar-rasters.py`,
  `prep-patches-from-tiles.py --lidar-dir`)
- empacotar patches em HDF5/Zarr para datasets muito grandes (próximo)
- splits por tile (e não por patch) para evitar vazamento entre train/test
- `evaluation.py` tile-by-tile (hoje só funciona no caminho legado)

Quando você quiser atacar isso, dê o ping para um agente — esse guia
continua válido como base.
