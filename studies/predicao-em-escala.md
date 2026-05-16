# Predição em escala — `predict-tiles.py`

Este documento explica **o que foi feito** na seção 3A do plano
`plans/04-tile-based-part2-predict-scale-lidar.md`, **por que** cada decisão
foi tomada e **como** você roda o script no dia a dia.

> Pré-requisitos: ter um `experiments/exp_<N>/models/model.pt` treinado e uma
> pasta de tiles RGBN (definida no `.env` via `LEUCAENA_TILES_DIR`).

---

## 1. O problema

`prediction.py` (script legado) precisa que todo o seu AOI esteja **em uma
única cena** (`.npy` ou VRT mosaicado). Ele faz mais ou menos isso:

```python
img = np.load("opt_img.npy")        # ~ (4000, 4000, 4) na AOI de Piracicaba
pad = np.pad(img, PATCH_SIZE)        # crescer para reflect padding
dataset = TreePredDataSet(...)       # extrai todas as janelas
for overlap in [0, 0.25, 0.5]:
    preds = roda_a_rede_em_tudo()
    media += sum(preds) / count
salva_geotiff(media.argmax(-1))
```

Funciona, **mas todo o AOI vive em RAM**. Para Piracicaba (4000×4000 px) cabe.
Para 50 cartas IBGE de 1 km × 1 km a 25 cm/pixel (200 GB de óptico bruto),
não cabe.

A solução nada criativa, padrão da indústria: **processar tile-a-tile**. Cada
tile já é georreferenciado; basta gerar `pred.tif` por tile e juntar via
`gdalbuildvrt` no final — o QGIS abre o `.vrt` como se fosse um único raster
contínuo.

---

## 2. O que o script faz

```
┌─────────────────────────────────────────────────────────────────┐
│ for cada tile em $TILES_DIR:                                    │
│   1. opt  = read_tile_as_bgrn(tile)        # (H, W, 4) uint8    │
│   2. lidar = read_lidar(stem)              # ou zeros + log     │
│   3. prob_hwc = predict_tile_probability(opt, lidar,            │
│        overlaps=[0, 0.25, 0.5], batch_size=128)                 │
│   4. write_class_geotiff(stem + "_pred.tif", argmax(prob))      │
│   5. write_prob_geotiff (stem + "_prob.tif", prob[:,:,1],       │
│        dtype=uint16)                                            │
│   6. log no manifest.csv                                        │
│                                                                 │
│ no fim:                                                         │
│   gdalbuildvrt pred.vrt *_pred.tif                              │
│   gdalbuildvrt prob.vrt *_prob.tif                              │
└─────────────────────────────────────────────────────────────────┘
```

Os passos 1–5 são exatamente o **mesmo cálculo** que `prediction.py` faz, só
que aplicado **por tile** em vez do AOI inteiro. As probabilidades são
calculadas no espaço normalizado (`opt_uint8/255`, LiDAR via `scale_lidar`
idêntico ao do `PatchFileDataset`), os overlaps são averageados, e o
`argmax` é tirado no final. Nada mudou no algoritmo de inferência — só na
estratégia de I/O.

A reflect-padding por `PATCH_SIZE` em cada lado garante que **todo pixel
real** vê o mesmo número de janelas, então os bordos não saem mais ruins
que o miolo. Sem isso, um tile que termina no canto da AOI teria a borda
direita vista por menos overlaps.

---

## 3. As 4 decisões de design (as suas perguntas)

### 3.1 Onde gravar a saída → **local, fora do OneDrive**

O OneDrive sincroniza o repo inteiro. Uma rodada de inferência em ~50 tiles
4000×4000 produz ~100 GB de GeoTIFFs (mesmo com DEFLATE+PREDICTOR) e iria
emperrar o OneDrive em minutos.

Solução: **caminho de saída controlado pelo `.env`**:

```env
LEUCAENA_PREDICTIONS_HOST_DIR=/mnt/d/leucaena-predictions   # disco local D:
LEUCAENA_PREDICTIONS_DIR=/data/predictions                   # dentro do container
```

`docker-compose.yml` monta o primeiro no segundo; `conf/paths.py` lê
`LEUCAENA_PREDICTIONS_DIR` e define `PATH_PREDICTIONS_DIR`; `predict-tiles.py`
usa esse caminho como default (sub-pasta `exp_<N>/`).

**Para você** (Windows): `D:\leucaena-predictions` é a pasta real. Aparece no
WSL como `/mnt/d/leucaena-predictions`. Pode também ser um SSD externo.

### 3.2 Tipo do TIF de predição → **os dois (`--save-prob` ligado por default)**

| Saída | dtype | bytes/px | Para que serve |
|---|---|---|---|
| `_pred.tif` | uint8 (0/1) | 1 | Mapa final, QGIS abre direto, análise visual |
| `_prob.tif` | uint16 (default) | 2 | **Análise científica**: ROC, threshold sweeps, incerteza |

A probabilidade é o que vale para tese — o argmax é só conveniência. Se
você precisar comparar threshold ≠ 0.5, calcular IoU vs threshold, ou
treinar um modelo "stacker" em cima, precisa do prob.

#### Sobre a pergunta "não dá pra usar float16 ao invés de float32?"

**Resposta curta**: não, mas dá uma coisa equivalente (e melhor).

**Resposta longa**: o GeoTIFF (e por extensão GDAL) **não tem `GDT_Float16`
nativo**. Os tipos de pixel suportados são:

```
Byte (uint8)    UInt16    Int16
UInt32          Int32     Float32     Float64
CInt16   CInt32   CFloat32   CFloat64
```

Existe um *workaround* (BitsPerSample=16 + SAMPLEFORMAT_IEEEFP) que alguns
softwares leem como Float16, mas:

1. **GDAL não escreve esse formato** — você teria que escrever bytes na mão.
2. **QGIS não lê confiavelmente** — pode aparecer como NaN ou inválido.
3. **Não é portátil entre rasterio / numpy / xarray**.

A solução padrão da comunidade científica (CMIP, ESA Sentinel, NASA SDAP) é
**quantizar para uint16 com `scale_factor`**:

```python
arr = (prob * 65535).astype(np.uint16)         # 2 bytes por pixel
band.SetScale(1 / 65535)                        # GDAL anota o fator
# QGIS / rasterio / xarray lêem como float [0, 1] automaticamente
```

Isso te dá **a mesma economia de espaço** que float16 (2 bytes/pixel ao invés
de 4), **mais precisão útil** que float16 (65 536 níveis lineares em [0,1]
versus ~1 000 níveis de float16 em [0,1], porque float16 desperdiça expoente
em valores fora de [0,1]) e funciona em **qualquer ferramenta**. Por isso o
default do `--prob-dtype` é `uint16`. Se você precisar do float bruto,
`--prob-dtype float32`.

```
float32  : 4 bytes/px  → 1 GB num tile de 16000×16000
uint16   : 2 bytes/px  → 0.5 GB ← default
uint8    : 1 byte/px   → 0.25 GB (256 níveis, só visual)
```

#### Como abrir no QGIS

Arrasta o `.vrt` ou um `_prob.tif`. Vai aparecer como float `[0, 1]` por
mágica — o `scale_factor` é aplicado na leitura. Symbology → Singleband
pseudocolor → min 0, max 1.

### 3.3 Overlaps de inferência → **`[0, 0.25, 0.5]` por padrão, `--overlap 0` para preview**

Os três valores são o default do `general.PREDICTION_OVERLAPS` (que
`prediction.py` também usa). Por que três?

- **0%**: rápido, janelas sem sobreposição. Borda fica visível na predição.
- **25%**: ameniza as bordas.
- **50%**: as bordas somem completamente (todo pixel é coberto por 4
  janelas em locais quaisquer).

Tirar a média das três probabilidades softmax dá uma predição que tira
proveito das três escalas de coverage — qualquer pixel que se beneficiaria
de uma escala extra de contexto vê 7 janelas no total. É barato (o tempo
extra é proporcional aos overlaps maiores) e suaviza muito a fronteira.

Para **preview** (sanity-check rápido):

```bash
python predict-tiles.py -e 1 --max-tiles 3 --overlap 0
```

Uma única passada com overlap 0 leva ~1/7 do tempo. As bordas vão estar
visíveis, mas o miolo do tile dá uma noção honesta de como o modelo está
indo.

### 3.4 Quando faltar LiDAR → **prediz com zeros + `lidar=missing` no manifest**

Cenário: você treinou `-e 3` (late fusion) com os tiles que tinham LiDAR
e quer rodar inferência em **todos** os tiles. Alguns não têm LiDAR
correspondente. Três alternativas:

1. **Pular o tile** — mas aí a saída fica buraco no mosaico. Ruim para
   apresentar mapas contínuos.
2. **Erro fatal** — para o pipeline inteiro num tile. Pior ainda.
3. **Prever com zeros + logar** ← **escolha** ↩
   O modelo recebe LiDAR de zeros (idêntico ao que o `PatchFileDataset`
   já faz no treino quando o `.npy` LiDAR não existe). A predição vai sair
   com um viés (porque você está mostrando para a rede um sinal que ela
   nunca viu em treino), mas você **sabe**: o `manifest.csv` carimba
   `lidar_status=missing` na linha desse tile.

No manifest dá pra filtrar:

```python
import pandas as pd
m = pd.read_csv("manifest.csv")
m[m.lidar_status == "missing"]   # tiles a refazer com LiDAR
```

E nada está escondido: o `predict_<e>.txt` imprime `lidar=missing -- predicting with zeros, decision logged` para cada caso.

---

## 4. Anatomia dos arquivos novos

| Arquivo | O que faz |
|---|---|
| `utils/inference.py` | I/O de tile (read RGBN/LiDAR), sliding-window c/ softmax averaging, escrita de `_pred.tif` e `_prob.tif` packed |
| `predict-tiles.py` | Loop principal: itera tiles, chama `predict_tile_probability`, grava manifest, monta VRT no fim |
| `conf/paths.py` | Adicionou `PATH_PREDICTIONS_DIR` (lê `LEUCAENA_PREDICTIONS_DIR` do env) |
| `docker-compose.yml` | Monta `$LEUCAENA_PREDICTIONS_HOST_DIR` em `/data/predictions` |
| `.env*` | Acrescentou `LEUCAENA_PREDICTIONS_HOST_DIR` e `LEUCAENA_PREDICTIONS_DIR` |

`utils/inference.py` é uma camada fina e reusável. Se um dia você quiser
fazer um Streamlit que prediz um upload do usuário, vai chamar
`predict_tile_probability` diretamente — não precisa do CLI.

---

## 5. Limitações conhecidas e próximos passos

- **Cada tile é carregado inteiro em RAM.** Para tiles >12 000×12 000 a 4
  bandas uint8 isso são ~ 0.5 GB só de óptico. Se o tile for muito grande,
  vai precisar quebrar internamente em sub-grades. Hoje não acontece — os
  tiles IBGE/IGC ficam abaixo disso.
- **Sem multi-GPU.** Single device, single process. Em multi-tile podemos
  trivialmente paralelizar com `xargs -P` chamando o script por subconjuntos
  de tiles (cada um com `--max-tiles` + `--tiles-glob` em pastas distintas);
  fica para depois.
- **Sem reuso da inferência entre overlaps.** Cada overlap roda uma passada
  separada. Já dá pra atalhar — mas o ganho seria 5-15% e complica o código.
- **`gdalbuildvrt` só funciona se os tiles tiverem o mesmo CRS.** Por
  construção (RGBN são todos da mesma malha IBGE/IGC) isso vale. Se um dia
  você quiser misturar grids, precisa rodar `gdalwarp` antes ou gerar um
  `.vrt` por CRS.

Quando esses limites incomodarem, abrir uma issue / outro plano. O design
atual é o caminho mais simples que **resolve a escala que você precisa
hoje** (estado de SP) sem inflar complexidade.

---

## 6. Como verificar que ficou OK (smoke test)

```bash
# 1. dois tiles pequenos, overlap 0, sem prob
python predict-tiles.py -e 1 --max-tiles 2 --overlap 0 --no-save-prob

# 2. inspecionar
ls /data/predictions/exp_1/
cat /data/predictions/exp_1/manifest.csv
cat /data/predictions/exp_1/predict_1.txt

# 3. abrir o pred.vrt no QGIS — deve dar um mosaico de 2 tiles
```

Se isso roda em < 1 min e abre no QGIS sem buraco, o pipeline está OK.
