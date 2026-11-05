# Runner PhD (pool de tiles + AOI vetorial + registry)

Um motor (`02_train.py --arch ...`) para todas as redes, configurado por
`experiments/<rede>/<nome>/experiment.yaml` sob `PHD_RUN_ROOT`.

Wrappers na **raiz do repo**: `run.ps1` / `run.cmd` (preferir `.cmd` no lab).

## PowerShell (lab / home)

Rode **sempre no PowerShell** a partir da raiz do repo. No lab a ExecutionPolicy
costuma bloquear `.ps1` e o Windows pode abrir o arquivo no editor — use o `.cmd`
(ou Bypass explícito):

```powershell
cd C:\Users\CMQ\Documents\MSB\0-GITHUB\leucaena-earth-segmentation   # lab
# cd <seu-clone>                                                      # home

# Preferido no lab:
.\run.cmd labels -aoi 261002-riodaspedras -experiment manet/latefusion-multitask -run run-001

# Se precisar chamar o .ps1 direto:
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 labels -aoi 261002-riodaspedras -experiment manet/latefusion-multitask -run run-001
```

Ordem típica (`all` já roda LiDAR python + 1–4):

```powershell
.\run.cmd all -aoi 261002-riodaspedras -experiment manet/latefusion-multitask -run run-001
```

Ou por fase (LiDAR python é o default de `.\run_lidar.cmd`):

1. `.\run_lidar.cmd -aoi ...`   (ou incluso no `all`)
2. `.\run.cmd labels  ... -run run-001`
3. `.\run.cmd train   ... -run run-001`
4. `.\run.cmd predict ... -run run-001`
5. `.\run.cmd post    ... -run run-001`

## .env (por PC)

```dotenv
PHD_RUN_ROOT=E:/dados-msb/00_leucaena-earth-experiments   # lab
# PHD_RUN_ROOT=C:/00_PHD_MSB_RUN                           # home
TRAIN_PROFILE=lab   # home | lab  (batch/workers via profiles no experiment.yaml)
R_BIN=C:/Program Files/R/R-4.6.1/bin/Rscript.exe
```

## Antes do treino: LiDAR do AOI

O MANet late-fusion precisa de `tiles/lidar/rasters/<tile>.tif` `[CHM, INTENSITY]`.

```powershell
cd <repo>
.\run_lidar.cmd -aoi 261002-riodaspedras                 # Engine python (default, PDAL)
.\run_lidar.cmd -aoi 261002-riodaspedras -Engine r       # lidR (TIN + lspikefree)
```

Debug / engines / buracos TIN: ver
[`scripts/lidar-products/README.md`](../lidar-products/README.md).

## Como rodar (raiz do repo)

No lab use **`.cmd`** (evita abrir `.ps1` no editor e a ExecutionPolicy):

```powershell
cd <repo>

# proc 1 - rotulos
.\run.cmd labels  -aoi 261002-riodaspedras -experiment manet/latefusion-multitask -run run-001

# proc 2 - treino
.\run.cmd train   -aoi 261002-riodaspedras -experiment manet/latefusion-multitask -run run-001

# proc 3 - predicao (mesma ou outra AOI; reusa pesos, NAO cria run novo)
.\run.cmd predict -aoi 261002-riodaspedras -experiment manet/latefusion-multitask -run run-001

# proc 4 - pos-processo
.\run.cmd post    -aoi 261002-riodaspedras -experiment manet/latefusion-multitask -run run-001

# tudo de uma vez
.\run.cmd all     -aoi 261002-riodaspedras -experiment manet/latefusion-multitask -run run-001

# relatorio agregado
.\run.cmd report
```

Equivalente se precisar forçar Bypass:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 all -aoi 261002-riodaspedras -experiment manet/latefusion-multitask -run run-001
```

## Onde cai o resultado

```text
PHD_RUN_ROOT/
  datasets/<aoi>__lbl-v1/masks/
  experiments/manet/latefusion-multitask/run-001/
    models/
    predictions/<aoi>/    # pred_class.tif, pred_dist.tif, pred_dist_bin_*.tif
    run.yaml
  reports/summary.md  summary.csv
```

## Docker

- Engine running basta; o runner sobe o container com `docker compose run --rm`.
- Imagem: `leucaena-segmentation:cuda`

```powershell
docker images leucaena-segmentation:cuda
# se faltar:
docker build -t leucaena-segmentation:cuda .
```

Compose: `scripts/phd-run/docker-compose.yml`  
Mounts: `/data/tiles`, `/data/aois`, `/data/annotations`, `/data/datasets`, `/data/experiments`, `/data/reports`.

## Experimentos semeados

| `-experiment` | rede | tarefa |
|---|---|---|
| `manet/latefusion-multitask` | manet | multitask |
| `resunet/latefusion` | resunet | classification |

Templates YAML: `scripts/phd-run/templates/`.

## Debug (labels / train)

### `Writing through VRTSourcedRasterBand is not supported`

Causa: o `01_build_labels.py` lia o mosaico AOI (`.vrt`) e copiava `driver=VRT`
para a saída. GDAL não grava em VRT.

Fix (já no repo): saídas de label/distância forçadas para **GeoTIFF**.
Atualize e rode de novo:

```powershell
git pull
.\run.cmd labels -aoi 261002-riodaspedras -experiment manet/latefusion-multitask -run run-001
```

### Warning `More than one layer found in 'annotations.gpkg'`

Inofensivo se a layer default for `annotations`. O resolve usa essa layer.

### Docker / imagem

```powershell
docker images leucaena-segmentation:cuda
# engine running basta; o runner sobe o container sozinho
```
