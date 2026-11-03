# Orquestrador de produtos LiDAR (abordagem C)

Encadeia **R (lidR)** → **Python (alinhamento ao RGBNIR)** para gerar o raster
LiDAR de 2 bandas `[CHM, INTENSITY]` que o treino consome, usando o CHM do
`lidR` (ground CSF/classe 2 + `lspikefree`), que é mais fiel que o CHM antigo
feito só com PDAL (DTM = mínimo global).

## Arquivos

| Arquivo | Onde roda | O que faz |
|---|---|---|
| `make_lidar_products.R` | **host** (R + lidR) | Por `.laz`: `MDT`, `MDS`, `CHM` em `lidar/products/`. Pula o que já existe. |
| `align_lidar_to_optical.py` | **Docker** (GDAL+PDAL) | Mosaica os `*_CHM.tif`, alinha ao RGBNIR (ou grade nativa), gera `[CHM, INTENSITY]`. |
| `run_lidar_pipeline.ps1` | **host** | Orquestra os dois passos. |
| `docker-compose.yml` | — | Monta repo + AOI na imagem `leucaena-segmentation:cuda`. |

## Os "mundos" (parâmetro `-Mode`)

```powershell
# Na RAIZ do repo (preferido):
cd <repo>
.\run_lidar.ps1 -aoi 261002-riodaspedras
.\run_lidar.ps1 -aoi 261002-riodaspedras -Mode products
.\run_lidar.ps1 -aoi 261002-riodaspedras -Mode align
.\run_lidar.ps1 -aoi 261002-riodaspedras -Force
.\run_lidar.ps1 -aoi 261002-riodaspedras -NoIntensity

# Equivalente direto (scripts/lidar-products):
.\run_lidar_pipeline.ps1 -aoi 261002-riodaspedras
```

O AOI (`aois/<aoi>.gpkg`) define os tiles via `NOMENC_2K/5K/10K`. O script
resolve quais `.laz` do **pool** pertencem a esses tiles e passa `--names` /
`--laz-names` — **sem pasta temporária**.

No modo `all`, se os `*_CHM.tif` do AOI já existirem, o passo R é pulado
(use `-Force` para refazer).

## Entradas e saídas (pool em PHD_RUN_ROOT)

- LAZ: `tiles/lidar/raw/*.laz`
- Produtos R: `tiles/lidar/products/<laz>_{MDT,MDS,CHM}.tif`
- Óptico: `tiles/rgbnir/<tile>.tif` (tiles do AOI)
- **Saída final**: `tiles/lidar/rasters/<tile>.tif` (2 bandas `[CHM, INTENSITY]`)

## .env (raiz do repo)

```dotenv
PHD_RUN_ROOT=C:/00_PHD_MSB_RUN
R_BIN=C:/Program Files/R/R-4.6.1/bin/Rscript.exe   # se Rscript nao estiver no PATH
```

## Pré-requisitos

- **Host**: R + pacotes `lidR`, `terra` (e `RCSF` se usar `--csf`).
- **Docker**: imagem `leucaena-segmentation:cuda` (mesma do treino).
