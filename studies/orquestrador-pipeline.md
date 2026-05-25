# Orquestrador do Pipeline de Segmentação

## O que é

`run_pipeline.py` é o ponto de entrada único que automatiza as cinco
etapas do fluxo de dados — da cópia dos arquivos brutos até o treinamento do
modelo — com um único comando.

```powershell
python run_pipeline.py `
  --aoi "G:\My Drive\PHD\02-Tese\02-data\adote-uma-leucena\v1-LEUCENA MAPPING\gdb-leucena_v2.gpkg" `
  --layer articulacao_laser_voo22_AOI_treino `
  --source D:\ `
  --dest "C:\00_DATASETS_AI\260515-piracicaba-aoi" `
  --build-ovr `
  --train
```

---

## Estrutura de destino (`--dest`)

```
C:\00_DATASETS_AI\260515-piracicaba-aoi\
│
├── annotations\           ← GeoJSON / GPKG com polígonos de anotação (fornecido pelo usuário)
│
├── lidar\
│   ├── raw\               ← .laz copiados (saída etapa 1)
│   └── chm\               ← GeoTIFFs CHM + Intensidade (saída etapa 3)
│
├── opt\
│   ├── raw\
│   │   ├── rgb\           ← tiles RGB copiados (saída etapa 1)
│   │   └── ir\            ← tiles IR copiados (saída etapa 1)
│   └── rgbnir\            ← tiles 4 bandas RGBNIR (saída etapa 2)
│
├── patches\               ← patches opt + lidar + máscara (saída etapa 4)
│   ├── opt\
│   ├── lbl\
│   └── manifest.csv
│
├── models\
│   └── logs\
│       └── pipeline.log   ← log de cada execução do orquestrador
│
└── .pipeline_cache\
    └── tile_index.json    ← cache do índice de arquivos em D:\
```

---

## Etapas

### Etapa 1 — Cópia de Tiles (`prep-copy-tiles-from-aoi.py`)

| Parâmetro | Valor                         |
|-----------|-------------------------------|
| Entrada   | `D:\rgb`, `D:\ir`, `D:\laz`   |
| Seleção   | tile IDs lidos do GeoPackage AOI |
| Saída     | `dest\opt\raw\rgb`, `dest\opt\raw\ir`, `dest\lidar\raw` |

- Resume-safe: arquivos já existentes são ignorados por padrão.
- `--overwrite` força re-cópia.
- `--dry-run` imprime o que seria copiado sem tocar no disco.

### Etapa 2 — Fusão RGBNIR (`prep-rgbnir-from-rgb-ir.py`)

| Parâmetro | Valor                         |
|-----------|-------------------------------|
| Entrada   | `dest\opt\raw\rgb` + `dest\opt\raw\ir` |
| Saída     | `dest\opt\rgbnir\<TILE_ID>.tif` |
| Bandas    | 1=R, 2=G, 3=B, 4=NIR          |

- Processa tile por tile em janelas (sem carregar o raster inteiro na RAM).
- Paralelo com `ProcessPoolExecutor` (`--workers`, default 4).
- DEFLATE + PREDICTOR 2 + tiles de 512×512.

### Etapa 2b — Overviews (opcional, `--build-ovr`)

- `gdaladdo` com níveis `2 4 8 16 32 64` para todos os `.tif` em `opt\rgbnir\`.
- Executa dentro do container Docker (usa o GDAL já instalado).
- Arquivos `.ovr` ficam ao lado dos `.tif`; QGIS os detecta automaticamente.

### Etapa 3 — LAZ → CHM (`prep-lidar-rasters.py`, Docker/PDAL)

| Parâmetro | Valor                          |
|-----------|--------------------------------|
| Entrada   | `dest\lidar\raw\*.laz`         |
| Ref. opt. | `dest\opt\rgbnir` (para alinhar) |
| Saída     | `dest\lidar\chm\<TILE_ID>.tif` (2 bandas: CHM, Intensidade) |

- Requer Docker com `leucaena-segmentation:cuda` construída.

### Etapa 4 — Geração de Patches (`prep-patches-from-tiles.py`, Docker/GDAL)

| Parâmetro | Valor                          |
|-----------|--------------------------------|
| Entrada   | `dest\opt\rgbnir`, `dest\lidar\chm`, `dest\annotations\leucaena.geojson` |
| Saída     | `dest\patches\` + `manifest.csv` |

- Aplica a lógica de labelagem refinada (CHM > 4.5 m + NDVI > 0.3 dentro dos polígonos).

### Etapa 5 — Treinamento (`train.py`, Docker/GPU)

| Parâmetro | Valor                     |
|-----------|---------------------------|
| Entrada   | `dest\patches\manifest.csv` |
| Saída     | `experiments\exp_<N>\`    |

- Ativado com `--train`.
- Número do experimento via `--experiment N` (default 1).

---

## Argumentos completos

```
python run_pipeline.py --help
```

| Flag              | Descrição                                                |
|-------------------|----------------------------------------------------------|
| `--aoi`           | Caminho do GeoPackage (obrigatório)                      |
| `--layer`         | Nome da camada AOI (obrigatório)                         |
| `--dest`          | Raiz do dataset de destino (obrigatório)                 |
| `--source`        | Raiz dos dados brutos, default `D:\`                     |
| `--id-column`     | Coluna tile-ID (auto-detectada se omitida)               |
| `--annotations`   | Caminho do GeoJSON de anotações (default `dest\annotations\leucaena.geojson`) |
| `--experiment`    | Número do experimento para train.py (default 1)          |
| `--workers`       | Workers paralelos para etapa 2 (default 4)               |
| `--steps`         | Etapas a executar, ex. `1,2` ou `3,4,5` (default todas) |
| `--build-ovr`     | Gerar `.ovr` após etapa 2                                |
| `--train`         | Incluir etapa 5 (treinamento)                            |
| `--overwrite`     | Re-processar saídas já existentes                        |
| `--dry-run`       | Imprimir comandos sem executar nada                      |
| `--rebuild-index` | Forçar re-scan de `--source` (ignora cache)              |
| `--verbose`       | Logging DEBUG                                            |

---

## Exemplos de uso

### Pipeline completo

```powershell
python run_pipeline.py `
  --aoi "G:\My Drive\PHD\...\gdb-leucena_v2.gpkg" `
  --layer articulacao_laser_voo22_AOI_treino `
  --source D:\ `
  --dest "C:\00_DATASETS_AI\260515-piracicaba-aoi" `
  --build-ovr --train
```

### Dry-run para checar antes de rodar

```powershell
python run_pipeline.py `
  --aoi "G:\My Drive\PHD\...\gdb-leucena_v2.gpkg" `
  --layer articulacao_laser_voo22_AOI_treino `
  --source D:\ --dest "C:\00_DATASETS_AI\260515-piracicaba-aoi" `
  --dry-run --verbose
```

### Rodar só as etapas 1 e 2 (cópia + fusão)

```powershell
python run_pipeline.py ... --steps 1,2
```

### Retomar a partir da etapa 3 (já copiou, só falta CHM + patches + treino)

```powershell
python run_pipeline.py ... --steps 3,4,5 --train
```

### Re-rodar tudo do zero

```powershell
python run_pipeline.py ... --overwrite --rebuild-index
```

---

## Arquitetura do pacote `pipeline/`

```
pipeline/
├── __init__.py        # pacote
├── layout.py          # DestLayout: todos os caminhos derivados de --dest
├── log.py             # setup_logging: console + arquivo pipeline.log
├── runners.py         # run_cmd + docker_run: subprocess helpers
└── tile_index.py      # TileIndex: cache JSON do índice de D:\
```

### `DestLayout`

Dataclass que deriva todos os sub-diretórios a partir de `--dest`.  
Usar `layout.opt_rgbnir`, `layout.lidar_chm`, etc. garante que nenhum
caminho fica hardcoded no orquestrador.

### `TileIndex`

Lista `D:\rgb`, `D:\ir`, `D:\laz` uma única vez e salva em
`dest\.pipeline_cache\tile_index.json`.  Invocações seguintes (retomada,
dry-run) reusam o cache sem re-escanear o disco.

---

## Pré-requisitos

### Etapas 1 e 2 (nativas, sem Docker)
```
pip install geopandas rasterio
```

### Etapas 3, 4, 5 (Docker)
```powershell
docker compose build   # uma vez, na raiz do repositório
```

A imagem `leucaena-segmentation:cuda` deve existir antes de rodar as etapas
3–5. O orquestrador verifica isso automaticamente e aborta com mensagem clara
se a imagem não for encontrada.

---

## Decisões de design

| Decisão | Motivo |
|---------|--------|
| Etapas 1+2 nativas (sem Docker) | Geopandas + rasterio rodam bem no Windows; sem overhead de container para operações IO-bound |
| Etapas 3–5 via `docker run` (não `docker compose run`) | Permite montar volumes arbitrários sem depender do `.env` ou de paths WSL |
| Cache de tile index (`TileIndex`) | Evitar re-scan de `D:\` (potencialmente 10k+ arquivos) em cada invocação |
| `DestLayout` centralizado | Mudança de estrutura de pastas requer alterar apenas um lugar |
| `--steps` para seleção granular | Permite retomar o pipeline sem re-processar etapas já concluídas |
| Scripts originais não duplicados | `run_pipeline.py` chama `prep-*.py` via subprocess, reusando 100% da lógica existente |
