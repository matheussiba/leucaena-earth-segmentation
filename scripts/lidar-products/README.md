# Orquestrador de produtos LiDAR (abordagem C)

Encadeia **R (lidR)** → **Python/Docker (align)** para gerar o raster LiDAR
de 2 bandas `[CHM, INTENSITY]` que o treino consome.

O AOI (`aois/<aoi>.gpkg`) define os tiles via `NOMENC_2K / NOMENC_5K / NOMENC_10K`.
O script resolve quais `.laz` do **pool** pertencem a esses tiles e filtra por
`--names` / `--laz-names` — **sem pasta temporária**.

## Arquivos

| Arquivo | Onde roda | O que faz |
|---|---|---|
| `resolve_aoi_lidar.py` | host | AOI.gpkg → tiles ópticos + stems LAZ (NOMENC) |
| `make_lidar_products.R` | host (R + lidR) | Por `.laz`: `MDT` + `CHM` (MDS só com `--mds`) |
| `align_lidar_to_optical.py` | Docker (GDAL+PDAL) | Mosaica CHM, alinha ao RGBNIR, gera `[CHM, INTENSITY]` |
| `run_lidar_pipeline.ps1` | host | Orquestra resolve → R → align |
| `docker-compose.yml` | — | Monta `PHD_RUN_ROOT/tiles` em `/data/tiles` |
| `../../run_lidar.ps1` / `../../run_lidar.cmd` | raiz do repo | Wrappers (preferir o `.cmd` no lab) |

## Pré-requisitos

1. **`.env` na raiz do repo** (já configurado por PC):

```dotenv
PHD_RUN_ROOT=E:/dados-msb/00_leucaena-earth-experiments   # lab
# PHD_RUN_ROOT=C:/00_PHD_MSB_RUN                           # home
TRAIN_PROFILE=lab                                          # ou home
R_BIN=C:/Program Files/R/R-4.6.1/bin/Rscript.exe           # se Rscript nao estiver no PATH
```

2. **R** no host com pacotes `lidR`, `terra` (e `RCSF` se usar `--csf`).
3. **Docker Desktop** com engine running + imagem `leucaena-segmentation:cuda`.
4. Pool populado sob `PHD_RUN_ROOT` (ver layout abaixo).

## Como rodar (raiz do repo)

No lab, **use o `.cmd`** (o Windows costuma abrir `.ps1` no editor, e a
ExecutionPolicy do PowerShell bloqueia scripts):

```powershell
cd <repo>   # ex.: C:\Users\CMQ\Documents\MSB\0-GITHUB\leucaena-earth-segmentation

# Pipeline completo: R (MDT/MDS/CHM) + align [CHM, INTENSITY]
.\run_lidar.cmd -aoi 261002-riodaspedras

# So produtos R
.\run_lidar.cmd -aoi 261002-riodaspedras -Mode products

# So align (CHM ja existe)
.\run_lidar.cmd -aoi 261002-riodaspedras -Mode align

# Refazer mesmo com produtos existentes
.\run_lidar.cmd -aoi 261002-riodaspedras -Force

# Sem intensidade (banda 2 = zeros) / sem alinhar ao optico
.\run_lidar.cmd -aoi 261002-riodaspedras -NoIntensity
.\run_lidar.cmd -aoi 261002-riodaspedras -NoOptical

# Resolucao dos produtos R (default atual: 0.5 m). Ex.: 1 m
.\run_lidar.cmd -aoi 261002-riodaspedras -Res 1

# Paralelismo: 4 LAZ ao mesmo tempo (default). RAM alta -> baixe Workers.
.\run_lidar.cmd -aoi 261002-riodaspedras -Workers 4 -Threads 12
.\run_lidar.cmd -aoi 261002-riodaspedras -Workers 2 -Threads 12
```

Cada worker processa 1 LAZ; threads do `lidR` por worker ≈ `Threads/Workers`
(ex.: 4 workers × 3 threads).

Equivalente explícito se o `.cmd` não existir ainda:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_lidar.ps1 -aoi 261002-riodaspedras
```

Saída esperada do resolve (Rio das Pedras):

```text
optical: SF-23-Y-A-IV-4-SO-A
laz:     SF-23-Y-A-IV-4-SO-A-I .. -IV  (4 arquivos)
```

**TIF final:** `PHD_RUN_ROOT/tiles/lidar/rasters/SF-23-Y-A-IV-4-SO-A.tif`

## Layout do pool (`PHD_RUN_ROOT`)

```text
PHD_RUN_ROOT/
  aois/261002-riodaspedras.gpkg          # NOMENC_2K/5K/10K
  tiles/
    rgbnir/SF-23-Y-A-IV-4-SO-A.tif       # optico do AOI
    lidar/
      raw/SF-23-Y-A-IV-4-SO-A-*.laz      # LAZ do AOI (NOMENC_5K ou prefixo do tile)
      products/<laz>_{MDT,CHM}.tif       # saida do R (MDS opcional --mds)
      rasters/<tile>.tif                 # saida final [CHM, INTENSITY]
```

### Buracos no MDT (tile sem pontos de solo)

Se ~¼ da cena não tem classe 2, o `normalize_height` padrão do `lidR` tenta
**IDW** e pode demorar horas. O script **ignora** esses pontos (extract
bilinear do MDT; onde é NA, o ponto não entra no CHM). O buraco fica sem
altura (CHM=0) — melhor do que interpolar terreno fantasma e travar o R.

Se um LAZ ficou pela metade (só MDT/MDS, sem CHM), rode de novo com `-Force`
depois de atualizar o script (`git pull`).

### Paralelismo (vários LAZ)

Default: `-Workers 4 -Threads 12` → até 4 LAZ em paralelo. Se a RAM do PC
estourar (cada LAZ carrega a nuvem), use `-Workers 2`. Sequencial: `-Workers 1`.

## Resolução (onde alterar)

| Etapa | Script / parâmetro | Default | Efeito |
|---|---|---|---|
| Produtos R (MDT/MDS/CHM) | `run_lidar*.ps1` **`-Res`** → `make_lidar_products.R --res` | `0.5` | Resolução nativa dos produtos em `products/` |
| Intensidade (PDAL) | `align_lidar_to_optical.py --resolution` | `1.0` | Grade nativa da intensidade antes do warp |
| TIF final com óptico | grade do `tiles/rgbnir/<tile>.tif` | **0.25 m** | Align reamostra o CHM para a grade do RGBNIR |
| TIF final sem óptico | `-NoOptical` + `-Res` | = `-Res` | Mantém grade nativa do LiDAR |

Atenção: com align padrão (ao RGBNIR), o TIF em `rasters/` fica a **0.25 m**,
independente do `-Res`. Para final a 1 m: `-Res 1 -NoOptical` (e no
`experiment.yaml` use `chm_resolution: 1.0`).

---

## Debug (lab / home)

### 1) `.ps1` abre no editor / não executa

Causa: associação do Windows ou clique no arquivo.

**Fix:** usar `.\run_lidar.cmd ...` (nunca depender de abrir o `.ps1`).

### 2) "execução de scripts foi desabilitada neste sistema"

Causa: `ExecutionPolicy` do PowerShell.

**Fix (já embutido no `.cmd`):**

```powershell
.\run_lidar.cmd -aoi 261002-riodaspedras
# ou
powershell -NoProfile -ExecutionPolicy Bypass -File .\run_lidar.ps1 -aoi 261002-riodaspedras
```

Não precisa liberar scripts no sistema inteiro.

### 3) Docker: só "engine running" — preciso fazer algo?

Engine running **já basta**. Não precisa subir container manualmente
(`docker compose run --rm` é feito pelo script).

Confira a imagem:

```powershell
docker images leucaena-segmentation:cuda
```

Se não existir, build uma vez na raiz do repo:

```powershell
docker build -t leucaena-segmentation:cuda .
```

### 4) `AOI nao resolveu nenhum LAZ (cheque NOMENC_* vs tiles/lidar/raw)`

O AOI foi lido, mas **nenhum `.laz` no pool** casou com o NOMENC.

Checklist no lab (`$root` = valor de `PHD_RUN_ROOT` no `.env`):

```powershell
$root = "E:\dados-msb\00_leucaena-earth-experiments"   # ajuste ao seu .env

# pasta existe e tem LAZ?
Get-ChildItem "$root\tiles\lidar\raw" -Filter *.laz | Select-Object -First 15 Name
(Get-ChildItem "$root\tiles\lidar\raw" -Filter *.laz -ErrorAction SilentlyContinue).Count

# o que o AOI pede (optico + laz)
python scripts\lidar-products\resolve_aoi_lidar.py --root $root --aoi 261002-riodaspedras
```

Esperado para Rio das Pedras:

```text
tiles/lidar/raw/
  SF-23-Y-A-IV-4-SO-A-I.laz
  SF-23-Y-A-IV-4-SO-A-II.laz
  SF-23-Y-A-IV-4-SO-A-III.laz
  SF-23-Y-A-IV-4-SO-A-IV.laz
```

Causas comuns:
- LAZ ainda na pasta **antiga** da AOI (não em `tiles/lidar/raw/`) → copiar/mover para o pool
- Nomes dos arquivos ≠ `NOMENC_5K` / prefixo do tile óptico
- `PHD_RUN_ROOT` apontando para outro disco/pasta vazia
- AOI gpkg sem `NOMENC_*` ou com valores que não batem com os stems

Como o match funciona (`resolve_aoi_lidar.py`):
1. Qualquer `NOMENC_2K/5K/10K` do gpkg cujo `.laz` exista no pool
2. Qualquer `.laz` com stem == tile óptico **ou** começando com `"<tile>-"`
   (ex.: `SF-23-Y-A-IV-4-SO-A-I` sob o tile `SF-23-Y-A-IV-4-SO-A`)

### 5) `AOI nao resolveu nenhum tile optico`

Cheque `tiles/rgbnir/` e os campos NOMENC do gpkg:

```powershell
Get-ChildItem "$root\tiles\rgbnir" -Filter *.tif | Select-Object Name
python scripts\lidar-products\resolve_aoi_lidar.py --root $root --aoi 261002-riodaspedras
```

### 6) `Rscript nao encontrado`

Ajuste `R_BIN` no `.env` para o `Rscript.exe` instalado nesse PC
(ex.: `C:/Program Files/R/R-4.6.1/bin/Rscript.exe`). Confirme:

```powershell
Test-Path $env:R_BIN   # ou o caminho escrito no .env
```

### 7) Passo R ok, align falha

- Docker engine running?
- Imagem `leucaena-segmentation:cuda` presente?
- Existe `tiles/rgbnir/<tile>.tif` para cada tile óptico do AOI?
- Existem `*_CHM.tif` em `tiles/lidar/products/` para os LAZ do AOI?

```powershell
Get-ChildItem "$root\tiles\lidar\products" -Filter "*_CHM.tif" | Select-Object Name
```

---

## Depois do LiDAR → treino MANet

Com o TIF em `tiles/lidar/rasters/` pronto:

```powershell
cd <repo>
.\run.cmd all -aoi 261002-riodaspedras -experiment manet/latefusion-multitask -run run-001
```

Ver também `scripts/phd-run/README.md`.
