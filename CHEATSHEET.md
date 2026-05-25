# Cheat sheet — leucaena-earth-segmentation

Guia rápido para **você** e para **um colega** rodar o projeto.

---

## Pipeline Orquestrado (comando único — Windows PowerShell)

Executa as 5 etapas automaticamente: cópia → fusão RGBNIR → CHM → patches → treino.

```powershell
python run_pipeline.py `
  --aoi "G:\My Drive\PHD\02-Tese\02-data\adote-uma-leucena\v1-LEUCENA MAPPING\gdb-leucena_v2.gpkg" `
  --layer articulacao_laser_voo22_AOI_treino `
  --source D:\ `
  --dest "C:\00_DATASETS_AI\260515-piracicaba-aoi" `
  --build-ovr `
  --train
```

**Dry-run (nenhum arquivo é alterado):**
```powershell
python run_pipeline.py --aoi ... --layer ... --source D:\ --dest ... --dry-run --verbose
```

**Rodar etapas específicas:**
```powershell
python run_pipeline.py ... --steps 1,2          # só cópia + fusão
python run_pipeline.py ... --steps 3,4 --train  # CHM + patches + treino
```

> Documentação completa: [`studies/orquestrador-pipeline.md`](studies/orquestrador-pipeline.md)

---

Dois caminhos para ambiente de execução:

| Caminho | Quando usar |
|---------|-------------|
| **A — WSL + conda** | Dia a dia: editar código, treinar, debug |
| **B — Docker** | Experimento final, tese, reprodutibilidade, mandar para outro |

E para a **fonte dos patches** há dois pipelines:

| Fonte | Script | Quando usar |
|-------|--------|-------------|
| `scene` (legado) | `prep-data.py` | Uma cena/VRT que cabe em RAM |
| `file` (escalável) | `prep-patches-from-tiles.py` | Pasta com **tiles** + GeoJSON, qualquer escala |

**Repo (Windows):**  
`C:\Users\mathe\OneDrive\Documents\0-GITHUB\leucaena-earth-segmentation`

**No WSL (mesma pasta):**  
`/mnt/c/Users/mathe/OneDrive/Documents/0-GITHUB/leucaena-earth-segmentation`

---

## O que foi instalado neste PC (resumo)

| Componente | Onde | Para quê |
|------------|------|----------|
| WSL2 + Ubuntu | Windows | Terminal Linux |
| Driver NVIDIA | Windows | GPU no WSL (`nvidia-smi`) |
| Docker Desktop | Windows | Containers |
| NVIDIA Container Toolkit | Ubuntu (WSL) | GPU **dentro** do Docker |
| Imagem `leucaena-segmentation:cuda` | Disco do Docker | Ambiente PyTorch+CUDA+GDAL |
| Ambiente conda `leucaena` | WSL (você cria no caminho A) | Ambiente leve para o dia a dia |

---

# CAMINHO A — WSL + conda (dia a dia)

## A1 — Primeira vez (só uma vez por máquina)

### 1) Abrir Ubuntu (WSL)

Menu Iniciar → **Ubuntu**.

### 2) Testar GPU

```bash
nvidia-smi
```

Tem que listar a placa (ex.: RTX 4080). Se falhar → atualizar driver NVIDIA no **Windows** e reiniciar.

### 3) Instalar Miniconda (se ainda não tiver)

https://docs.conda.io/en/latest/miniconda.html — instalador **Linux x86_64** no WSL, ou:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# seguir instruções; depois:
source ~/.bashrc
```

### 4) Ir ao projeto

```bash
cd "/mnt/c/Users/mathe/OneDrive/Documents/0-GITHUB/leucaena-earth-segmentation"
```

### 5) Criar ambiente Python

```bash
conda create -n leucaena python=3.11 gdal -c conda-forge -y
conda activate leucaena
pip install -r requirements.txt
```

### 6) Instalar PyTorch com CUDA (importante)

O `requirements.txt` só diz `torch` — instale a versão CUDA explícita:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 7) Testar

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from osgeo import gdal; print('GDAL OK')"
```

Ambos sem erro → pronto para o caminho A.

### 8) Dados

No Windows, pasta do projeto:

```
data/
  optical.tif
  masks.geojson
  lidar.tif    (opcional)
```

---

## A2 — Toda vez que for trabalhar (conda)

```bash
cd "/mnt/c/Users/mathe/OneDrive/Documents/0-GITHUB/leucaena-earth-segmentation"
conda activate leucaena
```

### Pipeline completo

```bash
# 1 — Preparar (sem LiDAR = experimento 1)
python prep-data.py --optical data/optical.tif --masks data/masks.geojson --no-lidar

# Com LiDAR (experimentos 2 e 3)
# python prep-data.py --optical data/optical.tif --lidar data/lidar.tif --masks data/masks.geojson

# 2 — Treinar (sempre -e 1, 2 ou 3)
python train.py -e 1 -b 8

# 3 — Prever
python prediction.py -e 1

# 4 — Avaliar
python evaluation.py -e 1
```

### Atalhos úteis

| Situação | Comando |
|----------|---------|
| GPU sem memória | `python train.py -e 1 -b 4` ou `-b 2` |
| Continuar treino | `python train.py -e 1 -c` |
| Ver log ao vivo | `tail -f experiments/exp_1/logs/train_1.txt` |
| Sair do ambiente | `conda deactivate` |

---

# CAMINHO B — Docker (reprodutibilidade / tese)

## B1 — Primeira vez (só uma vez por máquina)

### 1) WSL + GPU

```bash
nvidia-smi
```

### 2) Docker Desktop (Windows)

- Instalar: https://www.docker.com/products/docker-desktop/
- Settings → **WSL integration** → Ubuntu **ON** → Apply & Restart

No Ubuntu:

```bash
docker --version
docker compose version
```

### 3) NVIDIA Container Toolkit (Ubuntu)

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
```

**Reiniciar Docker Desktop.**

### 4) Testar GPU no Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

### 5) Build da imagem do projeto (demora ~10–30 min na 1ª vez)

```bash
cd "/mnt/c/Users/mathe/OneDrive/Documents/0-GITHUB/leucaena-earth-segmentation"
docker compose build
```

### 6) Entrar no container e testar

```bash
docker compose run --rm segmentation bash
```

Dentro do container:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from osgeo import gdal; print('GDAL OK', gdal.VersionInfo())"
exit
```

---

## B2 — Toda vez que for usar Docker

```bash
cd "/mnt/c/Users/mathe/OneDrive/Documents/0-GITHUB/leucaena-earth-segmentation"
docker compose run --rm segmentation bash
```

Dentro do container (`/workspace` = pasta do projeto):

```bash
python prep-data.py --optical data/optical.tif --masks data/masks.geojson --no-lidar
python train.py -e 1 -b 8
python prediction.py -e 1
python evaluation.py -e 1
exit
```

### Quando rodar `docker compose build` de novo?

| Situação | Precisa rebuild? |
|----------|------------------|
| Só treinar / prever de novo | **Não** |
| Mudou `Dockerfile` ou `requirements-docker.txt` | **Sim** → `docker compose build` |
| GDAL deu erro após mudança no Dockerfile | **Sim** |

### Um comando sem abrir shell

```bash
docker compose run --rm segmentation python train.py -e 1 -b 8
```

---

# Arquivos do projeto (o que cada um faz)

| Arquivo | Função |
|---------|--------|
| `prep-data.py` | GeoTIFF + máscaras → `prepared/*.npy` (cena única, RAM) |
| `prep-patches-from-tiles.py` | Pasta de tiles + GeoJSON → `prepared/patches/` (escalável) |
| `prep-lidar-rasters.py` | LAZ → 2-band TIF (CHM, INTENSITY) alinhado ao RGBN |
| `train.py` | Treina → `experiments/exp_N/models/model.pt` |
| `prediction.py` | Cena única (legado) → `experiments/exp_N/predicted/` |
| `predict-tiles.py` | **Inferência tile-by-tile, escalável** → `$LEUCAENA_PREDICTIONS_DIR/exp_N/` |
| `predict-tile-preview.py` | PNG triptych + F1 num único tile (sanity check) |
| `evaluation.py` | Métricas → `experiments/exp_N/logs/eval_N.txt` |
| `utils/inference.py` | Helpers de sliding-window + escrita de prob TIF |
| `utils/lidar.py` | Helpers PDAL + GDAL (pipelines, alinhamento, CHM) |
| `conf/paths.py` | Caminhos dos arquivos em `data/` e tile dirs |
| `conf/general.py` | Patch size, LR, early stopping, normalização LiDAR |
| `Dockerfile` | Receita da imagem Docker (CUDA + GDAL + PDAL) |
| `docker-compose.yml` | Como subir container (GPU + pastas montadas) |
| `DOCKER.md` | Guia longo Docker + troubleshooting |
| `CHEATSHEET.md` | Este arquivo |

---

# Experimentos

| `-e` | Modelo | Dados |
|------|--------|-------|
| `1` | Só óptico | RGB/NIR, sem LiDAR |
| `2` | Fusão early | óptico + LiDAR concatenados |
| `3` | Fusão late | dois encoders |

Sempre use `-e 1`, `-e 2` ou `-e 3` (não use o default `9` do script).

---

# Para um colega (checklist mínimo)

1. Clonar o repo: `git clone https://github.com/matheussiba/leucaena-earth-segmentation.git`
2. Colocar dados em `data/`
3. Escolher **A (conda)** ou **B (Docker)** e seguir a seção **primeira vez** correspondente
4. Rodar os 4 scripts na ordem
5. Para tese/paper: preferir **B** e citar commit + `docker compose build`

---

# Problemas comuns

| Sintoma | O que fazer |
|---------|-------------|
| `nvidia-smi` falha no WSL | Driver NVIDIA no Windows |
| `CUDA: False` no conda | Reinstalar torch com index `cu124` |
| `CUDA: False` no Docker | Toolkit NVIDIA + reiniciar Docker Desktop |
| GDAL / `GLIBCXX` no Docker | `docker compose build` (Dockerfile usa conda-forge GDAL) |
| Treino muito lento | Copiar repo para `~/projects/...` (fora de `/mnt/c/`) |
| OOM na GPU | `-b 4` ou `-b 2` |
| Pasta `experiments/` vazia | Rodar `train.py` antes de `prediction.py` |

---

# Qual caminho usar? (decisão em 5 segundos)

```
Vou codificar / debugar / treinar várias vezes hoje?
  → SIM → Caminho A (conda activate leucaena)

Preciso garantir que outra pessoa ou eu daqui a 1 ano
reproduz exatamente o mesmo ambiente?
  → SIM → Caminho B (docker compose run ...)
```

---

# Pipeline tile-based (escalável)

Para quando seus dados estão divididos em **muitos tiles** (caso típico do PhD: imagens IGC, voo 22, etc.) e o GeoJSON cobre uma área muito maior que um único tile.

### Organização dos dados (uma vez)

Caminhos da **sua máquina** ficam no arquivo **`.env`** (gitignored — não vai pro GitHub):

```bash
cp .env.example .env
# edite .env com seus caminhos no Windows
```

Layout recomendado (disco local **C:**, sem espaços no caminho):

```
C:/00_DATASETS_AI/<dataset-id>/
  tiles/                    # GeoTIFFs RGB+IR
  annotations/
    polygons.geojson        # máscaras (export leucaena.earth)
```

Exemplo no `.env` — rode **`docker compose` no WSL** com caminhos `/mnt/c/...` (não use `C:/` nem `G:/` no `.env`):

```
LEUCAENA_TILES_HOST_DIR=/mnt/c/00_DATASETS_AI/260515-piracicaba-aoi/tiles
LEUCAENA_MASKS_HOST_DIR=/mnt/c/00_DATASETS_AI/260515-piracicaba-aoi/annotations
LEUCAENA_MASKS_PATH=/data/masks/polygons.geojson
```

**Erro `invalid volume specification`?** Use `/mnt/c/...` no `.env`, não `C:/...` nem Google Drive `G:/My Drive/...`.

Novo experimento: crie outra pasta em `00-datasets-ai/<id>/` e atualize o `.env`.

`docker-compose.yml` lê `.env` e monta as pastas no container. **Patches gerados** vão para `prepared/patches/` dentro do repo (no C:, via OneDrive).

### Gerar patches

Dentro do container (`docker compose run --rm segmentation bash`):

```bash
# defaults vêm do .env via conf/paths.py — não precisa repetir caminhos:
python prep-patches-from-tiles.py --band-order RGBN
```

Saída em `prepared/patches/`:
- `opt/<patch_id>.npy` uint8 (256×256×4 BGRN)
- `lbl/<patch_id>.npy` uint8 (256×256, 0/1)
- `manifest.csv` com coluna `split` (`train`/`val`/`test`)
- `preparation.txt` (log com contagens por tile)

### Treinar com patches do disco

```bash
python train.py -e 1 -b 8 --patch-source file
```

Sem `--patch-source file`, treina pelo caminho antigo (`prep-data.py`). Predição (`prediction.py`) continua usando o caminho cena/VRT por enquanto.

### Flags úteis do `prep-patches-from-tiles.py`

| Flag | Default | Para que serve |
|------|---------|----------------|
| `--tiles-dir` | `/data/rgbir` | Pasta dos tiles `.tif` |
| `--tiles-glob` | `*.tif` | Filtro de nomes |
| `--masks` | `data/masks.geojson` | GeoJSON com polígonos |
| `--out-dir` | `prepared/patches` | Onde salvar |
| `--patch-size` | `256` | Lado do patch |
| `--overlap` | `0.6` | Sobreposição do sliding window (60%) |
| `--min-target-class` | `0.01` | Fração mínima de pixels leucaena por patch |
| `--test-split` | `0.2` | Fração para teste |
| `--val-split` | `0.2` | Fração de validação dentro do treino |
| `--band-order` | `RGBN` | Reordena para BGRN automaticamente |
| `--max-tiles` | — | Processa só N tiles (debug) |

### Como o split funciona

Splits são feitos **no nível de patch** com `--seed` (default 42). Patches do mesmo tile podem aparecer em splits diferentes — para evitar isso (mais rigoroso), o próximo passo é usar `--split-by tile`, ainda não implementado.

---

# Pipeline LiDAR — do `.laz` ao patch

Use quando você tem **nuvem de pontos LiDAR** (`.laz` / `.copc.laz`) e quer treinar
os experimentos `2` (early fusion) ou `3` (late fusion). O caminho tile-based espera
**rasters LiDAR alinhados aos RGBN** — não a nuvem bruta. Esse pipeline faz a
conversão.

### Fluxo em 3 passos

```
D:/laz/<tile>.laz                       (entrada — nuvem de pontos)
        │
        ▼
[ 1. prep-lidar-rasters.py ]            (PDAL → DSM/DTM/Intensity, CHM = DSM − DTM)
        │
        ▼
C:/00_DATASETS_AI/.../lidar/<tile>.tif  (2 bandas float32: CHM, INTENSITY,
                                         mesmo grid do RGBN)
        │
        ▼
[ 2. prep-patches-from-tiles.py --lidar-dir ... ]
        │
        ▼
prepared/patches/lidar/<patch_id>.npy   (256×256×2 float32, alinhado ao opt/lbl)
        │
        ▼
[ 3. train.py -e 2  (ou -e 3) --patch-source file ]
```

### Pré-requisitos

- **No Docker**: a imagem já inclui PDAL (`docker compose build` se ainda não fez).
- **No conda (caminho A)**: `conda install -y -c conda-forge pdal python-pdal`.

### Configurar os caminhos (`.env`)

```env
# LAZ na sua máquina (D:\laz monta como /mnt/d/laz no WSL)
LEUCAENA_LAZ_HOST_DIR=/mnt/d/laz
LEUCAENA_LIDAR_HOST_DIR=/mnt/c/00_DATASETS_AI/260515-piracicaba-aoi/lidar
LEUCAENA_LAZ_DIR=/data/laz       # caminho dentro do container (não mudar)
LEUCAENA_LIDAR_DIR=/data/lidar   # idem
```

`docker compose run --rm segmentation bash` monta `/data/laz` (somente leitura
prática — você não escreve aqui) e `/data/lidar` (saída).

### 1. Rasterizar LAZ → 2-band TIF

```bash
# --- escala TESTE (1–2 tiles, valida toolchain antes de 300 horas) ---
python prep-lidar-rasters.py --max-tiles 2

# --- inspeção rápida sem produzir nada (só metadados PDAL + match RGBN) ---
python prep-lidar-rasters.py --inspect-only --max-tiles 5

# --- escala FINAL (todos os LAZ que têm RGBN correspondente) ---
python prep-lidar-rasters.py
```

Resultado em `/data/lidar/` (mapeado para `C:\00_DATASETS_AI\.../lidar` no host):

- `<tile>.tif` — 2 bandas float32 `[CHM, INTENSITY]`, mesmo grid do RGBN
- `lidar_manifest.csv` — uma linha por LAZ: `status` (`ok` / `skip-no-rgbn` /
  `skip-existing` / `error`), `n_points`, dimensões, tempo, mensagem de erro.
- `preparation.txt` — log completo do batch.

### 2. Gerar patches com LiDAR (ativa o label refinado)

> Passar `--lidar-dir` ativa automaticamente o **label refinado** (regra
> nova do professor): pixels fora do polígono viram `IGNORE (255)`;
> dentro do polígono, `1` apenas se `CHM ≥ 4.5 m` E `NDVI ≥ 0.3`, senão
> `0`. Detalhes em
> [`studies/labeling-refinado.md`](studies/labeling-refinado.md).
> Tiles **sem** LiDAR correspondente são **pulados** nesse modo.

```bash
python prep-patches-from-tiles.py \
    --lidar-dir /data/lidar \
    --band-order RGBN
```

Saída em `prepared/patches/`:

- `opt/<patch_id>.npy`   uint8 (256×256×4 BGRN)
- `lbl/<patch_id>.npy`   uint8 (256×256)
- `lidar/<patch_id>.npy` **float32 (256×256×2)** — só para tiles que tinham LiDAR
- `manifest.csv` agora com colunas extras `lidar_tile_name`, `has_lidar`

> Patches sem LiDAR ficam com `has_lidar=False`. O `PatchFileDataset` devolve
> tensor de zeros para eles, então treinar `-e 1` (óptico) continua valendo
> mesmo com manifesto misto.

### 3. Treinar com fusão

```bash
python train.py -e 2 -b 8 --patch-source file   # early fusion
python train.py -e 3 -b 4 --patch-source file   # late fusion (mais memória)
```

### Sintonia rápida do `prep-lidar-rasters.py`

| Flag | Default | Para que serve |
|------|---------|----------------|
| `--laz-dir` | `/data/laz` | Pasta com `.laz` / `.copc.laz` |
| `--tiles-dir` | `/data/rgbir` | RGBN de referência (alinhamento de grid) |
| `--out-dir` | `/data/lidar` | Onde grava os `.tif` 2-band |
| `--resolution` | `1.0` m | Resolução do raster PDAL antes do warp |
| `--chm-max-m` | `50.0` | Cap do CHM (filtra spikes) |
| `--max-tiles` | — | Processa só N LAZ (debug) |
| `--require-rgbn` / `--no-require-rgbn` | `True` | Pular LAZ sem RGBN correspondente |
| `--overwrite` | `False` | Refazer mesmo se o `.tif` já existe |
| `--inspect-only` | `False` | Só imprime metadados PDAL, sem rasterizar |

### Como o casamento LAZ ↔ RGBN funciona

Por **stem** (nome do arquivo sem extensão), com `.copc` removido:

```
D:\laz\SF-23-Y-A-IV-2-SE-E-I.copc.laz   →   stem = SF-23-Y-A-IV-2-SE-E-I
C:\...\tiles\SF-23-Y-A-IV-2-SE-E-I.tif  →   match!
```

Se você quiser um mapeamento não-trivial (nomes diferentes), o caminho mais
limpo é renomear os RGBN para casar com o LAZ (ou vice-versa). Um suporte a
`--mapping-csv` está na lista de futuros.

### Constantes que dá pra mexer (em `conf/general.py`)

| Constante | Valor | Efeito |
|-----------|-------|--------|
| `LIDAR_RASTER_RESOLUTION_M` | `1.0` | Resolução intermediária do PDAL |
| `LIDAR_CHM_MAX_M` | `50.0` | Tudo acima vira 50 m. CHM em [0, 1] = altura/50 |
| `LIDAR_INTENSITY_MAX` | `32768` | Divisor para intensidade. Idem [0, 1] |

Essas três regem a normalização final que vai para a rede. Não precisa nada
extra no dataloader: ele lê `lidar/<id>.npy` e aplica a regra automaticamente.

---

# Predição em escala (tile-by-tile)

Use quando você quiser **rodar inferência sobre muitos tiles** sem mosaicar
nada em RAM. `prediction.py` carrega a cena inteira; `predict-tiles.py` itera
tile por tile e gera GeoTIFFs georreferenciados que o QGIS abre como mosaico
via VRT. Discussão didática em `studies/predicao-em-escala.md`.

### Onde a saída vai parar

**Fora do OneDrive** — uma rodada estado/país pode gerar dezenas de GB. Caminho
controlado pelo `.env`:

```env
LEUCAENA_PREDICTIONS_HOST_DIR=/mnt/d/leucaena-predictions  # disco local
LEUCAENA_PREDICTIONS_DIR=/data/predictions                  # dentro do container
```

Estrutura por experimento (default `$LEUCAENA_PREDICTIONS_DIR/exp_<N>/`):

```
<stem>_pred.tif   uint8   mapa de classe (0 = fundo, 1 = leucaena)
<stem>_prob.tif   uint16  prob da classe 1, scale_factor = 1/65535  (default)
pred.vrt          mosaico virtual de todos os *_pred.tif
prob.vrt          mosaico virtual de todos os *_prob.tif
manifest.csv      uma linha por tile (status, lidar_status, frac_leucaena, ...)
predict_<N>.txt   log completo
```

### Comandos típicos

```bash
# default: todos os tiles em PATH_TILES_DIR, overlaps [0, 0.25, 0.5],
# probabilidade gravada em uint16 (2 bytes/pixel) com scale_factor
python predict-tiles.py -e 1

# preview rápido: 3 tiles, uma única passada, sem média de overlaps
python predict-tiles.py -e 1 --max-tiles 3 --overlap 0

# experimento de fusão (LiDAR real; tiles sem LiDAR usam zeros + log)
python predict-tiles.py -e 3 --lidar-dir /data/lidar

# manter prob em float32 (4 bytes/pixel) se quiser análise crua
python predict-tiles.py -e 1 --prob-dtype float32
```

### Flag-a-flag

| Flag | Default | Para que serve |
|------|---------|----------------|
| `-e` | (obrigatório) | Número do experimento (carrega `experiments/exp_<e>/models/model.pt`) |
| `--tiles-dir` | `PATH_TILES_DIR` (`.env`) | Pasta dos RGBN |
| `--tiles-glob` | `*.tif` | Filtro de nomes |
| `--lidar-dir` | `PATH_LIDAR_DIR` | Só usado se o modelo é multimodal |
| `--out-dir` | `$PREDICTIONS_DIR/exp_<e>` | Override do destino |
| `--band-order` | `RGBN` | Bate com o treinamento |
| `-b/--batch-size` | `128` | Batch da inferência |
| `--overlaps` | `0,0.25,0.5` | Lista averageada |
| `--overlap` | — | Valor único; sobrescreve `--overlaps` (preview) |
| `--save-prob` / `--no-save-prob` | `True` | Gravar `_prob.tif` |
| `--prob-dtype` | `uint16` | `float32` \| `uint16` \| `uint8` |
| `--max-tiles` | `0` (todos) | Limita para debug |
| `--overwrite` | `False` | Refaz tiles já gravados |
| `--no-build-vrt` | (constrói por padrão) | Pula `gdalbuildvrt` no fim |
| `--device` | `auto` | `auto` \| `cuda` \| `cpu` |

### Por que `uint16` na probabilidade?

GeoTIFF/GDAL **não** têm Float16 nativo. O padrão científico para "prob em meio
da precisão" é `uint16 + scale_factor = 1/65535`: 2 bytes por pixel, 65 536
níveis, lido como `[0, 1]` em QGIS/rasterio sem código extra. Detalhes em
`studies/predicao-em-escala.md`.

### Quando faltar LiDAR

`--lidar-dir` é varrido por `<stem>_lidar.tif` (e `<stem>.tif` como fallback).
Se um tile não tem LiDAR mas o modelo é fusion: **prediz com zeros e marca
`lidar_status=missing` no `manifest.csv`**. Decisão fica explícita no log,
não some.

---

# Espaço em disco (Docker)

```bash
docker system df
```

Limpar imagens antigas (cuidado — apaga o que não está em uso):

```bash
docker system prune
```
