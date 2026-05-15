# Cheat sheet — leucaena-earth-segmentation

Guia rápido para **você** e para **um colega** rodar o projeto.  
Dois caminhos:

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
| `prep-data.py` | GeoTIFF + máscaras → `prepared/*.npy` |
| `train.py` | Treina → `experiments/exp_N/models/model.pt` |
| `prediction.py` | Mapa completo → `experiments/exp_N/predicted/` |
| `evaluation.py` | Métricas → `experiments/exp_N/logs/eval_N.txt` |
| `conf/paths.py` | Caminhos dos arquivos em `data/` |
| `conf/general.py` | Patch size, LR, early stopping |
| `Dockerfile` | Receita da imagem Docker |
| `docker-compose.yml` | Como subir container (GPU + pasta) |
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

Layout recomendado (Google Drive / disco local):

```
G:/My Drive/PHD/02-Tese/00-datasets-ai/<dataset-id>/
  tiles/                    # GeoTIFFs RGB+IR
  annotations/
    polygons.geojson        # máscaras (export leucaena.earth)
    platform-backup/        # .db opcional (não entra no treino)
```

Exemplo no `.env` (copie de `.env.example`):

```
LEUCAENA_DATASET_ID=260515-piracicaba-aoi
LEUCAENA_TILES_HOST_DIR=G:/My Drive/PHD/02-Tese/00-datasets-ai/260515-piracicaba-aoi/tiles
LEUCAENA_MASKS_HOST_DIR=G:/My Drive/PHD/02-Tese/00-datasets-ai/260515-piracicaba-aoi/annotations
LEUCAENA_MASKS_PATH=/data/masks/polygons.geojson
```

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
| `--overlap` | `0.5` | Sobreposição do sliding window |
| `--min-target-class` | `0.01` | Fração mínima de pixels leucaena por patch |
| `--test-split` | `0.2` | Fração para teste |
| `--val-split` | `0.2` | Fração de validação dentro do treino |
| `--band-order` | `RGBN` | Reordena para BGRN automaticamente |
| `--max-tiles` | — | Processa só N tiles (debug) |

### Como o split funciona

Splits são feitos **no nível de patch** com `--seed` (default 42). Patches do mesmo tile podem aparecer em splits diferentes — para evitar isso (mais rigoroso), o próximo passo é usar `--split-by tile`, ainda não implementado.

---

# Espaço em disco (Docker)

```bash
docker system df
```

Limpar imagens antigas (cuidado — apaga o que não está em uso):

```bash
docker system prune
```
