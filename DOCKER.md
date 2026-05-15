# Docker on WSL (CUDA)

Run the full **leucaena-earth-segmentation** pipeline inside a Linux container with NVIDIA GPU support. This avoids installing GDAL and CUDA PyTorch directly on Windows.

**Pipeline:** `prep-data.py` → `train.py` → `prediction.py` → `evaluation.py`

---

## Prerequisites (one-time)

### 1. WSL2 + Ubuntu

You already have WSL. Use an Ubuntu distro (22.04 or 24.04 recommended).

### 2. NVIDIA driver on Windows

Install or update the [NVIDIA driver for Windows](https://www.nvidia.com/Download/index.aspx) (WSL2 GPU support is included in recent drivers).

Inside WSL, verify the GPU is visible:

```bash
nvidia-smi
```

If this fails, fix the Windows driver before continuing.

### 3. Docker

**Option A — Docker Desktop (simplest)**

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/).
2. Settings → Resources → WSL integration → enable your Ubuntu distro.
3. Restart Docker Desktop.

**Option B — Docker Engine inside Ubuntu**

Follow [Docker’s Ubuntu install guide](https://docs.docker.com/engine/install/ubuntu/).

### 4. NVIDIA Container Toolkit (required for GPU in containers)

On your **Ubuntu WSL** terminal:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
# If using Docker Desktop, restart Docker Desktop instead of systemctl.
```

### 5. Quick GPU test

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

You should see your GPU listed inside the container.

---

## Where to put the repository

**Recommended (faster I/O for large `.npy` files):**

```bash
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/matheussiba/leucaena-earth-segmentation.git
cd leucaena-earth-segmentation
```

**Alternative (slower):** work from `/mnt/c/Users/.../leucaena-earth-segmentation` — OneDrive/NTFS mounts are much slower for training I/O.

Place your data under `data/` in the repo root (same layout as the README).

---

## Build the image

From the repository root:

```bash
docker compose build
```

First build downloads the PyTorch CUDA base image (~several GB) and installs GDAL + Python packages. Subsequent builds use the cache.

---

## Run an interactive shell (with GPU)

```bash
docker compose run --rm segmentation bash
```

Or use the helper script:

```bash
chmod +x scripts/docker-shell.sh
./scripts/docker-shell.sh
```

Inside the container, verify the stack:

```bash
nvidia-smi
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python -c "from osgeo import gdal; print('GDAL', gdal.VersionInfo())"
```

---

## Run the pipeline

All commands below assume you are **inside the container** at `/workspace`, with `data/` mounted from the host.

### Step 1 — Prepare data

```bash
# Optical only (experiment 1)
python prep-data.py --optical data/optical.tif --masks data/masks.geojson --no-lidar

# With LiDAR (experiments 2 and 3)
python prep-data.py --optical data/optical.tif --lidar data/lidar.tif --masks data/masks.geojson
```

### Step 2 — Train

Always pass `-e 1`, `2`, or `3` (there is no `model_9.py`):

```bash
python train.py -e 1 -b 8
python train.py -e 2 -b 8
python train.py -e 3 -b 8
```

Reduce batch size if you hit GPU OOM: `-b 4` or `-b 2`.

### Step 3 — Predict

```bash
python prediction.py -e 1
```

### Step 4 — Evaluate

```bash
python evaluation.py -e 1
```

Outputs appear on the host under `prepared/` and `experiments/` (bind-mounted volumes).

---

## One-shot commands (no interactive shell)

```bash
docker compose run --rm segmentation python train.py -e 1 -b 8
docker compose run --rm segmentation python prediction.py -e 1
```

If GPU is not picked up with `docker compose`, try:

```bash
docker compose run --rm --gpus all segmentation python train.py -e 1 -b 8
```

---

## Troubleshooting

| Symptom | What to try |
|--------|-------------|
| `nvidia-smi` fails in WSL | Update Windows NVIDIA driver; reboot |
| Container has no GPU | Install NVIDIA Container Toolkit; restart Docker |
| `CUDA available: False` | Use `--gpus all`; check `nvidia-smi` inside container |
| GDAL import error (`GLIBCXX_3.4.30` / `_gdal`) | Image uses conda-forge GDAL; rebuild: `docker compose build` |
| Very slow training | Move repo to `~/projects` (not `/mnt/c/`); lower `-b` |
| Out of memory | `python train.py -e 1 -b 4` or `-b 2` |
| Stale partial outputs | Delete bad files under `experiments/` and re-run |

---

## Files added for Docker

| File | Purpose |
|------|---------|
| `Dockerfile` | CUDA PyTorch base + GDAL + Python deps |
| `docker-compose.yml` | GPU service, volume mount, interactive shell |
| `requirements-docker.txt` | Pinned pip packages (torch from base image) |
| `.dockerignore` | Exclude `data/`, `prepared/`, `experiments/` from build context |
| `scripts/docker-shell.sh` | Build + open shell with GPU |

---

## Post-setup checklist

- [ ] `nvidia-smi` works in WSL
- [ ] `docker run --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` works
- [ ] `docker compose build` succeeds
- [ ] Inside container: `torch.cuda.is_available()` is `True`
- [ ] Inside container: `from osgeo import gdal` succeeds
- [ ] `python prep-data.py --help` runs
- [ ] Short training run with real data completes (`train.py -e 1 -b 2`)
