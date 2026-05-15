# 01 — Docker WSL CUDA

- **Status:** `built + deployed`
- **Owner:** Matheus
- **Last update:** 2026-05-15

## Why

Run the full `prep-data` → `train` → `prediction` → `evaluation` pipeline
inside a reproducible Linux container with NVIDIA GPU access from Windows
via WSL2. Avoid installing GDAL and CUDA PyTorch directly on Windows; make
the environment trivial to recreate on another machine or for the PhD
defence.

## Decisions

- Base image: `pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel` (Python 3.11
  inside `/opt/conda`).
- GDAL via `conda-forge` (not `apt` + `pip`): apt's `libgdal` requires
  `GLIBCXX_3.4.30`, which the conda `libstdc++` in the PyTorch base image
  does not provide. Trying the apt route causes
  `ImportError: ... GLIBCXX_3.4.30 not found ... required by /lib/libgdal.so.30`
  on `from osgeo import gdal`.
- `requirements-docker.txt` pins everything except torch/torchvision/cuda
  (those come from the base image).
- Compose service `segmentation` mounts the repo at `/workspace` and exposes
  GPU via the `deploy.resources` block; one-shot or interactive shell.
- Daily work can stay on **conda in WSL** (see CHEATSHEET.md); Docker is for
  reproducibility / clean reruns / sharing with others.

## Files touched

- [`Dockerfile`](../Dockerfile)
- [`docker-compose.yml`](../docker-compose.yml)
- [`requirements-docker.txt`](../requirements-docker.txt)
- [`.dockerignore`](../.dockerignore)
- [`DOCKER.md`](../DOCKER.md)
- [`CHEATSHEET.md`](../CHEATSHEET.md)
- [`scripts/docker-shell.sh`](../scripts/docker-shell.sh)
- [`README.md`](../README.md) (section link)

## How to run (recap)

```bash
# WSL Ubuntu
nvidia-smi
# Docker Desktop + WSL integration
# NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi

# Project
cd "/mnt/c/Users/mathe/OneDrive/Documents/0-GITHUB/leucaena-earth-segmentation"
docker compose build
docker compose run --rm segmentation bash
# inside container
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python -c "from osgeo import gdal; print('GDAL OK', gdal.VersionInfo())"
```

## Outcome (2026-05-15)

- Build succeeds on RTX 4080 / WSL2 Ubuntu.
- Inside container: `CUDA: True`, `GDAL OK 3120400`.
- Pipeline scripts importable; ready for data.

## Risks / known issues

| Risk | Mitigation |
|------|------------|
| Slow I/O on `/mnt/c/` | Document moving the repo to `~/projects/` |
| GLIBCXX mismatch | Pin GDAL install to conda-forge in `Dockerfile` |
| Out of GPU memory | Reduce `-b` in `train.py` (4 or 2) |

## Out of scope (followed up elsewhere)

- Tile-based pipeline (-> plan 03).
- Multi-GPU / distributed training.
