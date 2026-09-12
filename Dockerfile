# CUDA-enabled PyTorch image for leucaena-earth-segmentation
# RTX 50-series (Blackwell, sm_120) needs PyTorch >= 2.7 built with CUDA >= 12.8.
# Older GPUs still work on CUDA 12.8 if the host NVIDIA driver is recent enough.
# Base ships Python 3.11, PyTorch 2.9.x, CUDA 12.8, cuDNN 9 (conda env at /opt/conda)
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

LABEL org.opencontainers.image.title="leucaena-earth-segmentation"
LABEL org.opencontainers.image.description="PyTorch ResUNet segmentation with GDAL for aerial/LiDAR GeoTIFFs"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CONDA_NO_PLUGINS=true \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PROJ_LIB=/opt/conda/share/proj \
    GDAL_DATA=/opt/conda/share/gdal \
    GTIFF_SRS_SOURCE=EPSG

WORKDIR /workspace

# GDAL + PDAL via conda-forge.
# - GDAL: rasters / vectors / warping (used by every step of the pipeline).
# - PDAL + python-pdal: needed by prep-lidar-rasters.py to turn LAZ point
#   clouds into CHM + INTENSITY GeoTIFFs. Adds ~500 MB but keeps the LiDAR
#   pipeline reproducible in the same image.
# PDAL pulls sqlite 3.32.x, which breaks Python/conda until we reinstall sqlite.
# Do not run `conda clean` here: it crashes while sqlite is broken. Wipe caches with rm.
RUN conda install -y -c conda-forge \
        gdal proj proj-data libstdcxx-ng \
        pdal python-pdal \
    && conda install -y -c conda-forge "sqlite>=3.45.0" "libsqlite>=3.45.0" \
    && python -c "import sqlite3; print('sqlite3 OK', sqlite3.sqlite_version)" \
    && rm -rf /opt/conda/pkgs/* /root/.conda/pkgs/* /root/.cache/conda/*

COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements-docker.txt \
    && rm /tmp/requirements-docker.txt

# Installing GDAL/PDAL via conda-forge upgrades libtiff to libtiff.so.6, which
# breaks the Pillow that shipped in the base image (it was linked against
# libtiff.so.5). Provide libtiff.so.5 at the OS level AND reinstall Pillow
# from a pip wheel (bundles its own libtiff) so matplotlib/torchmetrics work.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libtiff5 \
    && rm -rf /var/lib/apt/lists/* \
    && pip uninstall -y pillow || true \
    && conda remove -y --force pillow || true \
    && pip install --no-cache-dir --force-reinstall "pillow>=10,<11" \
    && python -c "from PIL import Image; import matplotlib; import torchmetrics; print('PIL OK:', Image.__file__)" \
    && python -c "import torch; print('PyTorch', torch.__version__, 'CUDA', torch.version.cuda)"

# Source code is bind-mounted at /workspace; copy only for standalone image builds
COPY . /workspace

CMD ["bash"]
