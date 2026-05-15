# CUDA-enabled PyTorch image for leucaena-earth-segmentation
# Base ships Python 3.11, PyTorch 2.4.x, CUDA 12.4, cuDNN 9 (conda env at /opt/conda)
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel

LABEL org.opencontainers.image.title="leucaena-earth-segmentation"
LABEL org.opencontainers.image.description="PyTorch ResUNet segmentation with GDAL for aerial/LiDAR GeoTIFFs"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PROJ_LIB=/opt/conda/share/proj \
    GDAL_DATA=/opt/conda/share/gdal \
    GTIFF_SRS_SOURCE=EPSG

WORKDIR /workspace

# GDAL via conda-forge (matches conda's libstdc++; apt+pip GDAL breaks on this base image)
RUN conda install -y -c conda-forge gdal proj proj-data libstdcxx-ng \
    && conda clean -afy

COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir -r /tmp/requirements-docker.txt \
    && rm /tmp/requirements-docker.txt

# Source code is bind-mounted at /workspace; copy only for standalone image builds
COPY . /workspace

CMD ["bash"]
