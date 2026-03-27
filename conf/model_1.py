"""Experiment 1: Optical only (B, G, R, NIR) — no LiDAR."""
from conf import general
from models.resunet import ResUnetOpt

def get_model():
    print('Model: Optical only (B, G, R, NIR)')
    lidar_bands = None
    depths = [32, 64, 128, 256]
    print(f'  Encoder depths: {depths}')
    model = ResUnetOpt(general.N_OPTICAL_BANDS, depths, general.N_CLASSES)
    return model, lidar_bands
