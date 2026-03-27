"""Experiment 3: Late fusion — separate encoders for Optical and LiDAR."""
from conf import general
from models.resunet import LateFusion

def get_model():
    print('Model: Late fusion (Optical | LiDAR)')
    lidar_bands = list(range(len(general.BAND_NAMES_LIDAR)))
    input_depth_0 = general.N_OPTICAL_BANDS
    input_depth_1 = len(lidar_bands)
    depths = [32, 64, 128, 256]
    print(f'  Encoder depths: {depths}')
    print(f'  Optical bands: {input_depth_0}, LiDAR bands: {input_depth_1}')
    model = LateFusion(input_depth_0, input_depth_1, depths, general.N_CLASSES)
    return model, lidar_bands
