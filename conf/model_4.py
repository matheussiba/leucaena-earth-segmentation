"""
Experiment 4 — Early fusion + NDVI.

Same architecture as exp_2 (ResUnet early fusion), but NDVI is computed
on-the-fly by PatchFileDataset and appended as a 5th optical channel.

Input stack seen by the encoder (7 channels total):
    x[0] : B, G, R, NIR, NDVI   (optical + derived vegetation index)
    x[1] : CHM, INTENSITY        (LiDAR)

NDVI = (NIR - RED) / (NIR + RED), scaled to [-1, 1].

Rationale: giving the model an explicit vegetation signal on top of the raw
spectral bands should help it distinguish leucaena canopy (high NDVI, high
CHM) from spectrally similar background (lower NDVI or lower CHM), without
requiring any extra data preparation beyond what exp_2 already uses.

COMPUTE_NDVI = True tells train.py / inspect_validation_errors.py to enable
on-the-fly NDVI computation in PatchFileDataset.__getitem__.
"""
from conf import general
from models.resunet import ResUnet

COMPUTE_NDVI = True


def get_model():
    print('Model: Early fusion + NDVI (B, G, R, NIR, NDVI | CHM, INTENSITY)')
    lidar_bands = list(range(len(general.BAND_NAMES_LIDAR)))
    input_depth_0 = general.N_OPTICAL_BANDS + 1  # +1 for NDVI channel
    input_depth_1 = len(lidar_bands)              # CHM + INTENSITY
    depths = [32, 64, 128, 256]
    print(f'  Encoder depths   : {depths}')
    print(f'  Optical channels : {input_depth_0}  (BGRN + NDVI)')
    print(f'  LiDAR channels   : {input_depth_1}  (CHM, INTENSITY)')
    print(f'  Total input      : {input_depth_0 + input_depth_1} channels')
    model = ResUnet(input_depth_0, input_depth_1, depths, general.N_CLASSES)
    return model, lidar_bands
