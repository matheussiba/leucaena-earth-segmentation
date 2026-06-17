"""
Global hyperparameters and naming for the leucaena-earth-segmentation project.

Tweak patch size / splits here; paths live in ``conf/paths.py``, CLI defaults in ``conf/default.py``.
"""
# --- Patch geometry ---
PATCH_SIZE = 256
PATCH_OVERLAP = 0.6 # 60% overlap between adjacent patches (step = (1 - PATCH_OVERLAP) * PATCH_SIZE).

# --- File prefixes for prepared numpy arrays ---
PREFIX_LABEL = 'label'
PREFIX_OPT = 'opt'
PREFIX_LIDAR = 'lidar'

# --- Classification ---
N_CLASSES = 2              # 0 = background, 1 = leucaena
IGNORE_INDEX = 255         # pixels outside the mapped area (unlabeled)
# torchmetrics / trainer use this name for pixels to skip in F1 (same value as IGNORE_INDEX).
DISCARDED_CLASS = IGNORE_INDEX
N_OPTICAL_BANDS = 4        # B, G, R, NIR
BAND_NAMES_OPTICAL = ['BLUE', 'GREEN', 'RED', 'NIR']
BAND_NAMES_LIDAR = ['CHM', 'INTENSITY']

# --- LiDAR rasterization + dataloader normalisation ---
# Native resolution (metres) used by PDAL when rasterising LAZ point clouds.
# Lower = denser raster but more no-data pixels; 1.0 m is a reasonable default
# for ~4 pts/m² aerial LiDAR. Output is always resampled to match the RGBN tile.
LIDAR_RASTER_RESOLUTION_M = 1.0
# Hard caps used by PatchFileDataset to scale raw LiDAR values to [0, 1] for the network.
# CHM beyond LIDAR_CHM_MAX_M is clipped (treetop spikes / building artefacts);
# INTENSITY beyond LIDAR_INTENSITY_MAX is clipped (raw uint16 sensor range).
LIDAR_CHM_MAX_M = 50.0
LIDAR_INTENSITY_MAX = 32768.0

# --- Label refinement inside annotated polygons (prep-patches-from-tiles.py) ---
# These thresholds are only applied when --lidar-dir is set, i.e. when CHM is
# available. A pixel inside an annotated polygon is labelled as leucaena (1)
# only if BOTH conditions hold:
#   CHM >= LEUCAENA_CHM_MIN_M  (filters grass / clearings / low shrubs)
#   NDVI >= LEUCAENA_NDVI_MIN  (filters towers / antennas / bare ground)
# Otherwise it becomes background (0). Pixels OUTSIDE every polygon are
# labelled IGNORE_INDEX (255) so the loss and metrics simply skip them.
#
# NDVI threshold calibrated to THIS imagery: the IGC-SP aerial orthophotos are
# 8-bit with a modest NIR response, so vegetation NDVI sits around 0.2-0.36
# (not the 0.6-0.9 of satellite reflectance). A textbook 0.3 cut sliced through
# the middle of real canopy and produced a "salt-and-pepper" label. With the
# height filter (>=4 m) already removing grass/shrubs, NDVI>=0.1 still rejects
# the non-vegetated tall surfaces (towers, roofs, bare ground sit near 0 or
# negative) while keeping the canopy continuous. See diag-bands.py /
# viz-ndvi-sweep.py for the diagnosis (band order confirmed correct: band 3 is
# the true NIR).
LEUCAENA_CHM_MIN_M = 4.0
LEUCAENA_NDVI_MIN = 0.1

# --- Data augmentation (training only, PatchFileDataset) ---
# Continuous random rotation in [0, AUG_ROTATION_DEG) degrees, applied with
# bilinear interpolation to optical/LiDAR and nearest-neighbour to the label.
# Translation is sampled uniformly in [-AUG_TRANSLATE_FRAC, +AUG_TRANSLATE_FRAC]
# of the patch side, in both x and y. Pixels rotated/translated out of frame
# are filled with 0 (image) or IGNORE_INDEX (label), so the loss never sees
# fabricated content.
AUG_ROTATION_DEG = 360.0
AUG_TRANSLATE_FRAC = 0.10

# --- Training hyperparameters ---
MAX_EPOCHS = 300

LEARNING_RATE = 1e-4
LEARNING_RATE_BETAS = (0.9, 0.999)
LEARNING_RATE_SCHEDULER_GAMMA = 0.995
LEARNING_RATE_SCHEDULER_MILESTONES = [5, 20]

# Class weights: [background, leucaena] — higher weight for leucaena (minority class)
CLASSES_WEIGHTS = [0.3, 0.7]

# --- Early stopping ---
# Warmup before the FIRST checkpoint can be saved. With min_epochs = 5 the
# EarlyStop skips saving during epochs 1-5 and writes the first model.pt at
# epoch 6 (see utils/trainer.py EarlyStop.testEpoch — first save happens at
# min_epochs + 1). Lowered from 20 so a usable checkpoint exists early, in
# case training is interrupted before convergence.
EARLY_STOP_MIN_EPOCHS = 5

EARLY_STOP_PATIENCE = 15

EARLY_STOP_MIN_DELTA = 0.00005

# --- Prediction ---
PREDICTION_OVERLAPS = [0, 0.25, 0.5]

# --- Train/test split ---
TEST_SPLIT = 0.2           # fraction of patches reserved for testing
VAL_SPLIT = 0.2            # fraction of train patches reserved for validation
