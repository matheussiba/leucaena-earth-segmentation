"""
Global hyperparameters and naming for the leucaena-earth-segmentation project.

Tweak patch size / splits here; paths live in ``conf/paths.py``, CLI defaults in ``conf/default.py``.
"""
# --- Patch geometry ---
PATCH_SIZE = 256
# 0.6 = 60% overlap between adjacent patches (step = 0.4 * PATCH_SIZE).
# Higher than the previous 0.5 default to give the network more views of each
# leucaena region (professor's suggestion to reduce class confusion).
PATCH_OVERLAP = 0.6

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
LEUCAENA_CHM_MIN_M = 4.5
LEUCAENA_NDVI_MIN = 0.3

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
EARLY_STOP_MIN_EPOCHS = LEARNING_RATE_SCHEDULER_MILESTONES[-1]

EARLY_STOP_PATIENCE = 15

EARLY_STOP_MIN_DELTA = 0.00005

# --- Prediction ---
PREDICTION_OVERLAPS = [0, 0.25, 0.5]

# --- Train/test split ---
TEST_SPLIT = 0.2           # fraction of patches reserved for testing
VAL_SPLIT = 0.2            # fraction of train patches reserved for validation
