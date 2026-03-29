"""
Global hyperparameters and naming for the leucaena-earth-segmentation project.

Tweak patch size / splits here; paths live in ``conf/paths.py``, CLI defaults in ``conf/default.py``.
"""
# --- Patch geometry ---
PATCH_SIZE = 256
PATCH_OVERLAP = 0.5

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
