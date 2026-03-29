"""
Evaluate saved predictions against the prepared test label raster.

Inputs (expected on disk)
-------------------------
- ``experiments/exp_<e>/predicted/pred.npy`` — flat uint8 class id per pixel (same order as test label).
- ``prepared/label_test.npy`` — from ``prep-data.py``; pixels ``255`` (``IGNORE_INDEX``) are excluded.

Output
------
Writes a text report to ``experiments/exp_<e>/logs/eval_<e>.txt`` (stdout is redirected to that file).

Metrics are per-class precision / recall / F1 plus sklearn ``classification_report`` and confusion matrix.
"""
import argparse
import pathlib
import os
import sys

import numpy as np

# tqdm: progress bar — useful when evaluation iterates over many tiles or large arrays.
# Not strictly needed in the current implementation (numpy operations are vectorised),
# but kept here for when you add tile-by-tile or image-by-image evaluation loops.
from tqdm import tqdm

# classification_report: prints precision, recall, F1 and support per class in one call.
# confusion_matrix: shows TP, TN, FP, FN in a 2x2 table for binary classification.
from sklearn.metrics import classification_report, confusion_matrix

from conf import default, general, paths

# load_dict: reads the class remap dictionary saved by prep-data.py (pickle file).
# In binary segmentation this is trivial {0:0, 1:1}, but useful if you extend to
# multi-class experiments where some classes are merged or discarded.
from utils.ops import load_dict

parser = argparse.ArgumentParser(description='Evaluate experiment predictions')
parser.add_argument('-e', '--experiment', type=int, default=1)
parser.add_argument('-x', '--experiments-path', type=pathlib.Path, default=paths.PATH_EXPERIMENTS)

args = parser.parse_args()

# Must match the experiment id used for ``prediction.py`` and ``train.py``.
exp_path = os.path.join(str(args.experiments_path), f'exp_{args.experiment}')
logs_path = os.path.join(exp_path, 'logs')
predicted_path = os.path.join(exp_path, 'predicted')

outfile = os.path.join(logs_path, f'eval_{args.experiment}.txt')
with open(outfile, 'w') as sys.stdout:
    pred = np.load(os.path.join(predicted_path, 'pred.npy')).flatten()
    label = np.load(os.path.join(paths.PREPARED_PATH, f'{general.PREFIX_LABEL}_test.npy')).flatten()

    # Exclude unlabeled/ignore pixels
    valid = label != general.IGNORE_INDEX
    pred = pred[valid]
    label = label[valid]

    print(f'Valid pixels: {valid.sum():,} / {len(valid):,}')
    print()

    class_names = ['background', 'leucaena']
    for class_id, name in enumerate(class_names):
        tp = ((pred == class_id) & (label == class_id)).sum()
        fp = ((pred == class_id) & (label != class_id)).sum()
        fn = ((pred != class_id) & (label == class_id)).sum()
        tn = ((pred != class_id) & (label != class_id)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        acc = (tp + tn) / (tp + tn + fp + fn)

        print(f'{name:>12s}: Acc={100*acc:.2f}%  F1={100*f1:.2f}%  '
              f'Prec={100*precision:.2f}%  Rec={100*recall:.2f}%  '
              f'Samples={tp+fn:,}')

    print()
    print('Confusion matrix:')
    cm = confusion_matrix(label, pred, labels=[0, 1])
    print(f'  TN={cm[0,0]:,}  FP={cm[0,1]:,}')
    print(f'  FN={cm[1,0]:,}  TP={cm[1,1]:,}')
    print()
    print(classification_report(label, pred, target_names=class_names, digits=4))
