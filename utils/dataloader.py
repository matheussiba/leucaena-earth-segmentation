"""
PyTorch datasets for training and full-scene prediction.

Contract
--------
Training: ``__getitem__`` returns ``((optical, lidar), label)`` where each tensor is
``(C, H, W)``. The ResUNet forward in this repo expects a tuple ``x[0], x[1]``.

Prediction: returns only ``(optical, lidar)``; labels are not loaded.

Patch indices in ``train_patches.npy`` / ``val_patches.npy`` are produced by
``prep-data.py`` (sliding windows over linear indices).
"""
# Dataset base class from PyTorch — all custom datasets must inherit from this
# and implement __len__ and __getitem__ so DataLoader can iterate over them.
from torch.utils.data import Dataset

# ToTensor converts a numpy HWC array (Height x Width x Channels) to a
# PyTorch CHW tensor (Channels x Height x Width) and normalises uint8 to [0, 1].
from torchvision.transforms import ToTensor

# one_hot converts an integer class label (0, 1, ...) to a binary vector.
# Example: one_hot(tensor(1), num_classes=2) → [0, 1]
# Not used in the current forward pass (CrossEntropyLoss accepts class indices directly),
# but useful if you switch to BCELoss or want to inspect class probabilities as vectors.
from torch.nn.functional import one_hot

import torch
import numpy as np
import os
from conf import paths, general
import random

# hflip / vflip: horizontal and vertical flip augmentations from torchvision.
# Applied to image AND label identically so spatial alignment is preserved.
from torchvision.transforms.functional import hflip, vflip

# view_as_windows: creates a sliding-window view of an array without copying data.
# Used here to extract patch index grids over the full scene.
# view_as_blocks: similar but uses non-overlapping blocks — an alternative to
# sliding windows, useful if you want perfectly tiled (non-overlapping) patches.
from skimage.util import view_as_windows, view_as_blocks

class TreeTrainDataSet(Dataset):
    """Loads prepared full scenes once, then indexes patches by precomputed window indices."""

    def __init__(self, path_to_patches, device, data_aug = False, transformer = ToTensor(), lidar_bands = None) -> None:
        opt_img = np.load(os.path.join(paths.PREPARED_PATH, f'{general.PREFIX_OPT}_img.npy'))
        lidar_img = np.load(os.path.join(paths.PREPARED_PATH, f'{general.PREFIX_LIDAR}_img.npy'))
        if lidar_bands is not None:
            lidar_img = lidar_img[:, :, lidar_bands]

        self.opt_img = opt_img.reshape((-1, opt_img.shape[-1]))
        self.lidar_img = lidar_img.reshape((-1, lidar_img.shape[-1]))

        # Both lines load the label raster. The commented version keeps a (N, 1) shape
        # which was needed when using BCELoss. The active version uses flatten() for
        # CrossEntropyLoss which expects a 1-D integer class index per pixel.
        #self.labels = np.load(os.path.join(paths.PREPARED_PATH, f'{general.PREFIX_LABEL}_train.npy')).reshape((-1,1)).astype(np.int64)
        self.labels = np.load(os.path.join(paths.PREPARED_PATH, f'{general.PREFIX_LABEL}_train.npy')).flatten().astype(np.int64)
        self.n_classes = np.unique(self.labels).shape[0]
        # [:200] is a debug slice — uncomment to test the pipeline with only 200 patches
        # before running a full training job (much faster iteration).
        self.patches = np.load(path_to_patches)#[:200]
        self.transformer = transformer
        self.data_aug = data_aug

        self.device = device

    def __len__(self):
        return self.patches.shape[0]

    def __getitem__(self, index):
        patch_idx = self.patches[index]
        # ToTensor adds a channel dimension → shape becomes (C, H, W).
        opt_tensor = self.transformer(self.opt_img[patch_idx]).to(self.device)
        lidar_tensor = self.transformer(self.lidar_img[patch_idx]).to(self.device)

        # The commented line uses ToTensor on the label (produces float with a channel dim).
        # The active line uses torch.tensor directly — labels stay as int64 (2D, H x W),
        # which is what CrossEntropyLoss expects.
        #label_tensor = self.transformer(self.labels[patch_idx].astype(np.int64)).squeeze(0).to(self.device)
        label_tensor = torch.tensor(self.labels[patch_idx]).to(self.device)

        # Same geometric transform on opt, lidar, and label so alignment is preserved.
        if self.data_aug:
            k = random.randint(0, 3)
            opt_tensor = torch.rot90(opt_tensor, k, (1,2))
            lidar_tensor = torch.rot90(lidar_tensor, k, (1,2))
            label_tensor = torch.rot90(label_tensor, k, (0,1))

            if bool(random.getrandbits(1)):
                opt_tensor = hflip(opt_tensor)
                lidar_tensor = hflip(lidar_tensor)
                label_tensor = hflip(label_tensor)

            if bool(random.getrandbits(1)):
                opt_tensor = vflip(opt_tensor)
                lidar_tensor = vflip(lidar_tensor)
                label_tensor = vflip(label_tensor)

        return (
            (
                opt_tensor,
                lidar_tensor
            ),
            label_tensor
        )


class TreePredDataSet(Dataset):
    """Dense sliding-window extraction over the padded prepared scene (inference only)."""

    def __init__(self, device, overlap = 0, transformer = ToTensor(), lidar_bands = None) -> None:
        opt_img = np.load(os.path.join(paths.PREPARED_PATH, f'{general.PREFIX_OPT}_img.npy'))
        lidar_img = np.load(os.path.join(paths.PREPARED_PATH, f'{general.PREFIX_LIDAR}_img.npy'))

        if lidar_bands is not None:
            lidar_img = lidar_img[:, :, lidar_bands]

        self.transformer = transformer
        self.device = device
        self.original_shape = opt_img.shape[:2]

        # Reflect-pad by one patch on each side so windows near the border still have full size.
        pad_shape = ((general.PATCH_SIZE, general.PATCH_SIZE),(general.PATCH_SIZE, general.PATCH_SIZE),(0,0))

        opt_img = np.pad(opt_img, pad_shape, mode = 'reflect')
        lidar_img = np.pad(lidar_img, pad_shape, mode = 'reflect')
        shape = opt_img.shape[:2]

        window_step = int(general.PATCH_SIZE*(1-overlap))

        idx = np.arange(shape[0]*shape[1]).reshape(shape)
        self.padded_shape = shape

        self.idx_patches = view_as_windows(idx, (general.PATCH_SIZE, general.PATCH_SIZE), window_step).reshape((-1, general.PATCH_SIZE, general.PATCH_SIZE))

        self.opt_img = opt_img.reshape((-1, opt_img.shape[-1]))
        self.lidar_img = lidar_img.reshape((-1, lidar_img.shape[-1]))

    def __len__(self):
        return self.idx_patches.shape[0]

    def __getitem__(self, index):
        patch_idx = self.idx_patches[index]
        opt_tensor = self.transformer(self.opt_img[patch_idx]).to(self.device)
        lidar_tensor = self.transformer(self.lidar_img[patch_idx]).to(self.device)

        return (
                opt_tensor,
                lidar_tensor
            )
