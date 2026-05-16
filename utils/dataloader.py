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
import csv
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


class PatchFileDataset(Dataset):
    """Reads patch triplets produced by ``prep-patches-from-tiles.py``.

    Storage layout (relative to ``patches_dir``)::

        opt/<patch_id>.npy     uint8    (H, W, 4)   BGRN order
        lbl/<patch_id>.npy     uint8    (H, W)      0/1
        lidar/<patch_id>.npy   float32  (H, W, k)   CHM, INTENSITY, ...   (optional)
        manifest.csv           row per patch with ``split`` and ``has_lidar`` columns

    The returned contract matches :class:`TreeTrainDataSet`::

        ((opt_tensor, lidar_tensor), label_tensor)

    LiDAR handling
    --------------
    - If a ``lidar/<patch_id>.npy`` file exists (produced by passing
      ``--lidar-dir`` to ``prep-patches-from-tiles.py``), it is loaded,
      band-selected via ``lidar_bands``, and scaled with the fixed clip
      constants in ``conf.general`` so that the network input lives in
      ``[0, 1]``.
    - If it does not exist, a zero LiDAR tensor of shape ``(1, H, W)`` is
      returned. This keeps experiment 1 (optical-only) working unchanged.

    ``has_lidar`` is read from the manifest when present; otherwise the
    presence of the ``.npy`` file is the authoritative signal.
    """

    def __init__(
        self,
        manifest_path,
        split,
        device,
        patches_dir=None,
        data_aug=False,
        transformer=ToTensor(),
        lidar_bands=None,
        cache_in_ram: bool = False,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"Unknown split: {split!r}")
        if patches_dir is None:
            patches_dir = os.path.dirname(manifest_path) or paths.PATH_PATCHES_DIR
        self.patches_dir = patches_dir
        self.split = split
        self.device = device
        self.data_aug = data_aug
        self.transformer = transformer
        self.lidar_bands = lidar_bands

        with open(manifest_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            self.records = [row for row in reader if row.get("split") == split]
        if not self.records:
            raise RuntimeError(
                f"No patches with split={split!r} found in {manifest_path}"
            )

        self._opt_dir = os.path.join(patches_dir, "opt")
        self._lbl_dir = os.path.join(patches_dir, "lbl")
        self._lidar_dir = os.path.join(patches_dir, "lidar")
        self._lidar_dir_exists = os.path.isdir(self._lidar_dir)
        # First label tells us the number of classes for the trainer header.
        first_lbl = np.load(
            os.path.join(self._lbl_dir, f"{self.records[0]['patch_id']}.npy")
        )
        self.n_classes = int(np.unique(first_lbl).size)
        if self.n_classes < general.N_CLASSES:
            # Background-only patches still count toward the binary task.
            self.n_classes = general.N_CLASSES

        self._cache: dict[str, tuple] | None = None
        if cache_in_ram:
            self._build_ram_cache()

    def _has_lidar_for(self, rec: dict, patch_id: str) -> bool:
        """Authoritative check: prefer manifest column, fall back to filesystem."""
        if not self._lidar_dir_exists:
            return False
        flag = rec.get("has_lidar")
        if flag is not None and flag != "":
            return str(flag).strip().lower() in ("true", "1", "yes")
        return os.path.isfile(os.path.join(self._lidar_dir, f"{patch_id}.npy"))

    def _build_ram_cache(self) -> None:
        """Load all patches of this split into RAM (faster epochs, ~300 KB/patch)."""
        from tqdm import tqdm

        self._cache = {}
        n = len(self.records)
        bytes_loaded = 0
        for rec in tqdm(self.records, desc=f'RAM cache [{self.split}]', unit='patch'):
            patch_id = rec['patch_id']
            opt = np.load(os.path.join(self._opt_dir, f'{patch_id}.npy'))
            lbl = np.load(os.path.join(self._lbl_dir, f'{patch_id}.npy'))
            lidar = None
            if self._has_lidar_for(rec, patch_id):
                lidar = np.load(os.path.join(self._lidar_dir, f'{patch_id}.npy'))
                bytes_loaded += lidar.nbytes
            self._cache[patch_id] = (opt, lbl, lidar)
            bytes_loaded += opt.nbytes + lbl.nbytes
        print(
            f'Cached {n:,} patches for split={self.split!r} '
            f'({bytes_loaded / (1024 ** 2):.0f} MiB RAM)'
        )

    def _scale_lidar(self, lidar_arr: np.ndarray) -> np.ndarray:
        """Apply fixed [0, 1] scaling per LiDAR band.

        Band order on disk matches ``general.BAND_NAMES_LIDAR``:
        ``[CHM, INTENSITY, ...]``. CHM is divided by ``LIDAR_CHM_MAX_M``;
        every other band by ``LIDAR_INTENSITY_MAX``. Values are clipped to
        ``[0, 1]`` so the network sees a well-defined range regardless of
        a few outlier pixels left over from rasterisation.
        """
        scaled = np.empty_like(lidar_arr, dtype=np.float32)
        for b in range(lidar_arr.shape[-1]):
            if b < len(general.BAND_NAMES_LIDAR) and general.BAND_NAMES_LIDAR[b].upper() == "CHM":
                denom = float(general.LIDAR_CHM_MAX_M)
            else:
                denom = float(general.LIDAR_INTENSITY_MAX)
            scaled[:, :, b] = np.clip(lidar_arr[:, :, b].astype(np.float32) / denom, 0.0, 1.0)
        return scaled

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        patch_id = rec["patch_id"]

        if self._cache is not None:
            opt_uint8, lbl_uint8, lidar_arr = self._cache[patch_id]
        else:
            opt_uint8 = np.load(os.path.join(self._opt_dir, f"{patch_id}.npy"))
            lbl_uint8 = np.load(os.path.join(self._lbl_dir, f"{patch_id}.npy"))
            lidar_arr = None
            if self._has_lidar_for(rec, patch_id):
                lidar_arr = np.load(os.path.join(self._lidar_dir, f"{patch_id}.npy"))

        # /255.0 -> float32 in [0, 1]; ToTensor will move HWC -> CHW.
        opt_float = (opt_uint8.astype(np.float32) / 255.0)
        opt_tensor = self.transformer(opt_float).to(self.device)

        if lidar_arr is not None:
            scaled = self._scale_lidar(lidar_arr)
            if self.lidar_bands is not None:
                scaled = scaled[:, :, self.lidar_bands]
            lidar_tensor = self.transformer(scaled).to(self.device)
        else:
            # Zero LiDAR placeholder keeps the (opt, lidar) tuple contract.
            lidar_tensor = torch.zeros(
                (1, opt_uint8.shape[0], opt_uint8.shape[1]),
                dtype=torch.float32,
                device=self.device,
            )

        label_tensor = torch.tensor(lbl_uint8.astype(np.int64)).to(self.device)

        if self.data_aug:
            k = random.randint(0, 3)
            opt_tensor = torch.rot90(opt_tensor, k, (1, 2))
            lidar_tensor = torch.rot90(lidar_tensor, k, (1, 2))
            label_tensor = torch.rot90(label_tensor, k, (0, 1))
            if bool(random.getrandbits(1)):
                opt_tensor = hflip(opt_tensor)
                lidar_tensor = hflip(lidar_tensor)
                label_tensor = hflip(label_tensor)
            if bool(random.getrandbits(1)):
                opt_tensor = vflip(opt_tensor)
                lidar_tensor = vflip(lidar_tensor)
                label_tensor = vflip(label_tensor)

        return ((opt_tensor, lidar_tensor), label_tensor)


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
