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

# one_hot converts an integer class label (0, 1, ...) to a binary vector.
# Example: one_hot(tensor(1), num_classes=2) → [0, 1]
# Not used in the current forward pass (CrossEntropyLoss accepts class indices directly),
# but useful if you switch to BCELoss or want to inspect class probabilities as vectors.
from torch.nn.functional import one_hot

import torch
import torch.nn.functional as F
import numpy as np
import os
import csv
from conf import paths, general
import random

# view_as_windows: creates a sliding-window view of an array without copying data.
# Used here to extract patch index grids over the full scene.
# view_as_blocks: similar but uses non-overlapping blocks — an alternative to
# sliding windows, useful if you want perfectly tiled (non-overlapping) patches.
from skimage.util import view_as_windows, view_as_blocks


class ToTensor:
    """Small torchvision-free replacement for transforms.ToTensor.

    Converts HWC numpy arrays to CHW float tensors. uint8 inputs are scaled to
    [0, 1]; float inputs keep their existing scale.
    """

    def __call__(self, array):
        tensor = torch.as_tensor(array)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        elif tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)
        if tensor.dtype == torch.uint8:
            tensor = tensor.float().div(255.0)
        else:
            tensor = tensor.float()
        return tensor.contiguous()


class InterpolationMode:
    BILINEAR = "bilinear"
    NEAREST = "nearest"


def hflip(tensor: torch.Tensor) -> torch.Tensor:
    return torch.flip(tensor, dims=(-1,))


def vflip(tensor: torch.Tensor) -> torch.Tensor:
    return torch.flip(tensor, dims=(-2,))


def affine(
    tensor: torch.Tensor,
    *,
    angle: float,
    translate: list[int],
    scale: float,
    shear: list[float],
    interpolation: str,
    fill: float,
) -> torch.Tensor:
    """Torch-only affine transform for CHW/HW tensors.

    This avoids importing torchvision, whose PIL dependency can require a
    libtiff ABI that is not present in the Docker image.
    """
    if shear != [0.0, 0.0]:
        raise ValueError("Shear augmentation is not supported by this lightweight affine helper.")
    squeeze_channel = tensor.ndim == 2
    x = tensor.unsqueeze(0) if not squeeze_channel else tensor.unsqueeze(0).unsqueeze(0)
    original_dtype = x.dtype
    x = x.float()

    _, _, h, w = x.shape
    tx, ty = translate
    radians = -float(angle) * np.pi / 180.0
    cos_a = float(np.cos(radians)) / float(scale)
    sin_a = float(np.sin(radians)) / float(scale)
    theta = torch.tensor(
        [[cos_a, -sin_a, -2.0 * tx / max(w, 1)], [sin_a, cos_a, -2.0 * ty / max(h, 1)]],
        dtype=torch.float32,
        device=x.device,
    ).unsqueeze(0)

    grid = F.affine_grid(theta, size=x.shape, align_corners=False)
    mode = "nearest" if interpolation == InterpolationMode.NEAREST else "bilinear"
    out = F.grid_sample(x, grid, mode=mode, padding_mode="zeros", align_corners=False)
    if fill != 0:
        valid = F.grid_sample(
            torch.ones((1, 1, h, w), dtype=torch.float32, device=x.device),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=False,
        )
        out = torch.where(valid > 0.5, out, torch.full_like(out, float(fill)))

    out = out.to(original_dtype) if original_dtype.is_floating_point else out.round().to(original_dtype)
    out = out.squeeze(0)
    return out.squeeze(0) if squeeze_channel else out

def _augment_opt_lidar_label(
    opt_tensor: torch.Tensor,
    lidar_tensor: torch.Tensor,
    label_tensor: torch.Tensor,
    *,
    rotation_deg: float,
    translate_frac: float,
    ignore_index: int,
):
    """Apply matching random rotation + translation + flips to all three tensors.

    - ``rotation_deg``: max rotation magnitude in degrees. Angle is sampled
      uniformly in ``[0, rotation_deg)``. Set to 0 to disable rotation.
    - ``translate_frac``: max translation magnitude as a fraction of the
      patch side. Translation is sampled independently for x and y in
      ``[-translate_frac, +translate_frac]``. Set to 0 to disable.
    - ``ignore_index``: value used to fill the label outside the original
      frame (so the loss skips those fabricated pixels).

    Optical / LiDAR get bilinear interpolation with fill=0; the label gets
    nearest-neighbour with fill=ignore_index. Horizontal/vertical flips are
    applied independently with 50% probability each (free augmentation,
    no resampling artefacts).
    """
    H = opt_tensor.shape[-2]
    W = opt_tensor.shape[-1]

    angle = float(random.uniform(0.0, float(rotation_deg))) if rotation_deg > 0 else 0.0
    if translate_frac > 0:
        tx = int(round(random.uniform(-translate_frac, translate_frac) * W))
        ty = int(round(random.uniform(-translate_frac, translate_frac) * H))
    else:
        tx, ty = 0, 0

    needs_affine = (angle != 0.0) or (tx != 0) or (ty != 0)
    if needs_affine:
        # Label needs a leading channel dim for torchvision.affine, then squeezed.
        label_in = label_tensor.unsqueeze(0)
        opt_tensor = affine(
            opt_tensor,
            angle=angle,
            translate=[tx, ty],
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        lidar_tensor = affine(
            lidar_tensor,
            angle=angle,
            translate=[tx, ty],
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        label_out = affine(
            label_in,
            angle=angle,
            translate=[tx, ty],
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=int(ignore_index),
        )
        label_tensor = label_out.squeeze(0)

    if bool(random.getrandbits(1)):
        opt_tensor = hflip(opt_tensor)
        lidar_tensor = hflip(lidar_tensor)
        label_tensor = hflip(label_tensor)

    if bool(random.getrandbits(1)):
        opt_tensor = vflip(opt_tensor)
        lidar_tensor = vflip(lidar_tensor)
        label_tensor = vflip(label_tensor)

    return opt_tensor, lidar_tensor, label_tensor


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
        compute_ndvi: bool = False,
    ) -> None:
        # When True, NDVI = (NIR - RED) / (NIR + RED) is computed on-the-fly
        # from the optical patch (BGRN order: R=2, NIR=3) and appended as a
        # 5th optical channel. The model input then has N_OPTICAL_BANDS + 1
        # channels in x[0]. Enable via COMPUTE_NDVI = True in the model config.
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

        self.compute_ndvi = compute_ndvi

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

        if self.compute_ndvi:
            # BGRN stored order: B=0, G=1, R=2, NIR=3
            red = opt_float[:, :, 2]
            nir = opt_float[:, :, 3]
            denom = nir + red
            denom = np.where(denom == 0.0, 1e-6, denom)
            ndvi = np.clip((nir - red) / denom, -1.0, 1.0).astype(np.float32)
            opt_float = np.concatenate([opt_float, ndvi[:, :, np.newaxis]], axis=-1)

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
            opt_tensor, lidar_tensor, label_tensor = _augment_opt_lidar_label(
                opt_tensor,
                lidar_tensor,
                label_tensor,
                rotation_deg=float(general.AUG_ROTATION_DEG),
                translate_frac=float(general.AUG_TRANSLATE_FRAC),
                ignore_index=int(general.IGNORE_INDEX),
            )

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
