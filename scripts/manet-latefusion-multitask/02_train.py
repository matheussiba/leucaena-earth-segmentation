"""
Multitask Training Script for Mangrove Species Mapping
=======================================================
Tasks:
  1. CHM Regression  (dist head)   → predicts canopy height from 'dist' folder
  2. Semantic Segmentation (label head) → predicts species labels from 'label' folder

Supported architectures:
  unet | unetplusplus | manet | linknet | BPATUNet

Usage example:
  python train_multitask.py \
      --arch unet \
      --channels 4 \
      --epochs 100 \
      --batch_size 8 \
      --lr 1e-4 \
      --num_classes 3 \
      --ignore_index 0 \
      --multitask \
      --pretrained
"""

import os
import sys
import argparse
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import segmentation_models_pytorch as smp
from PIL import Image
import tifffile
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────────────────────
# 0.  Reproducibility
# ─────────────────────────────────────────────────────────────
def set_seed(seed: int = 42, cudnn_benchmark: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic=True + benchmark=False → reprodutível, porém mais lento.
    # Com patch de tamanho FIXO, benchmark=True deixa o cuDNN escolher os
    # kernels de convolução mais rápidos (acelera após a 1ª iteração), ao custo
    # de perder o determinismo exato. Controlável via --cudnn_benchmark.
    if cudnn_benchmark:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────
# 1.  Utility image processing helpers
# ─────────────────────────────────────────────────────────────
def scale_rgb_contrast_8bit(img_chw: np.ndarray) -> np.ndarray:
    """Stretch each of the first 3 channels to [0, 1] (2-98 percentile) for display."""
    out = np.zeros((img_chw.shape[1], img_chw.shape[2], 3), dtype=np.float32)
    for c in range(3):
        band = img_chw[c].astype(np.float32)
        p2, p98 = np.percentile(band, [2, 98])
        if p98 > p2:
            band = np.clip((band - p2) / (p98 - p2), 0, 1)
        else:
            band = np.zeros_like(band)
        out[..., c] = band
    return out


def linear_stretch_2_98(arr_bhw: np.ndarray) -> np.ndarray:
    """Per-image 2-98 percentile stretch → [0,1]."""
    result = []
    for img in arr_bhw:
        img = img.astype(np.float32)
        p2, p98 = np.percentile(img, [2, 98])
        if p98 > p2:
            img = np.clip((img - p2) / (p98 - p2), 0, 1)
        else:
            img = np.zeros_like(img)
        result.append(img)
    return np.array(result)


# ─────────────────────────────────────────────────────────────
# 2.  Plot function (provided, integrated)
# ─────────────────────────────────────────────────────────────
def plot_figures(img, ref, pred, depth, dist, model_dir, epoch, set_name,
                 multitask=False, task=None, aug_target=None):
    """
    Plota as figuras de diagnóstico de acordo com a TAREFA.

    O conjunto de linhas depende de `task`:
      • 'regression'     → RGB | CHM referência | CHM previsto
      • 'classification' → RGB | labels referência | labels previstos
      • 'multitask'      → RGB | CHM referência | labels referência |
                           CHM previsto | labels previstos

    Compatibilidade: se `task` não for passado, deriva-se de `multitask`
    (True → 'multitask', False → 'regression').

    Parâmetros de dados (podem ser None conforme a task):
      img   : (B, C, H, W)   imagem de entrada (sempre; RGB se C ≥ 3, senão banda 1)
      depth : (B, H, W)      CHM de referência (regressão / multitask)
      dist  : (B, 1, H, W)   CHM previsto      (regressão / multitask)
      ref   : (B, H, W)      labels de referência (classificação / multitask)
      pred  : (B, C, H, W)   logits previstos    (classificação / multitask)
    """
    if task is None:
        task = 'multitask' if multitask else 'regression'

    show_reg = task in ('regression', 'multitask')
    show_clf = task in ('classification', 'multitask')

    def _to_np(x):
        if x is None:
            return None
        return x if type(x).__module__ == np.__name__ else x.data.cpu().numpy()

    img   = _to_np(img)
    depth = _to_np(depth) if show_reg else None
    dist  = _to_np(dist)  if show_reg else None
    ref   = _to_np(ref)   if show_clf else None
    pred  = _to_np(pred)  if show_clf else None

    batch = min(5, img.shape[0])
    ncols = batch

    # ── Imagem de entrada (sempre): composição RGB se houver ≥ 3 bandas,
    # senão só a 1ª banda em tons de cinza (stretch 2–98% por amostra) ──
    if img.shape[1] >= 3:
        img_vis = np.array([scale_rgb_contrast_8bit(im) for im in img[:batch, [0, 1, 2], :, :]])
        titulo_img, cmap_img = 'RGB composition', None
    else:
        img_vis = linear_stretch_2_98(img[:batch, 0, :, :])
        titulo_img, cmap_img = 'Input image (band 1)', 'gray'

    # ── Regressão (referência + previsto) ─────────────────────
    # IMPORTANTE: os mapas de distância/CHM já vêm normalizados em [0,1]
    # (normalização per-ITC na geração + sigmoid na predição), e têm FUNDO=0
    # dominante. Aplicar um stretch de percentil 2–98 aqui é destrutivo: quando
    # as copas cobrem < ~3% do patch, p2 e p98 caem ambos em 0 e o painel inteiro
    # zera (dados corretos, visualização apagada). Por isso visualizamos com um
    # simples clip [0,1], preservando fundo e contraste reais das copas.
    depth_gt_processed = dist_pred_processed = None
    aug_processed = None
    if show_reg:
        depth_gt = depth[:batch, :, :] if depth.ndim == 3 else depth[:batch, 0, :, :]
        depth_gt_processed = np.clip(depth_gt.astype(np.float32), 0, 1)
        if dist.ndim == 4:
            dist_pred = dist[:batch, 0, :, :]
        else:
            dist_pred = dist[:batch, :, :]
        dist_pred_processed = np.clip(dist_pred.astype(np.float32), 0, 1)
        # alvo aumentado (real + pseudo), se a pseudo-rotulagem estiver ativa
        aug = _to_np(aug_target)
        if aug is not None:
            aug_hw = aug[:batch, :, :] if aug.ndim == 3 else aug[:batch, 0, :, :]
            aug_processed = np.clip(aug_hw.astype(np.float32), 0, 1)

    # ── Classificação (referência + previsto) ─────────────────
    ref_processed = pred_processed = None
    discrete_cmap = all_unique = None
    if show_clf:
        # ref pode conter o valor de ignore (-100, nodata). Mapeia para 0
        # apenas para visualização, evitando overflow no astype(uint8).
        ref_vis = ref[:batch, :, :].astype(np.int64)
        ref_vis = np.where(ref_vis < 0, 0, ref_vis)
        ref_processed = ref_vis.astype(np.uint8)
        if pred.ndim == 4:
            pred_processed = np.argmax(pred[:batch, :, :, :], axis=1).astype(np.uint8)
        elif pred.dtype in (np.float32, np.float64):
            pred_processed = (pred[:batch, :, :] > 0.5).astype(np.uint8)
        else:
            pred_processed = pred[:batch, :, :].astype(np.uint8)

        all_unique = np.unique(np.concatenate([np.unique(ref_processed),
                                               np.unique(pred_processed)]))
        n_classes = len(all_unique)
        if n_classes <= 3:
            colors = ['black', 'blue', 'red'][:n_classes]
        elif n_classes <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, n_classes))
        discrete_cmap = ListedColormap(colors)

    # ── Monta a sequência de linhas conforme a task ───────────
    # cada item: (tipo, título)
    #   tipo ∈ {'rgb','chm_gt','chm_pred','lbl_gt','lbl_pred'}
    rows = [('rgb', titulo_img)]
    if task == 'regression':
        rows += [('chm_gt',   'Distance map (real)')]
        if aug_processed is not None:
            rows += [('chm_aug', 'Augmented target (real + pseudo)')]
        rows += [('chm_pred', 'Predicted distance map')]
    elif task == 'classification':
        rows += [('lbl_gt',   'Reference Labels'),
                 ('lbl_pred', 'Predicted Labels')]
    else:  # multitask
        rows += [('chm_gt',   'Distance map (real)')]
        if aug_processed is not None:
            rows += [('chm_aug', 'Augmented target (real + pseudo)')]
        rows += [('lbl_gt',   'Reference labels'),
                 ('chm_pred', 'Predicted distance map'),
                 ('lbl_pred', 'Predicted labels')]

    nrows = len(rows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             figsize=(ncols * 3, nrows * 2.5))
    axes = np.atleast_2d(axes)
    if nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    def _label_cbar(ax, im, text, ticks=False):
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(text, rotation=270, labelpad=15)
        if ticks:
            cbar.set_ticks(all_unique)
            cbar.set_ticklabels([f'Class {int(c)}' for c in all_unique])

    for r, (kind, title) in enumerate(rows):
        for col in range(ncols):
            ax = axes[r, col]
            ax.set_axis_off()

            if kind == 'rgb':
                if cmap_img is None:
                    ax.imshow(img_vis[col])
                else:
                    ax.imshow(img_vis[col], cmap=cmap_img, vmin=0, vmax=1)
            elif kind == 'chm_gt':
                im = ax.imshow(depth_gt_processed[col], cmap='magma', vmin=0, vmax=1)
                if col == ncols - 1:
                    _label_cbar(ax, im, 'Normalized distance')
            elif kind == 'chm_aug':
                # alvo aumentado em magma; pseudo-copas (aug>real) destacadas em ciano
                im = ax.imshow(aug_processed[col], cmap='magma', vmin=0, vmax=1)
                pseudo_only = (aug_processed[col] - depth_gt_processed[col]) > 0.05
                if pseudo_only.any():
                    overlay = np.zeros((*pseudo_only.shape, 4), dtype=np.float32)
                    overlay[pseudo_only] = [0.0, 1.0, 1.0, 0.45]   # ciano translúcido
                    ax.imshow(overlay)
                if col == ncols - 1:
                    _label_cbar(ax, im, 'Real + pseudo (cyan=pseudo)')
            elif kind == 'chm_pred':
                im = ax.imshow(dist_pred_processed[col], cmap='magma', vmin=0, vmax=1)
                if col == ncols - 1:
                    _label_cbar(ax, im, 'Predicted distance')
            elif kind == 'lbl_gt':
                im = ax.imshow(ref_processed[col], cmap=discrete_cmap,
                               vmin=all_unique.min(), vmax=all_unique.max())
                if col == ncols - 1:
                    _label_cbar(ax, im, 'Class label', ticks=True)
            elif kind == 'lbl_pred':
                im = ax.imshow(pred_processed[col], cmap=discrete_cmap,
                               vmin=all_unique.min(), vmax=all_unique.max())
                if col == ncols - 1:
                    _label_cbar(ax, im, 'Predicted class', ticks=True)

            # rótulo da linha (à esquerda) e título da coluna (no topo)
            if col == 0:
                ax.text(-0.1, 0.5, title, rotation=90, va='center',
                        transform=ax.transAxes, fontsize=10, fontweight='bold')
            if r == 0:
                ax.set_title(f'Sample {col + 1}', fontsize=12,
                             fontweight='bold', pad=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.94, hspace=0.15, wspace=0.1, left=0.08)

    os.makedirs(model_dir, exist_ok=True)
    fig_path = os.path.join(model_dir, f"{set_name}_epoch_{epoch}.pdf")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', format='pdf', facecolor='white')
    plt.close()
    print(f"Figure saved to: {fig_path}")

    if show_clf:
        print(f"Reference mask - unique values: {np.unique(ref_processed)}, dtype: {ref_processed.dtype}")
        print(f"Predicted labels - unique values: {np.unique(pred_processed)}, dtype: {pred_processed.dtype}")

    return fig_path


# ─────────────────────────────────────────────────────────────
# 3.  Dataset
# ─────────────────────────────────────────────────────────────
def _to_chw(arr: np.ndarray) -> np.ndarray:
    """Normaliza um array 2D/3D para (C, H, W) de forma robusta.

    O pipeline deste projeto SEMPRE salva patches em (C, H, W) — tanto RGB
    quanto LiDAR/CHM. O risco é confundir (C, H, W) com (H, W, C) quando o
    patch espacial é pequeno: p.ex. um CHM (1, 16, 16) não pode ser tratado
    como (H, W, 16). Regra adotada:
      • 2D                      → (1, H, W)
      • 3D com eixo0 == eixo1   → AMBÍGUO (quadrado); assume (C, H, W), que é
                                  o layout salvo pelo pipeline. Cobre (1,16,16).
      • 3D, último eixo pequeno E menor que os dois primeiros → (H, W, C)
      • caso contrário          → já está em (C, H, W)
    """
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError(f"Raster com ndim inesperado: {arr.ndim} (shape {arr.shape}).")

    c0, c1, c2 = arr.shape
    # (H, W, C) só quando o ÚLTIMO eixo é claramente o de canais:
    # pequeno (<=16) e estritamente menor que H e W (que são iguais ou maiores).
    if c2 <= 16 and c2 < c0 and c2 < c1:
        return arr.transpose(2, 0, 1)            # (H, W, C) → (C, H, W)
    # Demais casos (inclui o ambíguo quadrado, p.ex. (1,16,16) ou (C,H,W)):
    # já tratamos como (C, H, W) — layout padrão do pipeline.
    return arr


def load_raster(path: str) -> np.ndarray:
    """Load image/raster as float32 numpy array.
    Supports .npy, .tif/.tiff (via tifffile) and common image formats (via PIL).
    Returns shape (C, H, W).
    """
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        arr = np.load(path).astype(np.float32)
        arr = _to_chw(arr)                       # (C, H, W) robusto
    elif ext in ('.tif', '.tiff'):
        arr = tifffile.imread(path).astype(np.float32)
        arr = _to_chw(arr)
    else:
        img = Image.open(path).convert('RGB')
        arr = np.array(img, dtype=np.float32).transpose(2, 0, 1)  # (3, H, W)
    return arr


def load_label(path: str) -> np.ndarray:
    """Load a label mask as int64 numpy array with shape (H, W)."""
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        arr = np.load(path)
    elif ext in ('.tif', '.tiff'):
        arr = tifffile.imread(path)
    else:
        arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]   # take first channel if RGB/multi-band
    return arr.astype(np.int64)


class MangroveDataset(Dataset):
    """
    Expects the following folder structure under `root_dir`:
      root_dir/
        IMAGE/    ← multi-band input images  (*.npy, shape H×W×C or C×H×W)
        DIST/     ← Distance image          (*.npy, shape H×W or 1×H×W, float)
        LABEL/    ← species labels            (*.npy, shape H×W, int)

    Patches are already saved as numpy arrays.
    Input images are normalised by dividing all bands by 255.
    Files are matched by sorting; the same index must correspond to the
    same location across all three folders.
    """

    IMG_EXTS  = {'.npy', '.tif', '.tiff', '.png', '.jpg', '.jpeg'}

    def __init__(self, root_dir: str, multitask: bool = True,
                 augment: bool = False, latefusion: bool = False,
                 label_shift: int = 0, bg_ignore_value: int = -100,
                 require_dist: bool = True, norm_values=None):
        super().__init__()
        self.root_dir   = Path(root_dir)
        self.multitask  = multitask
        self.augment    = augment
        self.latefusion = latefusion
        # require_dist: quando False, a pasta DIST é OPCIONAL. Útil para a tarefa
        # de classificação pura (não há ramo de regressão, então o mapa de
        # distância não é necessário). Se a pasta não existir, o dataset devolve
        # um tensor de distância "dummy" (zeros) só para manter a interface dos
        # loops de treino/visualização; ele não entra em nenhuma loss de classif.
        self.require_dist = require_dist
        # norm_values: lista de tetos de normalização por banda do raster LiDAR.
        #   banda i → dividida por norm_values[i] e clampada em [0, 1].
        # Se None, cai no comportamento legado (CHM de 1 banda dividido por 30 m).
        self.norm_values = (np.asarray(norm_values, dtype=np.float32)
                            if norm_values is not None else None)
        # label_shift: somado aos rótulos no __getitem__.
        #   0  → mantém rótulos como estão (o que _resolve_ignore usa sempre).
        #  -1 → (legado) desloca 1,2,3,4 → 0,1,2,3 e descarta o fundo.
        self.label_shift     = label_shift
        # valor para onde o fundo (rótulo 0 antes do shift) é mapeado quando há
        # deslocamento. -100 = ignore_index padrão do CrossEntropy (não entra na loss).
        self.bg_ignore_value = bg_ignore_value

        self.img_dir   = self.root_dir / 'IMAGE'
        self.dist_dir  = self.root_dir / 'DIST'
        self.label_dir = self.root_dir / 'LABEL'
        self.lidar_dir = self.root_dir / 'LIDAR'

        # DIST só é obrigatória quando require_dist=True (regressão/multitask).
        self.has_dist = self.dist_dir.exists()

        required = [self.img_dir, self.label_dir]
        if self.require_dist:
            required.append(self.dist_dir)
        if self.latefusion:
            required.append(self.lidar_dir)
        for d in required:
            if not d.exists():
                raise FileNotFoundError(f"Directory not found: {d}")

        self.img_files   = sorted([p for p in self.img_dir.iterdir()
                                   if p.suffix.lower() in self.IMG_EXTS])
        self.label_files = sorted([p for p in self.label_dir.iterdir()
                                   if p.suffix.lower() in self.IMG_EXTS])

        n = len(self.img_files)
        assert len(self.label_files) == n, \
            f"Mismatch: {len(self.label_files)} label files vs {n} img files"

        if self.has_dist:
            self.dist_files = sorted([p for p in self.dist_dir.iterdir()
                                      if p.suffix.lower() in self.IMG_EXTS])
            assert len(self.dist_files) == n, \
                f"Mismatch: {len(self.dist_files)} dist files vs {n} img files"
        else:
            self.dist_files = None

        if self.latefusion:
            self.lidar_files = sorted([p for p in self.lidar_dir.iterdir()
                                       if p.suffix.lower() in self.IMG_EXTS])
            assert len(self.lidar_files) == n, \
                f"Mismatch: {len(self.lidar_files)} lidar files vs {n} img files"

        print(f"Dataset loaded: {n} samples from {root_dir}"
              f"{' (LateFusion: RGB + CHM)' if self.latefusion else ''}"
              f"{'' if self.has_dist else ' [sem DIST → mapa de distância dummy]'}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img   = load_raster(self.img_files[idx])   # (C, H, W)  float32
        label = load_label(self.label_files[idx])  # (H, W)     int64

        # DIST é opcional: se a pasta existir, carrega; senão, gera um mapa de
        # zeros (1, H, W) apenas para manter a interface (não entra na loss de
        # classificação).
        if self.has_dist:
            dist = load_raster(self.dist_files[idx])  # (1, H, W) float32
        else:
            h, w = label.shape[-2], label.shape[-1]
            dist = np.zeros((1, h, w), dtype=np.float32)

        # Input image: patches already saved as uint8 numpy arrays → divide by 255
        img = img / 255.0

        # ── CHM/LiDAR (resolução nativa, NÃO reamostrar aqui) ─────
        lidar = None
        if self.latefusion:
            lidar = load_raster(self.lidar_files[idx])   # (C, h, w) float32
            lidar = np.nan_to_num(lidar, nan=0.0)

            if self.norm_values is not None:
                # Normalização POR BANDA: cada banda i é dividida pelo teto
                # norm_values[i] e clampada em [0, 1]. Mantém a relação física
                # valor↔grandeza consistente entre patches e entre treino/predição.
                nv = self.norm_values
                c = lidar.shape[0]
                if nv.shape[0] != c:
                    raise ValueError(
                        f"--norm_values tem {nv.shape[0]} valores, mas o raster "
                        f"LiDAR tem {c} banda(s). Forneça um teto por banda.")
                # broadcast (C,1,1) sobre (C,h,w)
                lidar = lidar / (nv.reshape(c, 1, 1) + 1e-8)
                lidar = np.clip(lidar, 0.0, 1.0)
            else:
                # Comportamento legado: CHM de banda única dividido por um teto fixo
                # (altura máxima de mangue) — preserva contraste de altura em ~[0,1].
                lmax = 30  # this the maximum mangrove tree height
                if lmax > 1.0:
                    lidar = lidar / (lmax + 1e-8)

        # ── Normalização do mapa de distância → [0, 1] ────────────
        dist_max = dist.max()
        if dist_max > 1.0:
            dist = dist / (dist_max + 1e-8)

        if self.augment:
            img, dist, label, lidar = self._augment(img, dist, label, lidar)

        # ── Deslocamento de rótulos (legado; _resolve_ignore usa sempre 0) ──
        # Ex.: rótulos 1,2,3,4 (espécies) com 0=fundo → 0,1,2,3, e o fundo
        # (antes 0) vai para bg_ignore_value (-100), descartado pelo CE.
        if self.label_shift != 0:
            label = label.astype(np.int64)
            bg = (label == 0)
            label = label + self.label_shift          # 1..K → 0..K-1
            label[bg] = self.bg_ignore_value          # fundo → ignore

        img_t   = torch.from_numpy(img).float()
        dist_t  = torch.from_numpy(dist).float()
        label_t = torch.from_numpy(label).long()

        if self.latefusion:
            lidar_t = torch.from_numpy(lidar).float()
            # input é uma lista [rgb, lidar] (compatível com LateFusionGeneric)
            return [img_t, lidar_t], dist_t, label_t

        return img_t, dist_t, label_t

    # ── simple spatial augmentations (keep img/dist/label/lidar in sync) ──
    def _augment(self, img, dist, label, lidar=None):
        if random.random() > 0.5:                     # horizontal flip
            img   = img[:, :, ::-1].copy()
            dist  = dist[:, :, ::-1].copy()
            label = label[:, ::-1].copy()
            if lidar is not None: lidar = lidar[:, :, ::-1].copy()
        if random.random() > 0.5:                     # vertical flip
            img   = img[:, ::-1, :].copy()
            dist  = dist[:, ::-1, :].copy()
            label = label[::-1, :].copy()
            if lidar is not None: lidar = lidar[:, ::-1, :].copy()
        k = random.randint(0, 3)                      # 90° rotation
        if k > 0:
            img   = np.rot90(img,   k, axes=(1, 2)).copy()
            dist  = np.rot90(dist,  k, axes=(1, 2)).copy()
            label = np.rot90(label, k, axes=(0, 1)).copy()
            if lidar is not None: lidar = np.rot90(lidar, k, axes=(1, 2)).copy()
        return img, dist, label, lidar


# ─────────────────────────────────────────────────────────────
# 3b. Extração de patches ON-THE-FLY (estilo main_FINAL)
# ─────────────────────────────────────────────────────────────
# Em vez de ler patches pré-recortados de pastas, carrega os RASTERS
# GRANDES inteiros uma vez (RGB, label, distância e — em LateFusion —
# CHM/LiDAR) e recorta patches sob demanda, centrados em pixels
# rotulados, com oversampling das classes minoritárias. O número de
# patches por época é controlável (--samples). O contrato de saída é
# IDÊNTICO ao do MangroveDataset, então o restante do código (modelo,
# loss, métricas, visualização) não muda.

def _read_raster_full(path):
    """Lê um raster inteiro como np.ndarray.
    Retorna (C, H, W) float32 (sempre com dimensão de canal explícita).
    Usa rasterio (preserva ordem de bandas e nodata)."""
    import rasterio as rio
    with rio.open(path) as src:
        arr = src.read()                      # (C, H, W)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]                  # (1, H, W)
    return arr


def _read_raster_meta(path):
    """Retorna (transform, crs, height, width) de um raster."""
    import rasterio as rio
    with rio.open(path) as src:
        return src.transform, src.crs, src.height, src.width


def _same_grid(a, b, tol=0.01):
    """True se dois rasters abertos (rasterio) estão no mesmo grid: mesmo CRS,
    mesmo tamanho e mesma extensão (cantos a menos de `tol` pixel)."""
    if a.crs != b.crs or a.shape != b.shape:
        return False
    px = min(abs(a.transform.a), abs(a.transform.e))
    return np.allclose(tuple(a.bounds), tuple(b.bounds), rtol=0, atol=tol * px)


def _load_on_image_grid(image_path, ref_path, depth_path):
    """Carrega imagem, ref e depth num grid ÚNICO: o grid nativo da imagem.

    A saída da rede fica na resolução de --image_path (ex.: PAN 2 m). Se ref e
    depth já estão nesse grid, só lê os três. Senão (outro recorte, resolução
    ou CRS), a imagem é RECORTADA à área coberta pelo ref — sem reamostrar a
    imagem — e ref/depth são reprojetados para esse recorte por vizinho mais
    próximo (rótulos não podem ser interpolados; o depth usa o mesmo método p/
    continuar pixel a pixel coerente com o ref). Pixels do recorte fora da
    cobertura do ref recebem rótulo -1 (nodata → ignorado na loss e no sorteio
    de centros).

    Retorna (image (C,H,W) float32, ref (H,W), depth (H,W), transform, crs),
    com transform/crs do recorte da imagem (p/ localizar o CHM).
    """
    import rasterio as rio
    from rasterio.warp import reproject, transform_bounds, Resampling
    from rasterio.windows import Window

    with rio.open(image_path) as si, rio.open(ref_path) as sr, \
         rio.open(depth_path) as sd:
        if _same_grid(si, sr) and _same_grid(si, sd):
            return (_read_raster_full(image_path), _read_raster_full(ref_path)[0],
                    _read_raster_full(depth_path)[0], si.transform, si.crs)

        # área comum imagem ∩ ref, no CRS da imagem
        rb = (transform_bounds(sr.crs, si.crs, *sr.bounds)
              if sr.crs != si.crs else tuple(sr.bounds))
        ib = si.bounds
        left, bottom = max(ib.left, rb[0]), max(ib.bottom, rb[1])
        right, top   = min(ib.right, rb[2]), min(ib.top, rb[3])
        if right <= left or top <= bottom:
            raise ValueError("--image_path e --ref_path não se sobrepõem.")
        # janela inteira da imagem com os px cujo CENTRO cai na área comum
        inv = ~si.transform
        cs, rs = zip(*[inv * (x, y) for x in (left, right) for y in (bottom, top)])
        r0 = max(0, int(np.ceil(min(rs) - 0.5)))
        c0 = max(0, int(np.ceil(min(cs) - 0.5)))
        r1 = min(si.height, int(np.floor(max(rs) - 0.5)) + 1)
        c1 = min(si.width,  int(np.floor(max(cs) - 0.5)) + 1)
        win = Window(c0, r0, c1 - c0, r1 - r0)
        tf, crs = si.window_transform(win), si.crs
        image = si.read(window=win).astype(np.float32)      # (C, H, W)
        H, W = image.shape[1:]
        print(f"[on-the-fly] ref/depth fora do grid da imagem → imagem recortada "
              f"à área do ref (linhas {r0}:{r1}, colunas {c0}:{c1} → {H}x{W} px, "
              f"resolução nativa {abs(tf.a):.4g} x {abs(tf.e):.4g}); ref/depth "
              f"reamostrados p/ esse grid (vizinho mais próximo)...")

        def _warp(src, arr, dtype):
            dst = np.zeros((H, W), dtype=dtype)             # fora da cobertura → 0
            reproject(arr, dst, src_transform=src.transform, src_crs=src.crs,
                      dst_transform=tf, dst_crs=crs,
                      resampling=Resampling.nearest, num_threads=4)
            return dst

        # +1 antes de reprojetar: 0 no destino = fora da cobertura do ref → -1
        ref   = _warp(sr, sr.read(1).astype(np.int16) + 1, np.int16) - 1
        depth = _warp(sd, sd.read(1).astype(np.float32), np.float32)
    n_out = int((ref < 0).sum())
    if n_out:
        print(f"  [ref] {n_out:,} px do recorte fora da cobertura do ref → "
              f"rótulo -1 (ignorado).")
    return image, ref, depth, tf, crs


def _rgb_band_minmax(image):
    """Mínimo e máximo por banda dos pixels FINITOS (ignora NaN/nodata)."""
    vmin = np.array([np.nanmin(b) for b in image], dtype=np.float32)
    vmax = np.array([np.nanmax(b) for b in image], dtype=np.float32)
    return vmin, vmax


def _poly2(r, c):
    """Termos do polinômio de 2º grau em (row, col) do RGB (escala 1e-4 p/ um
    ajuste bem condicionado). Usado por _rgb_to_chm_map e no recorte."""
    r = np.asarray(r, dtype=np.float64) * 1e-4
    c = np.asarray(c, dtype=np.float64) * 1e-4
    return np.stack([np.ones_like(r), r, c, r * r, r * c, c * c], axis=-1)


def _rgb_to_chm_map(rgb_tf, rgb_crs, chm_tf, chm_crs, H, W, n=21):
    """Mapeamento (row, col) do grid RGB → (row, col) fracionário do grid CHM.

    Se os CRS diferem (ex.: RGB em graus WGS84 e CHM em UTM/metros), aplicar o
    transform inverso do CHM direto sobre coordenadas do RGB daria lixo. Aqui
    uma grade n×n de pontos de controle do RGB é reprojetada para o CRS do CHM
    e um polinômio de 2º grau é ajustado por mínimos quadrados (um afim deixa
    ~0.2 px de erro em ~40 km por causa da curvatura graus→UTM) — rápido por
    patch e serializável para os workers. Retorna (M (2, 6), erro_max_px):
    [row_chm, col_chm] = M @ _poly2(row_rgb, col_rgb)."""
    R, C = np.meshgrid(np.linspace(0, H, n), np.linspace(0, W, n), indexing='ij')
    R, C = R.ravel(), C.ravel()
    xs = rgb_tf.c + C * rgb_tf.a + R * rgb_tf.b
    ys = rgb_tf.f + C * rgb_tf.d + R * rgb_tf.e
    if rgb_crs is not None and chm_crs is not None and rgb_crs != chm_crs:
        from rasterio.warp import transform as _warp
        xs, ys = (np.asarray(v, dtype=np.float64) for v in _warp(rgb_crs, chm_crs, xs, ys))
    inv = ~chm_tf
    cc = inv.a * xs + inv.b * ys + inv.c
    rc = inv.d * xs + inv.e * ys + inv.f
    A = _poly2(R, C)
    target = np.stack([rc, cc], axis=1)
    M = np.linalg.lstsq(A, target, rcond=None)[0].T          # (2, 6)
    err = float(np.abs(A @ M.T - target).max())
    return M, err


def _oversample_coords(coords, labels, seed=42):
    """Replica as coordenadas das classes minoritárias até equiparar a
    classe majoritária (oversampling com reposição). Mesma ideia do
    `oversamp` do main_FINAL.

    coords : (N, 2) array de (row, col)
    labels : (N,)   array de rótulos inteiros (classe de cada coord)
    """
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) <= 1:
        return coords
    target = counts.max()
    out = []
    for cls in classes:
        idx = np.where(labels == cls)[0]
        if len(idx) == 0:
            continue
        if len(idx) < target:
            extra = rng.choice(idx, size=target - len(idx), replace=True)
            idx = np.concatenate([idx, extra])
        out.append(coords[idx])
    out = np.concatenate(out, axis=0)
    rng.shuffle(out)
    return out


def _sample_bg_coords(ref, image, margin, n, seed=42, max_iter=50, keep_fn=None):
    """Sorteia até `n` centros de FUNDO (ref == 0) respeitando a margem, por
    amostragem com rejeição — sem materializar np.where(ref == 0), que em
    rasters grandes teria centenas de milhões de pixels. Descarta centros sobre
    nodata da imagem (todas as bandas 0 ou não-finitas). `keep_fn(coords)`
    (opcional) filtra os candidatos, ex.: só os do split treino/validação."""
    rng = np.random.default_rng(seed)
    H, W = ref.shape
    out, got = [], 0
    for _ in range(max_iter):
        k = max(4 * (n - got), 1024)
        rr = rng.integers(margin, H - margin, size=k)
        cc = rng.integers(margin, W - margin, size=k)
        px = image[:, rr, cc]                                # (C, k)
        keep = ((ref[rr, cc] == 0) & np.isfinite(px).all(axis=0)
                & (np.abs(px).sum(axis=0) > 0))
        cand = np.stack([rr, cc], axis=1)
        if keep_fn is not None:
            keep &= keep_fn(cand)
        out.append(cand[keep])
        got += int(keep.sum())
        if got >= n:
            break
    return np.concatenate(out, axis=0)[:n]


def _split_blocks(fg_coords, H, W, block, frac, seed=42):
    """Divisão ESPACIAL treino/validação: o raster é dividido em blocos
    block×block e `frac` dos blocos vai para a validação — sorteado
    separadamente entre blocos COM rótulo (garante objetos na validação) e
    SEM rótulo (o fundo também precisa ser validado). Retorna is_val
    (n_blocos_y, n_blocos_x) bool."""
    nby, nbx = -(-H // block), -(-W // block)
    has = np.zeros((nby, nbx), dtype=bool)
    has[fg_coords[:, 0] // block, fg_coords[:, 1] // block] = True
    rng = np.random.default_rng(seed)
    is_val = np.zeros((nby, nbx), dtype=bool)
    for group in (has, ~has):
        ids = np.flatnonzero(group)
        n = int(round(frac * ids.size))
        if group is has and ids.size >= 2:
            n = min(max(n, 1), ids.size - 1)   # ≥1 bloco rotulado em cada split
        if n > 0:
            is_val.flat[rng.choice(ids, size=n, replace=False)] = True
    return is_val


def _window_in_split(coords, p, block, is_val, want_val):
    """True p/ os centros cujo patch p×p cai INTEIRO em blocos do split pedido.
    Patches que cruzam a fronteira treino/validação são descartados, então
    nenhum pixel é compartilhado entre os dois conjuntos (exige p <= block)."""
    r0 = coords[:, 0] - p // 2
    c0 = coords[:, 1] - p // 2
    ok = np.ones(coords.shape[0], dtype=bool)
    for rr in (r0, r0 + p - 1):
        for cc in (c0, c0 + p - 1):
            ok &= is_val[rr // block, cc // block] == want_val
    return ok


class OnTheFlyMangroveDataset(Dataset):
    """
    Dataset on-the-fly (estilo main_FINAL) com suporte a LateFusion.

    Carrega os rasters grandes uma vez:
      • image_path : imagem (C, H, W) — define o grid/resolução de SAÍDA
      • ref_path   : raster de rótulos de espécie (H, W), 0 = fundo
      • depth_path : raster de distância/altura (H, W) — alvo de regressão
        (ref/depth em outro grid são reamostrados p/ o da imagem, que é
         recortada à área do ref — ver _load_on_image_grid)
      • chm_path   : (LateFusion) raster CHM/LiDAR/multiespectral, lido na SUA
                     resolução nativa (recortado pela geo-referência)

    Os centros dos patches são os pixels rotulados (ref != 0), com uma
    margem de `patch_size` nas bordas (para o patch caber). Aplica
    oversampling por classe. Com `bg_as_class=True` (fundo 0 treinado como
    classe), soma-se `bg_center_ratio` × nº de centros rotulados sorteados no
    fundo. `samples` limita o nº de patches por época (None = usa todos os
    centros após oversampling).

    Com `val_fraction > 0`, o raster é dividido em blocos `val_block_size` e
    parte deles fica reservada para VALIDAÇÃO (ver _split_blocks): os centros
    de treino e de validação vêm de regiões disjuntas, sem patch cruzando a
    fronteira. `validation_view()` devolve o dataset de validação, que
    compartilha os rasters já carregados (centros fixos, sem augmentation).

    Saída por item (idêntica ao MangroveDataset):
      • sem LateFusion : (img_t [C,H,W], dist_t [1,H,W], label_t [H,W])
      • com LateFusion : ([img_t, lidar_t], dist_t, label_t)
    """

    def __init__(self, image_path, ref_path, depth_path,
                 patch_size=64, latefusion=False, chm_path=None,
                 lidar_channels=1, norm_values=None,
                 label_shift=0, bg_ignore_value=-100,
                 samples=None, augment=False, oversample=True,
                 rgb_norm_const=255.0, seed=42,
                 rgb_res=0.25, chm_res=1.0,
                 require_full_label=False,
                 num_classes=None, bg_as_class=False, bg_center_ratio=1.0,
                 rgb_normalize=False, rgb_norm_method='minmax',
                 rgb_min=None, rgb_max=None,
                 val_fraction=0.0, val_block_size=1024, val_samples=None,
                 lidar_norm='minmax', lidar_minmax_scope='band',
                 lidar_min=None, lidar_max=None):
        super().__init__()
        self.psize       = int(patch_size)
        self.latefusion  = bool(latefusion)
        self.rgb_res     = float(rgb_res)
        self.chm_res     = float(chm_res)
        self.label_shift = int(label_shift)
        self.bg_ignore_value = int(bg_ignore_value)
        # num_classes: rótulos fora de [0, num_classes) (nodata) → bg_ignore_value
        self.num_classes = int(num_classes) if num_classes is not None else None
        self.augment     = bool(augment)
        self.rgb_norm_const = float(rgb_norm_const)
        self.norm_values = (np.asarray(norm_values, dtype=np.float32)
                            if norm_values is not None else None)

        # ── Carrega rasters grandes (uma vez) ─────────────────
        # Grid de saída = grid nativo de --image_path; ref/depth em outro grid
        # (recorte/resolução/CRS) são reamostrados p/ ele.
        self.image, ref, depth, img_tf, img_crs = _load_on_image_grid(
            image_path, ref_path, depth_path)               # (C, H, W)
        # Normalização RGB (salva no checkpoint p/ o predict reproduzir):
        #   'none'     → usa os valores como estão (imagem já em 0–1)
        #   'minmax'   → (x - min)/(max - min) por banda, min/max do raster
        #                (ignorando NaN) ou de rgb_min/rgb_max
        #   'constant' → x / rgb_norm_const (ex.: 255 p/ uint8)
        self.rgb_norm = rgb_norm_method if rgb_normalize else 'none'
        if self.rgb_norm not in ('none', 'minmax', 'constant'):
            raise ValueError(f"rgb_norm_method inválido: {rgb_norm_method} "
                             f"(use 'minmax' ou 'constant').")
        self.rgb_min = self.rgb_max = None
        if self.rgb_norm == 'minmax':
            if rgb_min is not None and rgb_max is not None:
                self.rgb_min = np.asarray(rgb_min, dtype=np.float32)
                self.rgb_max = np.asarray(rgb_max, dtype=np.float32)
            else:
                self.rgb_min, self.rgb_max = _rgb_band_minmax(self.image)
        n_nan = int(np.isnan(self.image[0]).sum())
        if n_nan:
            print(f"  ⚠️  imagem tem {n_nan:,} pixels NaN por banda "
                  f"({100.0 * n_nan / self.image[0].size:.1f}%) → viram 0 após a "
                  f"normalização (senão a loss vira NaN).")
        self.ref   = ref.astype(np.int64)
        self.depth = depth.astype(np.float32)

        H, W = self.ref.shape
        if self.image.shape[1:] != (H, W):
            raise ValueError(
                f"RGB {self.image.shape[1:]} e ref {(H, W)} têm dimensões "
                f"diferentes. RGB, ref e depth precisam estar no mesmo grid.")
        if self.depth.shape != (H, W):
            raise ValueError(
                f"depth {self.depth.shape} e ref {(H, W)} têm dimensões "
                f"diferentes (mesmo grid esperado).")

        # ── LateFusion: prepara leitura do CHM por geo-referência ──
        self.chm = None
        self.lidar_norm = lidar_norm
        self.lidar_min = self.lidar_max = None
        if self.latefusion:
            if not chm_path:
                raise ValueError("latefusion=True exige chm_path (raster CHM/LiDAR).")
            import rasterio as rio
            from rasterio.windows import Window
            # localização: RGB(row,col) → CHM(row,col), CRS-aware, a partir do
            # grid da imagem já carregado (eventualmente recortado)
            with rio.open(chm_path) as s:
                chm_tf, chm_crs, chm_H, chm_W = s.transform, s.crs, s.height, s.width
            M, err = _rgb_to_chm_map(img_tf, img_crs, chm_tf, chm_crs, H, W)
            if img_crs != chm_crs:
                print(f"  [CHM] CRS diferente do RGB → reprojeção da localização "
                      f"(erro máx. do ajuste: {err:.4f} px do CHM)")

            # Patch CHM na resolução NATIVA do CHM, cobrindo a MESMA área do patch
            # RGB: lado = patch_size × (px do CHM por px do RGB), medido pela
            # geo-referência (M) — não pelos args --rgb/--chm_resolution, que
            # podem não conferir com os rasters. O train() repassa chm_patch ao
            # build_model; na fusão os mapas de atributos do braço CHM são
            # reamostrados p/ o grid do RGB (LateFusionGeneric.forward).
            pts = _poly2([0, H, 0], [0, 0, W]) @ M.T          # [row, col] no CHM
            self.chm_ratio = 0.5 * (np.hypot(*(pts[1] - pts[0])) / H
                                    + np.hypot(*(pts[2] - pts[0])) / W)
            self.chm_patch = max(1, int(round(self.psize * self.chm_ratio)))
            if abs(self.chm_res / self.rgb_res * self.chm_ratio - 1) > 0.05:
                print(f"  ⚠️  --chm_resolution/--rgb_resolution = {self.chm_res:g}/"
                      f"{self.rgb_res:g} não confere com os rasters (1 px do CHM = "
                      f"{1 / self.chm_ratio:.3f} px da imagem). Usando o medido.")
            print(f"  [CHM] resolução nativa: imagem {abs(img_tf.a):.4g}, CHM "
                  f"{abs(chm_tf.a):.4g} (1 px CHM = {1 / self.chm_ratio:.3f} px "
                  f"imagem) → patch {self.psize}px ↔ patch CHM {self.chm_patch}px")

            # carrega SÓ o recorte do CHM que cobre o RGB (+ margem de 1 patch)
            er, ec = np.meshgrid(np.linspace(0, H, 21), np.linspace(0, W, 21), indexing='ij')
            rows, cols = (_poly2(er.ravel(), ec.ravel()) @ M.T).T
            pad = self.chm_patch + 2
            r_lo = max(0, int(np.floor(rows.min())) - pad)
            r_hi = min(chm_H, int(np.ceil(rows.max())) + pad)
            c_lo = max(0, int(np.floor(cols.min())) - pad)
            c_hi = min(chm_W, int(np.ceil(cols.max())) + pad)
            if r_hi <= r_lo or c_hi <= c_lo:
                raise ValueError("O raster CHM/LiDAR não cobre a área da imagem RGB.")
            with rio.open(chm_path) as s:
                chm = s.read(window=Window(c_lo, r_lo, c_hi - c_lo, r_hi - r_lo)
                             ).astype(np.float32, copy=False)      # (Cc, h, w)
            M[0, 0] -= r_lo                    # termo constante → relativo
            M[1, 0] -= c_lo                    # ao recorte
            self.chm_M = M
            self.chm_h, self.chm_w = chm.shape[1], chm.shape[2]
            if chm.shape[0] != lidar_channels:
                print(f"  ⚠️  CHM tem {chm.shape[0]} banda(s) mas "
                      f"--lidar_channels={lidar_channels}. Usando {chm.shape[0]}.")

            if lidar_norm == 'minmax':
                # min-max por banda (ou global) sobre a área recortada, ignorando
                # NaN; NaN/fora da cobertura → 0. Guardado em float16 (valores
                # em [0, 1]) p/ caber em RAM com 128 bandas × N workers.
                if lidar_min is not None and lidar_max is not None:
                    lmin = np.asarray(lidar_min, dtype=np.float32)
                    lmax = np.asarray(lidar_max, dtype=np.float32)
                else:
                    lmin, lmax = _rgb_band_minmax(chm)
                    if lidar_minmax_scope == 'global':
                        lmin[:], lmax[:] = lmin.min(), lmax.max()
                for b in range(chm.shape[0]):
                    band = chm[b]
                    band -= lmin[b]
                    band /= max(float(lmax[b] - lmin[b]), 1e-12)
                    np.clip(band, 0.0, 1.0, out=band)
                    np.nan_to_num(band, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
                self.lidar_min, self.lidar_max = lmin, lmax
                self.chm = chm.astype(np.float16)
                del chm
                print(f"  [CHM] {self.chm.shape[0]} bandas | recorte {self.chm_h}x"
                      f"{self.chm_w} px | min-max {lidar_minmax_scope} "
                      f"(min∈[{lmin.min():.3f}, {lmin.max():.3f}], "
                      f"max∈[{lmax.min():.3f}, {lmax.max():.3f}]) | "
                      f"patch CHM={self.chm_patch}px | "
                      f"{self.chm.nbytes / 1e9:.2f} GB (float16)")
            else:
                self.chm = chm   # legado: tetos --norm_values aplicados por patch

        # ── Valores do ref fora de [0, num_classes) → serão ignorados ──
        if self.num_classes is not None:
            vmin, vmax = int(self.ref.min()), int(self.ref.max())
            if vmin < 0 or vmax >= self.num_classes:
                print(f"  ⚠️  ref contém valores fora de [0, {self.num_classes}) "
                      f"(min={vmin}, max={vmax}); tratados como nodata "
                      f"(ignorados na loss). Confira --num_classes.")

        # ── Coordenadas candidatas: pixels rotulados, com margem ──
        m = self.psize // 2
        valid = np.zeros_like(self.ref, dtype=bool)
        valid[m:H - m, m:W - m] = True
        labeled = (self.ref > 0) & valid                     # -1 = fora do ref
        if self.num_classes is not None:
            labeled &= (self.ref > 0) & (self.ref < self.num_classes)
        rr, cc = np.where(labeled)
        coords = np.stack([rr, cc], axis=1)                  # (N, 2)
        if coords.shape[0] == 0:
            raise ValueError("Nenhum pixel rotulado válido (após margem) "
                             "encontrado em ref_path.")

        # ── Filtro: exclui patches NÃO totalmente rotulados ───────
        # Se require_full_label=True, remove os centros cujo patch p×p contém
        # QUALQUER pixel de classe 0 (não-rotulado). Assim a classe 0 não entra
        # no treino de classificação, mas também não é "ignorada" via
        # ignore_index — o patch inteiro simplesmente não é usado.
        # Conta os zeros por patch com IMAGEM INTEGRAL (rápido e sem scipy).
        # Com o fundo como classe treinada o filtro não faz sentido (todo patch
        # já é "totalmente rotulado") e é desativado.
        if require_full_label and bg_as_class:
            print("  ⚠️  --require_full_label ignorado: com --ignore_index False "
                  "o fundo (0) é uma classe treinada, não pixel não-rotulado.")
        elif require_full_label:
            p = self.psize
            zero_mask = (self.ref == 0).astype(np.int64)      # (H, W)
            # integral image com borda zero → soma em janela em O(1) por centro
            ii = np.zeros((H + 1, W + 1), dtype=np.int64)
            ii[1:, 1:] = np.cumsum(np.cumsum(zero_mask, axis=0), axis=1)
            # para centro (r,c): patch é [r-m : r-m+p, c-m : c-m+p]
            r0 = coords[:, 0] - m
            c0 = coords[:, 1] - m
            r1 = r0 + p
            c1 = c0 + p
            # soma de zeros no patch via integral image
            zeros_in_patch = (ii[r1, c1] - ii[r0, c1]
                              - ii[r1, c0] + ii[r0, c0])
            keep = zeros_in_patch == 0
            n_before = coords.shape[0]
            coords = coords[keep]
            n_after = coords.shape[0]
            print(f"[on-the-fly] filtro require_full_label: "
                  f"{n_before - n_after} centros removidos "
                  f"(patch continha classe 0); restam {n_after}.")
            if n_after == 0:
                raise ValueError(
                    "Nenhum patch totalmente rotulado (sem classe 0) encontrado. "
                    "Desative --require_full_label ou reduza --patch_size.")

        # ── Divisão ESPACIAL treino/validação por blocos ──────────
        # Validar em patches vizinhos/sobrepostos aos de treino mede memorização,
        # não generalização. Aqui regiões inteiras (blocos) ficam de fora do
        # treino, e nenhum patch cruza a fronteira entre os dois conjuntos.
        p = self.psize
        self.val_coords = None
        trn_keep = val_keep = None
        if val_fraction and val_fraction > 0:
            B = int(val_block_size)
            if B < p:
                raise ValueError(f"--val_block_size ({B}) deve ser >= --patch_size ({p}).")
            is_val = _split_blocks(coords, H, W, B, val_fraction, seed=seed)
            trn_keep = lambda rc: _window_in_split(rc, p, B, is_val, False)
            val_keep = lambda rc: _window_in_split(rc, p, B, is_val, True)
            val_fg = coords[val_keep(coords)]
            n_all  = coords.shape[0]
            coords = coords[trn_keep(coords)]
            print(f"[on-the-fly] split espacial: {int(is_val.sum())}/{is_val.size} "
                  f"blocos de {B}px p/ validação | centros rotulados: "
                  f"treino={coords.shape[0]} val={val_fg.shape[0]} "
                  f"(descartados na fronteira={n_all - coords.shape[0] - val_fg.shape[0]})")
            if coords.shape[0] == 0 or val_fg.shape[0] == 0:
                raise ValueError("Split espacial deixou treino ou validação sem pixels "
                                 "rotulados. Ajuste --val_fraction / --val_block_size.")

        coords = self._make_centers(coords, oversample, bg_as_class,
                                    bg_center_ratio, m, seed, trn_keep, 'treino')
        if val_keep is not None:
            # validação: sem oversampling (distribuição real dos rótulos) e com
            # um conjunto FIXO de centros → métricas comparáveis entre épocas
            vc = self._make_centers(val_fg, False, bg_as_class, bg_center_ratio,
                                    m, seed + 1, val_keep, 'validação')
            if val_samples and vc.shape[0] > val_samples:
                vc = vc[np.random.default_rng(seed).choice(vc.shape[0], int(val_samples),
                                                          replace=False)]
            self.val_coords = vc

        # ── Normaliza a imagem inteira in-place (DEPOIS de sortear os centros,
        # que usam o NaN para detectar nodata). NaN → 0 em qualquer modo.
        for b in range(self.image.shape[0]):
            band = self.image[b]
            if self.rgb_norm == 'minmax':
                band -= self.rgb_min[b]
                band /= max(float(self.rgb_max[b] - self.rgb_min[b]), 1e-12)
                np.clip(band, 0.0, 1.0, out=band)
            elif self.rgb_norm == 'constant':
                band /= self.rgb_norm_const
            np.nan_to_num(band, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        if self.rgb_norm == 'minmax':
            print(f"[on-the-fly] RGB min-max por banda | min={self.rgb_min.tolist()} "
                  f"| max={self.rgb_max.tolist()}")
        elif self.rgb_norm == 'constant':
            print(f"[on-the-fly] RGB / {self.rgb_norm_const:g}")
        else:
            print("[on-the-fly] RGB sem normalização (valores originais; NaN → 0)")

        self.coords = coords
        # nº de patches por época
        if samples is not None and samples > 0 and samples < coords.shape[0]:
            self.samples = int(samples)
        else:
            self.samples = None

        print(f"[on-the-fly] {coords.shape[0]} centros após "
              f"{'oversampling' if oversample else 'sem oversampling'} | "
              f"patch={self.psize}px | "
              f"{'LateFusion RGB+CHM' if self.latefusion else 'single-source RGB'} | "
              f"samples/época={self.samples if self.samples else coords.shape[0]}")
        if self.val_coords is not None:
            print(f"[on-the-fly] validação: {self.val_coords.shape[0]} patches fixos")

    def _make_centers(self, fg, oversample, bg_as_class, bg_center_ratio,
                      margin, seed, keep_fn, name):
        """Centros rotulados (oversampling opcional) + centros de fundo
        sorteados dentro do mesmo split (keep_fn)."""
        coords = fg
        if oversample:
            labs = self.ref[coords[:, 0], coords[:, 1]]
            coords = _oversample_coords(coords, labs, seed=seed)
        # ── Centros no FUNDO (fundo treinado como classe) ─────────
        # Sem eles, todo patch é centrado num objeto rotulado e a rede nunca vê
        # áreas só de fundo. Somados DEPOIS do oversampling para que
        # bg_center_ratio controle a proporção fundo : rotulado.
        if bg_as_class and bg_center_ratio > 0:
            n_bg = int(round(bg_center_ratio * coords.shape[0]))
            bg_coords = _sample_bg_coords(self.ref, self.image, margin, n_bg,
                                          seed=seed, keep_fn=keep_fn)
            print(f"[on-the-fly] {name}: +{bg_coords.shape[0]} centros de fundo "
                  f"para {coords.shape[0]} centros rotulados "
                  f"(--bg_center_ratio {bg_center_ratio}).")
            coords = np.concatenate([coords, bg_coords.astype(coords.dtype)], axis=0)
            np.random.default_rng(seed).shuffle(coords)
        return coords

    def validation_view(self):
        """Dataset de VALIDAÇÃO que compartilha os rasters já carregados e
        normalizados (cópia rasa — nada é recarregado): centros fixos dos
        blocos de validação, sem augmentation. None se não houver split."""
        if self.val_coords is None:
            return None
        import copy as _copy
        v = _copy.copy(self)
        v.coords     = self.val_coords
        v.val_coords = None
        v.samples    = None
        v.augment    = False
        return v

    def __len__(self):
        return self.samples if self.samples else self.coords.shape[0]

    def _pick(self, idx):
        # Se samples < total, sorteia um centro a cada acesso (amostragem
        # com reposição por época, como no main_FINAL com --samples).
        # Usa o RNG do torch: o DataLoader semeia cada worker com uma semente
        # diferente. (Um np.random.Generator guardado no dataset é COPIADO
        # idêntico para cada worker → todos sorteavam os mesmos patches.)
        if self.samples:
            j = int(torch.randint(0, self.coords.shape[0], (1,)).item())
            return self.coords[j]
        return self.coords[idx]

    def _crop_chm(self, r, c):
        """Recorta o patch CHM correspondente à área geográfica do patch
        RGB centrado em (r, c). O canto do patch RGB vai para o grid do CHM
        (recortado) pelo mapeamento georreferenciado `chm_M` — robusto a
        origens, resoluções e CRS distintos. Faz clamp nas bordas e preenche
        com zeros (= nodata após a normalização) fora da cobertura."""
        p   = self.psize
        r0, c0 = r - p // 2, c - p // 2          # canto sup-esq no grid RGB
        rc_f, cc_f = self.chm_M @ _poly2(r0, c0)
        rc0, cc0 = int(round(rc_f)), int(round(cc_f))

        pc  = self.chm_patch
        Cc  = self.chm.shape[0]
        out = np.zeros((Cc, pc, pc), dtype=np.float32)
        r_lo, r_hi = max(0, rc0), min(self.chm_h, rc0 + pc)
        c_lo, c_hi = max(0, cc0), min(self.chm_w, cc0 + pc)
        if r_hi > r_lo and c_hi > c_lo:
            sub = self.chm[:, r_lo:r_hi, c_lo:c_hi]
            out[:, (r_lo - rc0):(r_lo - rc0) + sub.shape[1],
                   (c_lo - cc0):(c_lo - cc0) + sub.shape[2]] = sub
        return out

    def __getitem__(self, idx):
        r, c = self._pick(idx)
        p  = self.psize
        r0, c0 = r - p // 2, c - p // 2

        img   = self.image[:, r0:r0 + p, c0:c0 + p].copy()   # (C, p, p)
        label = self.ref[r0:r0 + p, c0:c0 + p].copy()        # (p, p)
        dist  = self.depth[r0:r0 + p, c0:c0 + p].copy()[None, ...]  # (1, p, p)

        # (RGB já normalizado + NaN→0 no __init__)

        # ── CHM/LiDAR (LateFusion) ────────────────────────────
        lidar = None
        if self.latefusion:
            lidar = self._crop_chm(r, c)                     # (Cc, pc, pc) float32
        if self.latefusion and self.lidar_norm != 'minmax':
            # (minmax: já normalizado + NaN→0 no __init__) — legado abaixo
            lidar = np.nan_to_num(lidar, nan=0.0)
            if self.norm_values is not None:
                nv = self.norm_values
                cc_ = lidar.shape[0]
                if nv.shape[0] != cc_:
                    raise ValueError(
                        f"--norm_values tem {nv.shape[0]} valores, mas o CHM "
                        f"tem {cc_} banda(s).")
                lidar = lidar / (nv.reshape(cc_, 1, 1) + 1e-8)
                lidar = np.clip(lidar, 0.0, 1.0)
            else:
                lmax = 30.0
                lidar = lidar / (lmax + 1e-8)

        # ── Normalização do mapa de distância → [0, 1] ────────
        dmax = dist.max()
        if dmax > 1.0:
            dist = dist / (dmax + 1e-8)

        if self.augment:
            img, dist, label, lidar = MangroveDataset._augment(
                self, img, dist, label, lidar)

        # ── Deslocamento de rótulos (legado; _resolve_ignore usa sempre 0) ─
        if self.label_shift != 0:
            label = label.astype(np.int64)
            bg = (label == 0)
            label = label + self.label_shift
            label[bg] = self.bg_ignore_value

        # ── Nodata/valores espúrios (fora de [0, num_classes)) → ignore ─
        if self.num_classes is not None:
            label = label.astype(np.int64)
            label[(label < 0) | (label >= self.num_classes)] = self.bg_ignore_value

        img_t   = torch.from_numpy(img).float()
        dist_t  = torch.from_numpy(dist).float()
        label_t = torch.from_numpy(label).long()

        if self.latefusion:
            lidar_t = torch.from_numpy(lidar).float()
            return [img_t, lidar_t], dist_t, label_t
        return img_t, dist_t, label_t

    def estimate_class_counts(self, num_classes, n_patches=2000, seed=0):
        """Estima a contagem de pixels por classe nos patches que o treino
        realmente vê (centros após oversampling + fundo), para os pesos da CE.
        Contar o raster inteiro não serve: o fundo domina (>99%) e o peso dele
        iria a ~0, fazendo a rede prever só a classe rara."""
        rng = np.random.default_rng(seed)
        n = min(int(n_patches), self.coords.shape[0])
        idx = rng.choice(self.coords.shape[0], size=n, replace=False)
        p = self.psize
        counts = np.zeros(num_classes, dtype=np.float64)
        for r, c in self.coords[idx]:
            lab = self.ref[r - p // 2:r - p // 2 + p, c - p // 2:c - p // 2 + p]
            lab = lab[(lab >= 0) & (lab < num_classes)]
            counts += np.bincount(lab.ravel(), minlength=num_classes)[:num_classes]
        return counts


# ─────────────────────────────────────────────────────────────
# 4.  Model architectures
# ─────────────────────────────────────────────────────────────

# ── 4a. Sub-modules used by BPATUNet ──────────────────────────
class BandAdaptationLayer(nn.Module):
    def __init__(self, in_channels: int, target_channels: int = 4):
        super().__init__()
        self.adapt = (
            nn.Conv2d(in_channels, target_channels, 1, bias=False)
            if in_channels != target_channels else None
        )
        if self.adapt is not None:
            nn.init.kaiming_normal_(self.adapt.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.adapt(x) if self.adapt is not None else x


class BoundaryEnhancementModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1),
        )
        self.enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        bmap = torch.sigmoid(self.edge_conv(x))
        return x + self.enhance(x) * bmap, bmap


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid = max(1, in_channels // reduction)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, mid, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(mid, in_channels, 1, bias=False),
        )

    def forward(self, x):
        avg = self.fc(F.adaptive_avg_pool2d(x, 1))
        mx  = self.fc(F.adaptive_max_pool2d(x, 1))
        return x * torch.sigmoid(avg + mx)


class DoubleConv(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
            nn.Conv2d(oc, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class Down(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(nn.MaxPool2d(2), DoubleConv(ic, oc))
    def forward(self, x): return self.net(x)


class Up(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.up   = nn.ConvTranspose2d(ic, ic // 2, 2, stride=2)
        self.conv = DoubleConv(ic, oc)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy = x2.size(2) - x1.size(2)
        dx = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


# ── 4d.  MultitaskHead — dois modos de interação ──────────────
class MultitaskHead(nn.Module):
    """
    Dois modos controlados por `dist_guided`:

    dist_guided=False  (padrão / baseline)
    ──────────────────
      1. feat      = backbone(x)
      2. dist_pred = sigmoid(reg_head(feat))       → [0,1], compatível com target normalizado
      3. seg_logits= clf_head(cat[feat, dist_pred])   dist como canal extra

    dist_guided=True  (Distance-guided Segmentation)
    ─────────────────
      Motivação: o mapa de distância euclidiana é unimodal por copa —
      pico no centro (≈1), zero na borda/fundo. Pixeis de alta distância são
      os mais representativos de cada espécie; pixeis de borda são ruidosos.
      dist_pred já está em [0,1] e é usado diretamente como mapa de atenção
      espacial que PONDERA as features antes da classificação.

      1. feat        = backbone(x)
      2. dist_pred   = sigmoid(reg_head(feat))     → [0,1], pico no centro da copa
      3. attn        = dist_pred                   já é o mapa de atenção
      4. feat_guided = feat * attn                 suprime borda/fundo
      5. seg_logits  = clf_head(cat[feat_guided, attn])
    """

    def __init__(self, feature_extractor: nn.Module,
                 dec_out_channels: int, num_classes: int,
                 dist_guided: bool = False, task: str = 'multitask',
                 reg_bias_init: float = 0.0):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.dist_guided       = dist_guided
        self.task              = task

        # ── Cabeça de regressão (distância) ───────────────────────
        if task in ('multitask', 'regression'):
            self.reg_head = nn.Sequential(
                nn.Conv2d(dec_out_channels, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1),
            )
            # Bias inicial da última conv da reg_head.
            # ATENÇÃO: um bias muito negativo (ex.: -4) faz sigmoid≈0 e a
            # DERIVADA da sigmoid p·(1-p)≈0, saturando o gradiente — a rede não
            # consegue "levantar" as copas e muitas árvores nunca são marcadas.
            # Default 0.0 (sigmoid=0.5, derivada=0.25) dá gradiente saudável.
            nn.init.constant_(self.reg_head[-1].bias, reg_bias_init)

        # ── Cabeça de classificação ────────────────────────────────
        # multitask      → dec_out_channels + 1 (canal de distância / atenção)
        # classification → dec_out_channels (sem ramo de regressão)
        if task in ('multitask', 'classification'):
            clf_in = dec_out_channels + 1 if task == 'multitask' else dec_out_channels
            self.clf_head = nn.Sequential(
                nn.Conv2d(clf_in, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes, 1),
            )

    def forward(self, x):
        # ── 1. Features compartilhadas ─────────────────────────────
        feat = self.feature_extractor(x)           # (B, C, H, W)

        # ── Regressão pura ─────────────────────────────────────────
        if self.task == 'regression':
            return torch.sigmoid(self.reg_head(feat))

        # ── Classificação pura ─────────────────────────────────────
        if self.task == 'classification':
            return self.clf_head(feat)

        # ── 2. Multitask: regressão de distância ───────────────────
        dist_pred = torch.sigmoid(self.reg_head(feat))   # (B, 1, H, W) ∈ [0,1]

        if self.dist_guided:
            attn = dist_pred                             # (B, 1, H, W) ∈ [0,1]
            feat_guided = feat * attn                    # (B, C, H, W)
            clf_input = torch.cat([feat_guided, attn], dim=1)   # (B, C+1, H, W)
        else:
            clf_input = torch.cat([feat, dist_pred], dim=1)     # (B, C+1, H, W)

        seg_logits = self.clf_head(clf_input)       # (B, num_classes, H, W)

        return dist_pred, seg_logits


# ── 4d.  Build backbone (um único braço, qualquer arquitetura) ─
def build_backbone(args, in_channels: int, out_ch: int, depth: int = 4) -> nn.Module:
    """
    Constrói UM backbone encoder-decoder que recebe `in_channels` canais e
    devolve um mapa com `out_ch` canais na resolução do input.

    `depth` (só usado por BPATUNet): nº de downsamplings. Padrão 4 (original).
    Útil para reduzir a profundidade do braço LiDAR quando o patch CHM é pequeno.

    Usado tanto pelo caminho single-source quanto pela LateFusion genérica
    (onde é instanciado uma vez por fonte: RGB e CHM/LiDAR).
    """
    SMP_COMMON = dict(
        encoder_name    = "resnet50",
        encoder_weights = "imagenet" if args.pretrained else None,
        in_channels     = in_channels,
        activation      = None,
        encoder_depth   = 4,
    )

    if args.arch == "unet":
        return smp.Unet(
            **SMP_COMMON, classes=out_ch,
            decoder_channels=(512, 256, 128, 64),
            decoder_use_batchnorm=True,
            decoder_attention_type="scse",
        )
    elif args.arch == "unetplusplus":
        return smp.UnetPlusPlus(
            **SMP_COMMON, classes=out_ch,
            decoder_channels=(512, 256, 128, 64),
            decoder_use_batchnorm=True,
            decoder_attention_type="scse",
        )
    elif args.arch == "manet":
        return smp.MAnet(
            **SMP_COMMON, classes=out_ch,
            decoder_channels=(512, 256, 128, 64),
            decoder_use_batchnorm=True,
            decoder_pab_channels=64,
        )
    elif args.arch == "linknet":
        return smp.Linknet(
            **SMP_COMMON, classes=out_ch,
            decoder_channels=(256, 128, 64, 32),
        )
    elif args.arch == "BPATUNet":
        return BPATUNet(
            in_channels=in_channels,
            n_classes=out_ch,
            base_channels=64,
            depth=depth,
        )
    elif args.arch == "resunet":
        # ResU-Net "single-arm": encoder+decoder com skips + conv final p/ out_ch
        return ResUnetBackbone(in_channels, [64, 128, 256, 512], out_ch)
    else:
        sys.exit(f"Unknown architecture: {args.arch}")


# ── 4d.  Build model ──────────────────────────────────────────
def build_model(args) -> nn.Module:
    """
    Constrói o modelo segundo `args.task` ∈ {multitask, regression, classification}.

    Em todos os casos o backbone produz um mapa de `feature_channels` (64) canais;
    as cabeças (reg_head / clf_head) presentes dependem da tarefa. Funciona com
    QUALQUER arquitetura (--arch) e tanto no caminho de fusão (--latefusion) quanto
    no single-source.
    """
    feature_channels = 64
    task = args.task   # 'multitask' | 'regression' | 'classification'

    # ── LateFusion (RGB + CHM/LiDAR), qualquer --arch ─────────
    if getattr(args, 'latefusion', False):
        scale = args.chm_resolution / args.rgb_resolution
        # patch CHM medido nos rasters pelo dataset on-the-fly (args.chm_patch);
        # sem ele (patches pré-extraídos / checkpoints antigos), deriva dos args
        chm_patch = (getattr(args, 'chm_patch', None)
                     or args.patch_size // max(1, int(round(scale))))

        # Profundidade de downsampling segura para CADA braço.
        #   Cada nível divide a resolução por 2. Exigimos bottleneck ≥ 4 px
        #   (mais robusto que 2 px: preserva contexto espacial mínimo e evita
        #   BatchNorm/atenção sobre mapas 2×2 instáveis).
        #   max_depth tal que side / 2**depth >= 4.
        def _safe_depth(side, cap=4, min_bottleneck=4):
            d = 0
            s = side
            while s % 2 == 0 and s // 2 >= min_bottleneck and d < cap:
                s //= 2
                d += 1
            return max(1, d)

        rgb_depth   = _safe_depth(args.patch_size, cap=4)   # ex.: 64 → 4 (btl 4px)
        lidar_depth = _safe_depth(chm_patch,       cap=4)   # ex.: 16 → 2 (btl 4px)

        if args.arch == 'BPATUNet':
            # A BPAT-UNet original tem 4 downsamplings (÷16). Para o braço LiDAR
            # com patch pequeno, reduzimos a profundidade para não degenerar o
            # bottleneck a 1×1 (que apagaria a informação espacial do CHM).
            if lidar_depth < 4:
                print(f"  ℹ️  Braço LiDAR (BPATUNet): patch CHM = {chm_patch}px → "
                      f"usando depth={lidar_depth} (em vez de 4) p/ evitar "
                      f"colapso espacial. Braço RGB usa depth={rgb_depth}.")
            backbone_rgb   = build_backbone(args, args.channels,       feature_channels,
                                            depth=rgb_depth)
            backbone_lidar = build_backbone(args, args.lidar_channels, feature_channels,
                                            depth=lidar_depth)
        else:
            # Demais arquiteturas (SMP/resunet): aviso de compatibilidade.
            req = 8 if args.arch == 'resunet' else 16
            if chm_patch < req or chm_patch % req != 0:
                print(f"  ⚠️  AVISO: patch CHM = {chm_patch}px (patch_size={args.patch_size}, "
                      f"scale={scale:g}). O backbone '{args.arch}' exige múltiplo de {req} "
                      f"(≥{req}). Ajuste --patch_size ou use --arch resunet/BPATUNet "
                      f"(toleram patches pequenos).")
            backbone_rgb   = build_backbone(args, args.channels,        feature_channels)
            backbone_lidar = build_backbone(args, args.lidar_channels,  feature_channels)

        model = LateFusionGeneric(
            backbone_rgb   = backbone_rgb,
            backbone_lidar = backbone_lidar,
            feat_channels  = feature_channels,
            num_classes    = args.num_classes,
            dist_guided    = args.dist_guided,
            task           = task,
            fusion         = getattr(args, 'fusion', 'concat'),
            reg_bias_init  = getattr(args, 'reg_bias_init', 0.0),
        )
        extra = (f"| {'dist-guided' if args.dist_guided else 'concat'} "
                 f"| clf ({args.num_classes} cls)") if task != 'regression' else ""
        print(f"[Model] LateFusion ({getattr(args, 'fusion', 'concat')}) {task} "
              f"| arch={args.arch} | RGB + CHM(LiDAR) {extra}")
        return model

    # ── Single-source ─────────────────────────────────────────
    backbone = build_backbone(args, args.channels, feature_channels)
    model = MultitaskHead(
        feature_extractor = backbone,
        dec_out_channels  = feature_channels,
        num_classes       = args.num_classes,
        dist_guided       = args.dist_guided,
        task              = task,
        reg_bias_init     = getattr(args, 'reg_bias_init', 0.0),
    )
    extra = (f"| {'dist-guided' if args.dist_guided else 'concat'} "
             f"| clf ({args.num_classes} cls)") if task != 'regression' else ""
    print(f"[Model] Single-source {task} | arch={args.arch} {extra}")
    return model


# ─────────────────────────────────────────────────────────────
# 4e.  BPATUNet (standalone, used above)
# ─────────────────────────────────────────────────────────────
class BPATUNet(nn.Module):
    """
    BPAT-UNet com profundidade de downsampling configurável (`depth`).

    depth = nº de níveis de downsampling (cada um reduz a resolução por 2).
    O fundo da rede fica em entrada / 2**depth, então o lado do patch deve ser
    múltiplo de 2**depth e ≥ 2**depth (senão o bottleneck degenera a <2px).

    Use depth=4 (padrão, comportamento original) para entradas ≥16 e múltiplas
    de 16 (ex.: RGB 64×64). Para o braço LiDAR com patches pequenos (ex.: CHM
    16×16 a 1 m com patch RGB de 64), use depth=2 → 16→8→4, sem degenerar.
    """
    def __init__(self, in_channels=4, n_classes=1, base_channels=64, depth=4):
        super().__init__()
        assert 1 <= depth <= 4, f"depth deve estar em [1,4], recebido {depth}"
        self.depth = depth
        bc = base_channels

        self.band_adapt = BandAdaptationLayer(in_channels, 4)
        self.inc        = DoubleConv(4, bc)

        # Encoder: cria só os `depth` níveis necessários.
        self.downs = nn.ModuleList()
        self.boundary = nn.ModuleList()
        self.chatt = nn.ModuleList()
        for i in range(depth):
            ic = bc * (2 ** i)
            oc = bc * (2 ** (i + 1))
            self.downs.append(Down(ic, oc))
            self.boundary.append(BoundaryEnhancementModule(oc))
            # ChannelAttention nos níveis intermediários (como no original:
            # aplicado em x3 e x4, i.e. níveis 2 e 3 quando depth=4).
            self.chatt.append(ChannelAttention(oc) if 2 <= (i + 1) <= depth - 1
                              else None)

        # Decoder: `depth` níveis de upsample espelhando o encoder.
        self.ups = nn.ModuleList()
        for i in range(depth, 0, -1):
            ic = bc * (2 ** i)
            oc = bc * (2 ** (i - 1))
            self.ups.append(Up(ic, oc))

        self.outc = nn.Conv2d(bc, n_classes, 1)

    def forward(self, x):
        x = self.band_adapt(x)
        feats = [self.inc(x)]                     # x1 (sem downsample)
        cur = feats[0]
        for i in range(self.depth):
            cur, _ = self.boundary[i](self.downs[i](cur))
            if self.chatt[i] is not None:
                cur = self.chatt[i](cur)
            feats.append(cur)
        # feats = [x1, x2, ..., x_{depth+1}]; faz upsample com skips reversos
        x = feats[-1]
        for j, up in enumerate(self.ups):
            skip = feats[-(j + 2)]
            x = up(x, skip)
        return self.outc(x)


# ─────────────────────────────────────────────────────────────
# 4f.  LateFusion multitask
# ─────────────────────────────────────────────────────────────
# Estratégia LateFusion adaptada para tarefa multitask:
#   - braço 0  → processa a imagem RGB  (IMAGE/, resolução 0.25 m)
#   - braço 1  → processa o CHM LiDAR   (LIDAR/, resolução nativa 1 m)
# Cada braço tem seu próprio encoder + decoder.
# As features decodificadas são fundidas tardiamente (late fusion) por
# concatenação e enviadas a DUAS cabeças:
#   - reg_head  → mapa de distância (regressão)
#   - clf_head  → imagem classificada (segmentação)
#
# IMPORTANTE (resolução): como o CHM entra em resolução nativa (ex. 32x32 para
# um patch RGB de 128x128), a saída do braço LiDAR é menor que a do braço RGB.
# Antes de concatenar, o braço LiDAR é reamostrado (upsample) para a resolução
# espacial do braço RGB. Assim o LiDAR é PROCESSADO em sua resolução nativa e
# só é alinhado no momento da fusão.

class ResidualBlock(nn.Module):
    """
    Bloco residual full pre-activation usado pelos encoders/decoders ResU-Net.
    Compatível com as assinaturas usadas em resunet.py:
        ResidualBlock(in_channels, out_channels, stride=1)
    stride=2 → reduz a resolução pela metade (downsampling do encoder).
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                      padding=1, padding_mode='reflect'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1,
                      padding=1, padding_mode='reflect'),
        )
        self.identity = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                  stride=stride)

    def forward(self, x):
        return self.residual(x) + self.identity(x)


class ResUnetEncoder(nn.Module):
    """Encoder ResU-Net com skip-connections (igual ao resunet.py)."""
    def __init__(self, input_depth, depths):
        super().__init__()
        self.first_res_block = nn.Sequential(
            nn.Conv2d(input_depth, depths[0], kernel_size=3, padding=1,
                      padding_mode='reflect'),
            nn.BatchNorm2d(depths[0]),
            nn.ReLU(),
            nn.Conv2d(depths[0], depths[0], kernel_size=3, padding=1,
                      padding_mode='reflect'),
        )
        self.first_res_cov = nn.Conv2d(input_depth, depths[0], kernel_size=1)
        self.enc_block_0 = ResidualBlock(depths[0], depths[1], stride=2)
        self.enc_block_1 = ResidualBlock(depths[1], depths[2], stride=2)
        self.enc_block_2 = ResidualBlock(depths[2], depths[3], stride=2)

    def forward(self, x):
        x_idt = self.first_res_cov(x)
        x = self.first_res_block(x)
        x_0 = x + x_idt
        x_1 = self.enc_block_0(x_0)
        x_2 = self.enc_block_1(x_1)
        x_3 = self.enc_block_2(x_2)
        return x_0, x_1, x_2, x_3


class ResUnetDecoder(nn.Module):
    """Decoder ResU-Net com skip-connections (igual ao resunet.py)."""
    def __init__(self, depths):
        super().__init__()
        self.dec_block_2 = ResidualBlock(depths[2] + depths[3], depths[2])
        self.dec_block_1 = ResidualBlock(depths[1] + depths[2], depths[1])
        self.dec_block_0 = ResidualBlock(depths[0] + depths[1], depths[0])
        self.upsample_2 = nn.Upsample(scale_factor=2)
        self.upsample_1 = nn.Upsample(scale_factor=2)
        self.upsample_0 = nn.Upsample(scale_factor=2)

    def forward(self, x):
        x_0, x_1, x_2, x_3 = x
        x_2u = self.upsample_2(x_3)
        x_2c = torch.cat((x_2u, x_2), dim=1)
        x_2 = self.dec_block_2(x_2c)
        x_1u = self.upsample_1(x_2)
        x_1c = torch.cat((x_1u, x_1), dim=1)
        x_1 = self.dec_block_1(x_1c)
        x_0u = self.upsample_0(x_1)
        x_0c = torch.cat((x_0u, x_0), dim=1)
        x_0 = self.dec_block_0(x_0c)
        return x_0


class ResUnetBackbone(nn.Module):
    """
    ResU-Net de braço único: encoder + decoder com skip-connections, seguido
    de uma conv 1x1 que projeta para `out_ch` canais na resolução do input.
    Disponível como --arch resunet, e reutilizável pela LateFusion genérica.
    """
    def __init__(self, input_depth, depths, out_ch):
        super().__init__()
        self.encoder = ResUnetEncoder(input_depth, depths)
        self.decoder = ResUnetDecoder(depths)
        self.out_conv = nn.Conv2d(depths[0], out_ch, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.out_conv(x)


# ─────────────────────────────────────────────────────────────
#  Módulos de fusão plugáveis (RGB ⊕ LiDAR)
# ─────────────────────────────────────────────────────────────
# Todos recebem dois mapas de features já alinhados espacialmente
#   f_rgb, f_lidar : (B, feat, H, W)
# e devolvem um único mapa fundido de 2*feat canais, mantendo a
# interface esperada pelas cabeças (reg_head / clf_head).
#
# Estratégias disponíveis (--fusion):
#   'concat'  → concatenação simples (late fusion clássica). Barata, sem
#               parâmetros de fusão; deixa as cabeças aprenderem a combinar.
#   'gated'   → fusão por gating cross-modal (channel attention condicionada
#               na outra modalidade). Cada modalidade é reponderada por um
#               gate derivado da concatenação das duas, controlando quanto
#               cada fonte contribui por canal antes de concatenar.
#   'xattn'   → atenção espacial cruzada bidirecional. RGB atende ao LiDAR e
#               vice-versa (consultas de uma modalidade, chaves/valores da
#               outra), capturando correspondências espaciais entre as fontes.

class ConcatFusion(nn.Module):
    """Late fusion clássica: concatena as features das duas modalidades."""
    out_mult = 2

    def __init__(self, feat_channels: int):
        super().__init__()

    def forward(self, f_rgb, f_lidar):
        return torch.cat((f_rgb, f_lidar), dim=1)      # (B, 2*feat, H, W)


class GatedFusion(nn.Module):
    """
    Gated cross-modal fusion.

    A partir da concatenação [f_rgb, f_lidar] produz dois gates por canal
    (g_rgb, g_lidar) ∈ [0,1] via squeeze-and-excitation (pooling global →
    MLP → sigmoid). Cada modalidade é reponderada pelo seu gate antes da
    concatenação, deixando a rede aprender quanto confiar em cada fonte,
    por canal e por amostra. Saída: 2*feat canais (compatível com 'concat').
    """
    out_mult = 2

    def __init__(self, feat_channels: int, reduction: int = 8):
        super().__init__()
        self.feat = feat_channels
        hidden = max(1, (2 * feat_channels) // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(2 * feat_channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2 * feat_channels, 1),
        )
        # mistura local após o gating (refinamento), mantém 2*feat canais
        self.mix = nn.Sequential(
            nn.Conv2d(2 * feat_channels, 2 * feat_channels, 3, padding=1),
            nn.BatchNorm2d(2 * feat_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, f_rgb, f_lidar):
        cat = torch.cat((f_rgb, f_lidar), dim=1)          # (B, 2*feat, H, W)
        gate = torch.sigmoid(self.mlp(self.pool(cat)))    # (B, 2*feat, 1, 1)
        g_rgb, g_lidar = gate[:, :self.feat], gate[:, self.feat:]
        f_rgb   = f_rgb   * g_rgb
        f_lidar = f_lidar * g_lidar
        fused = torch.cat((f_rgb, f_lidar), dim=1)
        return self.mix(fused)                            # (B, 2*feat, H, W)


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention espacial bidirecional (estilo transformer leve, 1 cabeça).

    Cada modalidade gera consultas (Q) que atendem às chaves/valores (K, V)
    da OUTRA modalidade, capturando correspondências espaciais entre RGB e
    LiDAR. Os contextos atendidos são somados (residual) às features originais
    e então concatenados. Saída: 2*feat canais (compatível com 'concat').

    Para manter custo de memória sob controle em mapas grandes, a atenção é
    calculada numa resolução reduzida (downsample por `attn_stride`) e o
    contexto resultante é re-projetado à resolução original.
    """
    out_mult = 2

    def __init__(self, feat_channels: int, attn_dim: int = None,
                 attn_stride: int = 2):
        super().__init__()
        self.feat = feat_channels
        self.attn_dim = attn_dim or max(16, feat_channels // 2)
        self.stride = attn_stride

        d = self.attn_dim
        # projeções Q/K/V para cada direção (compartilha estilo, pesos próprios)
        self.q_rgb = nn.Conv2d(feat_channels, d, 1)
        self.k_lid = nn.Conv2d(feat_channels, d, 1)
        self.v_lid = nn.Conv2d(feat_channels, d, 1)
        self.q_lid = nn.Conv2d(feat_channels, d, 1)
        self.k_rgb = nn.Conv2d(feat_channels, d, 1)
        self.v_rgb = nn.Conv2d(feat_channels, d, 1)
        # re-projeta o contexto atendido de volta a feat canais
        self.proj_rgb = nn.Conv2d(d, feat_channels, 1)
        self.proj_lid = nn.Conv2d(d, feat_channels, 1)
        self.scale = d ** -0.5

    def _attend(self, q, k, v, out_hw):
        # q,k,v: (B, d, h, w) na resolução reduzida
        B, d, h, w = q.shape
        Q = q.flatten(2).transpose(1, 2)              # (B, hw, d)
        K = k.flatten(2)                              # (B, d, hw)
        V = v.flatten(2).transpose(1, 2)              # (B, hw, d)
        attn = torch.softmax((Q @ K) * self.scale, dim=-1)   # (B, hw, hw)
        ctx = (attn @ V).transpose(1, 2).reshape(B, d, h, w) # (B, d, h, w)
        if (h, w) != out_hw:
            ctx = F.interpolate(ctx, size=out_hw, mode='bilinear',
                                align_corners=False)
        return ctx

    def forward(self, f_rgb, f_lidar):
        H, W = f_rgb.shape[2:]
        s = self.stride
        # downsample para a atenção (memória O((HW/s²)²))
        r = F.avg_pool2d(f_rgb,   s) if s > 1 else f_rgb
        l = F.avg_pool2d(f_lidar, s) if s > 1 else f_lidar

        # RGB consulta LiDAR
        ctx_rgb = self._attend(self.q_rgb(r), self.k_lid(l), self.v_lid(l), (H, W))
        # LiDAR consulta RGB
        ctx_lid = self._attend(self.q_lid(l), self.k_rgb(r), self.v_rgb(r), (H, W))

        f_rgb   = f_rgb   + self.proj_rgb(ctx_rgb)    # residual
        f_lidar = f_lidar + self.proj_lid(ctx_lid)
        return torch.cat((f_rgb, f_lidar), dim=1)     # (B, 2*feat, H, W)


def build_fusion(name: str, feat_channels: int) -> nn.Module:
    """Fábrica de módulos de fusão. `name` ∈ {concat, gated, xattn}."""
    name = (name or 'concat').lower()
    if name == 'concat':
        return ConcatFusion(feat_channels)
    if name == 'gated':
        return GatedFusion(feat_channels)
    if name in ('xattn', 'crossattn', 'cross_attention'):
        return CrossAttentionFusion(feat_channels)
    raise ValueError(f"--fusion desconhecido: '{name}'. "
                     f"Opções: concat | gated | xattn.")


class LateFusionGeneric(nn.Module):
    """
    LateFusion genérica: funciona com QUALQUER arquitetura de braço único
    (unet, unetplusplus, manet, linknet, BPATUNet, resunet).

    Recebe dois backbones já construídos — cada um devolvendo um mapa de
    `feat_channels` canais na resolução do seu próprio input:
        backbone_rgb   : processa a imagem RGB  (resolução 0.25 m)
        backbone_lidar : processa o CHM/LiDAR   (resolução nativa 1 m)

    As features são fundidas tardiamente (concatenação) após alinhar
    espacialmente o braço LiDAR à resolução do braço RGB.

    O parâmetro `task` controla as cabeças e a saída:
        'multitask'      → reg_head + clf_head  → retorna (dist_pred, seg_logits)
        'regression'     → só reg_head          → retorna dist_pred
        'classification' → só clf_head          → retorna seg_logits
    (em 'classification' a clf_head recebe apenas as features fundidas, sem o
     canal de distância, já que não há ramo de regressão.)

    forward(x) recebe x = [rgb, lidar]:
        rgb   : (B, C_rgb,  H,   W)
        lidar : (B, C_lid,  H/s, W/s)
    """
    def __init__(self, backbone_rgb: nn.Module, backbone_lidar: nn.Module,
                 feat_channels: int, num_classes: int, dist_guided: bool = False,
                 task: str = 'multitask', fusion: str = 'concat',
                 reg_bias_init: float = 0.0):
        super().__init__()
        self.backbone_rgb   = backbone_rgb
        self.backbone_lidar = backbone_lidar
        self.dist_guided    = dist_guided
        self.task           = task

        # ── Módulo de fusão plugável (concat | gated | xattn) ─────
        # Todos devolvem 2*feat canais, mantendo as cabeças inalteradas.
        self.fusion = build_fusion(fusion, feat_channels)
        self.fusion_name = (fusion or 'concat').lower()
        fused_ch = 2 * feat_channels

        # ── Cabeça de regressão (distância) ───────────────────────
        if task in ('multitask', 'regression'):
            self.reg_head = nn.Sequential(
                nn.Conv2d(fused_ch, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 1, 1),
            )
            # Bias inicial da reg_head — ver nota na MultitaskHead. Default 0.0
            # evita a saturação de gradiente que faz copas não serem marcadas.
            nn.init.constant_(self.reg_head[-1].bias, reg_bias_init)

        # ── Cabeça de classificação ───────────────────────────────
        # multitask      → fused + 1 (canal de distância / atenção)
        # classification → fused (sem ramo de regressão → sem canal extra)
        if task in ('multitask', 'classification'):
            clf_in = fused_ch + 1 if task == 'multitask' else fused_ch
            self.clf_head = nn.Sequential(
                nn.Conv2d(clf_in, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, num_classes, 1),
            )

    def forward(self, x):
        rgb, lidar = x[0], x[1]

        # ── Cada braço processa sua fonte na resolução nativa ─────
        f_rgb   = self.backbone_rgb(rgb)        # (B, feat, H, W)
        f_lidar = self.backbone_lidar(lidar)    # (B, feat, H/s, W/s)

        # ── Alinhamento espacial na fusão (upsample LiDAR → RGB) ───
        if f_lidar.shape[2:] != f_rgb.shape[2:]:
            f_lidar = F.interpolate(f_lidar, size=f_rgb.shape[2:],
                                    mode='bilinear', align_corners=False)

        # ── Fusão (estratégia escolhida em --fusion) ──────────────
        feat = self.fusion(f_rgb, f_lidar)           # (B, 2*feat, H, W)

        # ── Regressão pura ────────────────────────────────────────
        if self.task == 'regression':
            return torch.sigmoid(self.reg_head(feat))

        # ── Classificação pura ────────────────────────────────────
        if self.task == 'classification':
            return self.clf_head(feat)

        # ── Multitask (regressão + segmentação) ───────────────────
        dist_pred = torch.sigmoid(self.reg_head(feat))   # (B, 1, H, W) ∈ [0,1]
        if self.dist_guided:
            attn = dist_pred
            clf_input = torch.cat([feat * attn, attn], dim=1)
        else:
            clf_input = torch.cat([feat, dist_pred], dim=1)
        seg_logits = self.clf_head(clf_input)
        return dist_pred, seg_logits


# ─────────────────────────────────────────────────────────────
# 5.  Loss functions
# ─────────────────────────────────────────────────────────────
class MaskedMSELoss(nn.Module):
    """Regressão mascarada — calcula a perda só nos pixels válidos (mask=True).

    `loss_type` ∈ {'l1', 'mse', 'smooth_l1'}:
      • 'l1'        → erro absoluto. Gradiente CONSTANTE mesmo p/ alvos pequenos,
                      então copas de sinal fraco (bordas, árvores menores) são
                      aprendidas. RECOMENDADO p/ mapas de distância com pico no
                      centro e decaimento p/ 0 (a maioria dos pixels tem valor
                      baixo). É o que o main_v2 usa e gera mapas melhores.
      • 'mse'       → erro quadrático. Penaliza pouco erros pequenos → a rede
                      tende a IGNORAR copas de distância baixa ("não pega todas
                      as árvores"). Mantido p/ compatibilidade/experimentação.
      • 'smooth_l1' → híbrido (Huber): L2 perto de 0, L1 longe. Meio-termo.
    """
    def __init__(self, ignore_index: int = 0, loss_type: str = 'l1'):
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_type = (loss_type or 'l1').lower()
        if self.loss_type not in ('l1', 'mse', 'smooth_l1'):
            raise ValueError(f"loss_type inválido: {loss_type} "
                             f"(use 'l1', 'mse' ou 'smooth_l1').")

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        pred   : (B, 1, H, W) float
        target : (B, 1, H, W) or (B, H, W) float
        mask   : (B, H, W) bool – True where to INCLUDE the pixel
        """
        if target.ndim == 3:
            target = target.unsqueeze(1)
        if pred.shape != target.shape:
            target = F.interpolate(target.float(), size=pred.shape[2:], mode='nearest')

        if self.loss_type == 'mse':
            diff = (pred - target) ** 2
        elif self.loss_type == 'smooth_l1':
            diff = F.smooth_l1_loss(pred, target, reduction='none', beta=0.1)
        else:  # 'l1'
            diff = (pred - target).abs()

        if mask is not None:
            m = mask.unsqueeze(1).float()           # (B, 1, H, W)
            m = F.interpolate(m, size=pred.shape[2:], mode='nearest')
            loss = (diff * m).sum() / (m.sum() + 1e-8)
        else:
            loss = diff.mean()
        return loss


class MaskedCELoss(nn.Module):
    """Cross-entropy ignoring a specific class value.

    `class_weights` (opcional): tensor 1-D de tamanho num_classes com o peso de
    cada classe, repassado a nn.CrossEntropyLoss(weight=...). Usado para mitigar
    desbalanceamento de classes — erros em classes raras passam a pesar mais.
    """
    def __init__(self, ignore_index: int = 0, num_classes: int = 3,
                 label_smoothing: float = 0.1, class_weights: torch.Tensor = None):
        super().__init__()
        self.ignore_index    = ignore_index
        self.label_smoothing = label_smoothing
        # registra os pesos como buffer p/ acompanhar device/dtype do módulo
        if class_weights is not None:
            self.register_buffer('class_weights', class_weights.float())
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        logits : (B, C, H, W)
        target : (B, H, W) int64
        """
        if logits.shape[2:] != target.shape[1:]:
            logits = F.interpolate(logits, size=target.shape[1:], mode='bilinear',
                                   align_corners=False)
        # F.cross_entropy lê o buffer já no device correto do módulo
        return F.cross_entropy(
            logits, target,
            weight          = self.class_weights,
            ignore_index    = self.ignore_index,
            label_smoothing = self.label_smoothing,
        )


class DiceLoss(nn.Module):
    """Soft Dice loss (ignores ignore_index pixels)."""
    def __init__(self, ignore_index: int = 0, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape[2:] != target.shape[1:]:
            logits = F.interpolate(logits, size=target.shape[1:], mode='bilinear',
                                   align_corners=False)
        probs  = torch.softmax(logits, dim=1)   # (B, C, H, W)
        B, C, H, W = probs.shape

        # Build valid mask
        valid = (target != self.ignore_index).float()   # (B, H, W)
        total_dice = 0.0
        valid_classes = 0
        for c in range(C):
            if c == self.ignore_index:
                continue
            p = probs[:, c, :, :] * valid             # (B, H, W)
            g = (target == c).float() * valid
            inter = (p * g).sum()
            union = p.sum() + g.sum()
            total_dice += 1 - (2 * inter + self.smooth) / (union + self.smooth)
            valid_classes += 1
        return total_dice / max(1, valid_classes)


class MultitaskLoss(nn.Module):
    """
    Combined loss for simultaneous regression and segmentation.

    L_total = w_reg * L_reg  +  w_seg * (L_ce + L_dice)

    The segmentation mask (ignore_index pixels) is also propagated to
    the regression loss so that unlabelled pixels do not contribute.

    Learnable task weights use the homoscedastic uncertainty formulation
    (Kendall et al. 2018) when `learn_weights=True`.
    """

    def __init__(self, num_classes: int, ignore_index: int = 0,
                 w_reg: float = 1.0, w_seg: float = 1.0,
                 learn_weights: bool = True, label_smoothing: float = 0.1,
                 class_weights: torch.Tensor = None, reg_loss_type: str = 'l1',
                 reg_include_bg: bool = False, reg_bg_weight: float = 0.0,
                 reg_mask_source: str = 'dist'):
        super().__init__()
        self.ignore_index   = ignore_index
        self.reg_include_bg = reg_include_bg
        self.reg_bg_weight  = float(reg_bg_weight)
        self.reg_mask_source = reg_mask_source   # 'dist' | 'label'

        self.reg_loss = MaskedMSELoss(ignore_index=ignore_index,
                                      loss_type=reg_loss_type)
        self.ce_loss  = MaskedCELoss(ignore_index=ignore_index, num_classes=num_classes,
                                     label_smoothing=label_smoothing,
                                     class_weights=class_weights)
        self.dice_loss= DiceLoss(ignore_index=ignore_index)

        if learn_weights:
            # log(sigma^2) parameters — initialised to 0
            self.log_var_reg = nn.Parameter(torch.zeros(1))
            self.log_var_seg = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('log_var_reg', torch.zeros(1))
            self.register_buffer('log_var_seg', torch.zeros(1))
            # fixed weights stored as initial values
            self.log_var_reg.fill_(-2 * np.log(w_reg + 1e-8))
            self.log_var_seg.fill_(-2 * np.log(w_seg + 1e-8))

        self.learn_weights = learn_weights

    def forward(self, reg_pred, seg_logits, reg_target, seg_target,
                reg_weight_mask=None):
        """
        reg_pred   : (B, 1, H, W)
        seg_logits : (B, C, H, W)
        reg_target : (B, 1, H, W)   — mapa de distância normalizado [0,1]
        seg_target : (B, H, W)  int64

        Máscara de validade para a regressão: dist_target > 0
          → cobre todos os pixeis dentro de qualquer copa, mesmo os que
            têm label==0 por anotação incompleta de espécie.
        Máscara para a segmentação: ignore_index no CE/Dice (inalterado)
          → continua ignorando label==0 na classificação, correto.
        """
        # ── Máscara de validade guiada pelo mapa de distância ─────
        # dist_target > 0 → pixel está dentro de alguma copa
        # dist_target == 0 → fundo verdadeiro (fora de qualquer copa)
        # Mais preciso que (seg_target != ignore_index) porque captura
        # pixeis de copa com espécie não anotada (label=0 mas dist>0).
        dist_t = reg_target if reg_target.ndim == 3 else reg_target.squeeze(1)
        # ── Máscara base de validade (dist vs label) ──────────────
        # 'dist'  → (dist > 0): captura copas com espécie não anotada, mas pode
        #           incluir SOMBRA se o mapa de distância vazar para áreas escuras.
        # 'label' → (seg_target != ignore): só dentro do polígono de espécie
        #           anotado (estilo main_FINAL) — imune a sombra.
        if getattr(self, 'reg_mask_source', 'dist') == 'label':
            # != 0 também exclui o fundo quando ele é classe treinada
            base_inside = (seg_target != self.ignore_index) & (seg_target != 0)
        else:
            base_inside = (dist_t > 0.0)
        # ── Máscara da regressão ──────────────────────────────────
        # Se reg_weight_mask vier de fora (pseudo-rotulagem), usa-a diretamente:
        # ela já codifica copas reais (peso 1), pseudo-copas (peso reduzido) e
        # fundo (0 ou bg_weight). Senão, constrói a máscara SOFT padrão.
        if reg_weight_mask is not None:
            weight_mask = reg_weight_mask
        elif self.reg_include_bg:
            weight_mask = None
        elif self.reg_bg_weight > 0.0:
            inside = base_inside.float()
            weight_mask = inside + (1.0 - inside) * self.reg_bg_weight
        else:
            weight_mask = base_inside   # bool → só copas (sem fundo)

        # ── Regression ──────────────────────────────────────────
        l_reg  = self.reg_loss(reg_pred, reg_target, mask=weight_mask)

        # ── Segmentation ────────────────────────────────────────
        l_ce   = self.ce_loss(seg_logits, seg_target)
        l_dice = self.dice_loss(seg_logits, seg_target)
        l_seg  = l_ce + l_dice

        # ── Uncertainty weighting ───────────────────────────────
        prec_reg = torch.exp(-self.log_var_reg)
        prec_seg = torch.exp(-self.log_var_seg)

        loss = (prec_reg * l_reg  + self.log_var_reg +
                prec_seg * l_seg  + self.log_var_seg)

        return loss.squeeze(), {
            'loss_reg' : l_reg.item(),
            'loss_seg' : l_seg.item(),
            'loss_ce'  : l_ce.item(),
            'loss_dice': l_dice.item(),
            'w_reg'    : prec_reg.item(),
            'w_seg'    : prec_seg.item(),
        }


class SegmentationLoss(nn.Module):
    """
    Loss para a tarefa de classificação pura (single-task).
    L = L_ce + L_dice  (mesmas componentes da parte de segmentação do multitask).
    Retorna (loss, details) para manter a mesma interface usada no run_epoch.
    """
    def __init__(self, num_classes: int, ignore_index: int = 0,
                 label_smoothing: float = 0.1, class_weights: torch.Tensor = None):
        super().__init__()
        self.ce_loss   = MaskedCELoss(ignore_index=ignore_index,
                                      num_classes=num_classes,
                                      label_smoothing=label_smoothing,
                                      class_weights=class_weights)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)

    def forward(self, seg_logits, seg_target):
        l_ce   = self.ce_loss(seg_logits, seg_target)
        l_dice = self.dice_loss(seg_logits, seg_target)
        loss   = l_ce + l_dice
        return loss, {'loss_ce': l_ce.item(), 'loss_dice': l_dice.item(),
                      'loss_seg': loss.item()}


# ─────────────────────────────────────────────────────────────
# 6.  Metrics
# ─────────────────────────────────────────────────────────────
def compute_regression_metrics(pred, target, mask=None):
    """RMSE and MAE for valid pixels."""
    if isinstance(pred,   torch.Tensor): pred   = pred.cpu().numpy()
    if isinstance(target, torch.Tensor): target = target.cpu().numpy()
    if isinstance(mask,   torch.Tensor): mask   = mask.cpu().numpy()

    pred   = pred.squeeze()
    target = target.squeeze()
    if mask is not None:
        pred   = pred[mask]
        target = target[mask]
    diff   = pred.flatten() - target.flatten()
    rmse   = float(np.sqrt((diff ** 2).mean()))
    mae    = float(np.abs(diff).mean())
    return rmse, mae


def compute_segmentation_metrics(pred_logits, target, num_classes, ignore_index=0):
    """Mean-IoU ignoring ignore_index class."""
    if isinstance(pred_logits, torch.Tensor):
        pred_cls = pred_logits.argmax(dim=1).cpu().numpy()
    else:
        pred_cls = np.argmax(pred_logits, axis=1)
    if isinstance(target, torch.Tensor): target = target.cpu().numpy()

    valid = target != ignore_index
    iou_per_class = []
    for c in range(num_classes):
        if c == ignore_index: continue
        p = (pred_cls == c) & valid
        g = (target  == c) & valid
        inter = (p & g).sum()
        union = (p | g).sum()
        if union == 0: continue
        iou_per_class.append(inter / union)
    return float(np.mean(iou_per_class)) if iou_per_class else 0.0


# ─────────────────────────────────────────────────────────────
# 7.  Training / Validation loops
# ─────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_pseudo_targets(teacher, imgs, dists,
                            conf_thresh=0.5, peak_radius=3,
                            gauss_sigma=2.0, pseudo_weight=0.3,
                            sigma_scale=2.0, device='cuda'):
    """
    Pseudo-rotulagem ONLINE para regressão de distância (self-training).

    Dado um professor congelado, detecta COPAS de ALTA CONFIANÇA que o modelo
    enxerga mas que NÃO estão anotadas (caem no fundo real, dist==0), e as
    incorpora ao alvo de regressão como "copas sintéticas" gaussianas, com peso
    reduzido. As anotações REAIS (dist>0) nunca são alteradas.

    Salvaguardas (evitam o loop de auto-confirmação):
      • só picos com confiança ≥ conf_thresh entram (alta certeza);
      • só onde o alvo real é fundo (dist==0) — verdade > pseudo, jamais
        sobrescreve anotação real;
      • peso pseudo_weight < 1 na loss → a verdade real domina o gradiente;
      • detecção por PICO (máximo local), coerente com mapa de distância que
        tem pico no centro da copa.

    Retorna
    -------
    aug_target : (B, 1, H, W) alvo = real ∪ pseudo-copas
    weight_mask: (B, H, W) peso por pixel: 1.0 nas copas reais, pseudo_weight
                 nas pseudo-copas, 0.0 no fundo restante (ignorado).
    n_pseudo   : nº de pseudo-copas plantadas no batch (para logging)
    """
    teacher.eval()
    pred = teacher(imgs)                       # (B,1,H,W) — professor é regressão
    if isinstance(pred, (list, tuple)):
        pred = pred[0]                         # multitask → usa o ramo de distância
    pred = pred.float()
    B, _, H, W = pred.shape

    real = dists.float()                       # (B,1,H,W) alvo real
    real_hw = real.squeeze(1)                  # (B,H,W)
    bg = (real_hw == 0.0)                       # fundo real (onde pode haver copa não anotada)

    # ── Detecção de picos MULTI-ESCALA ────────────────────────
    # Copas pequenas têm pico nítido (janela pequena); grandes têm pico mais
    # espalhado e mais baixo (janela maior). Um único raio + limiar favorece as
    # pequenas. Detectamos em VÁRIOS raios e unimos os picos, para pegar copas de
    # tamanhos diversos. peak_radii: lista de raios (ex.: [2,4,8]).
    if isinstance(peak_radius, (list, tuple)):
        radii = list(peak_radius)
    else:
        radii = [peak_radius]

    is_peak = torch.zeros_like(real_hw, dtype=torch.bool)
    for rad in radii:
        k = 2 * rad + 1
        pooled = F.max_pool2d(pred, kernel_size=k, stride=1, padding=rad)
        pk = (pred == pooled) & (pred >= conf_thresh)     # (B,1,H,W)
        is_peak = is_peak | (pk.squeeze(1) & bg)          # une as escalas, só no fundo

    # ── Estima o tamanho de cada copa e planta gaussiana ADAPTATIVA ──
    # O tamanho local da copa é estimado pela "extensão" do sinal predito ao
    # redor do pico: medimos quanto da vizinhança supera metade do pico. Isso
    # dá um sigma proporcional, então copas grandes ganham pseudo-copas grandes.
    pseudo = torch.zeros_like(real_hw)
    n_pseudo = int(is_peak.sum().item())
    if n_pseudo > 0:
        pred_hw = pred.squeeze(1)                          # (B,H,W)
        # Mapa de extensão local: fração da vizinhança (janela grande) acima de
        # meio-pico → proxy do TAMANHO da copa. Copa grande → área grande → sigma
        # maior. sigma vai de gauss_sigma (mín) a gauss_sigma*(1+sigma_scale).
        big = 2 * max(radii) + 1
        local_area = F.avg_pool2d(
            (pred_hw > (conf_thresh * 0.5)).float().unsqueeze(1),
            kernel_size=big, stride=1, padding=max(radii)).squeeze(1)   # (B,H,W) em [0,1]
        sigma_map = gauss_sigma * (1.0 + sigma_scale * local_area)

        bidx, yidx, xidx = torch.where(is_peak)
        H_, W_ = real_hw.shape[1], real_hw.shape[2]
        for n in range(bidx.shape[0]):
            b, y, x = int(bidx[n]), int(yidx[n]), int(xidx[n])
            s = max(0.8, min(float(sigma_map[b, y, x]),
                             gauss_sigma * (1.0 + sigma_scale)))
            r = max(1, int(round(3 * s)))
            y0, y1 = max(0, y - r), min(H_, y + r + 1)
            x0, x1 = max(0, x - r), min(W_, x + r + 1)
            ay = torch.arange(y0 - y, y1 - y, device=pred.device, dtype=torch.float32)
            ax = torch.arange(x0 - x, x1 - x, device=pred.device, dtype=torch.float32)
            gyv, gxv = torch.meshgrid(ay, ax, indexing='ij')
            g = torch.exp(-(gxv**2 + gyv**2) / (2 * s * s))
            patch = pseudo[b, y0:y1, x0:x1]
            pseudo[b, y0:y1, x0:x1] = torch.maximum(patch, g)
        pseudo = pseudo * bg.float()                       # só no fundo real

    # ── Alvo aumentado e máscara de pesos ─────────────────────
    aug_target = torch.maximum(real_hw, pseudo).unsqueeze(1)   # real ∪ pseudo
    weight_mask = torch.zeros_like(real_hw)
    weight_mask[real_hw > 0.0] = 1.0                          # copas reais: peso 1
    pseudo_region = (pseudo > 0.05) & bg
    weight_mask[pseudo_region] = pseudo_weight               # pseudo: peso reduzido
    return aug_target, weight_mask, n_pseudo


def run_epoch(model, loader, criterion, optimizer, device,
              num_classes, ignore_index, is_train: bool,
              task: str, scaler=None, teacher=None):
    model.train() if is_train else model.eval()

    do_reg = task in ('multitask', 'regression')
    do_clf = task in ('multitask', 'classification')

    total_loss = 0.0
    all_details: dict = {}
    all_rmse, all_mae, all_miou = [], [], []

    # Collect a batch for visualisation
    vis_img = vis_reg_gt = vis_seg_gt = vis_reg_pred = vis_seg_pred = None
    vis_aug_target = None        # alvo de distância aumentado (real + pseudo)
    # guarda o aug_target do 1º batch quando a pseudo-rotulagem está ativa
    _first_aug = None

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch_idx, (imgs, dists, labels) in enumerate(
                tqdm(loader, leave=False, desc='train' if is_train else 'val')):

            # imgs pode ser um tensor (single-source) ou uma lista [rgb, lidar]
            # (LateFusion). Move tudo para o device de forma transparente.
            if isinstance(imgs, (list, tuple)):
                imgs = [t.to(device) for t in imgs]   # [rgb, lidar]
                rgb_for_vis = imgs[0]                  # usado na visualização
            else:
                imgs = imgs.to(device)                 # (B, C, H, W)
                rgb_for_vis = imgs

            dists  = dists.to(device)    # (B, 1, H, W)
            labels = labels.to(device)   # (B, H, W)

            reg_pred = seg_logits = None
            with torch.amp.autocast('cuda', enabled=(scaler is not None)):
                if task == 'multitask':
                    reg_pred, seg_logits = model(imgs)
                    # ── Pseudo-rotulagem online (só DISTÂNCIA) ───────
                    # Em multitask, o professor (de regressão) injeta pseudo-copas
                    # APENAS no alvo de distância. A classificação (espécie) usa só
                    # rótulos REAIS — nunca inventamos espécie, evitando rótulo
                    # de classe errado. seg_target (labels) segue intacto.
                    if is_train and teacher is not None:
                        aug_target, w_mask, n_ps = generate_pseudo_targets(
                            teacher, imgs, dists,
                            conf_thresh   = getattr(args, 'pseudo_conf', 0.5),
                            peak_radius   = getattr(args, 'pseudo_peak_radii', [2, 4, 8]),
                            gauss_sigma   = getattr(args, 'pseudo_sigma', 2.0),
                            pseudo_weight = getattr(args, 'pseudo_weight', 0.3),
                            sigma_scale   = getattr(args, 'pseudo_sigma_scale', 2.0),
                            device        = device,
                        )
                        all_details['n_pseudo'] = all_details.get('n_pseudo', 0) + n_ps
                        bg_w = float(getattr(args, 'reg_bg_weight', 0.0))
                        if bg_w > 0.0:
                            untouched_bg = (w_mask == 0.0)
                            w_mask = w_mask + untouched_bg.float() * bg_w
                        loss, details = criterion(reg_pred, seg_logits,
                                                  aug_target, labels,
                                                  reg_weight_mask=w_mask)
                        if batch_idx == 0:
                            _first_aug = aug_target.detach()
                    else:
                        loss, details = criterion(reg_pred, seg_logits, dists, labels)
                    for k, v in details.items():
                        all_details[k] = all_details.get(k, 0.0) + v
                elif task == 'classification':
                    seg_logits = model(imgs)
                    loss, details = criterion(seg_logits, labels)
                    for k, v in details.items():
                        all_details[k] = all_details.get(k, 0.0) + v
                else:  # regression
                    reg_pred = model(imgs)
                    dist_hw = dists.squeeze(1)
                    bg_w = float(getattr(args, 'reg_bg_weight', 0.0))

                    # ── Máscara base de validade da regressão ────────
                    # 'dist'  → (dist > 0): inclui copas com espécie não anotada,
                    #           mas pode incluir SOMBRA se o mapa de distância vazar.
                    # 'label' → (label != ignore): só dentro do polígono de espécie
                    #           anotado (estilo main_FINAL) — imune a sombra.
                    _mask_src = getattr(args, 'reg_mask_source', 'dist')
                    if _mask_src == 'label':
                        # != 0 também exclui o fundo quando ele é classe treinada
                        base_inside = (labels != ignore_index) & (labels != 0)
                    else:
                        base_inside = (dist_hw > 0.0)

                    # ── Pseudo-rotulagem online (self-training) ──────
                    # Só no TREINO e se houver professor. Aumenta o alvo com
                    # copas de alta confiança detectadas no fundo não anotado.
                    if is_train and teacher is not None:
                        aug_target, w_mask, n_ps = generate_pseudo_targets(
                            teacher, imgs, dists,
                            conf_thresh   = getattr(args, 'pseudo_conf', 0.5),
                            peak_radius   = getattr(args, 'pseudo_peak_radii', [2, 4, 8]),
                            gauss_sigma   = getattr(args, 'pseudo_sigma', 2.0),
                            pseudo_weight = getattr(args, 'pseudo_weight', 0.3),
                            sigma_scale   = getattr(args, 'pseudo_sigma_scale', 2.0),
                            device        = device,
                        )
                        all_details['n_pseudo'] = all_details.get('n_pseudo', 0) + n_ps
                        # combina com o soft-mask do fundo, se ativo: pixels de
                        # fundo que NÃO viraram pseudo-copa recebem bg_w.
                        if bg_w > 0.0:
                            untouched_bg = (w_mask == 0.0)
                            w_mask = w_mask + untouched_bg.float() * bg_w
                        loss = criterion(reg_pred, aug_target, mask=w_mask)
                        if batch_idx == 0:
                            _first_aug = aug_target.detach()
                    elif getattr(args, 'reg_include_bg', False):
                        loss = criterion(reg_pred, dists, mask=None)
                    elif bg_w > 0.0:
                        inside = base_inside.float()
                        weight_mask = inside + (1.0 - inside) * bg_w
                        loss = criterion(reg_pred, dists, mask=weight_mask)
                    else:
                        loss = criterion(reg_pred, dists, mask=base_inside)

            # ── Loss NaN/Inf: descarta o batch em vez de envenenar os pesos ──
            # (um único backward com NaN deixa TODOS os pesos NaN e a rede passa
            # a prever só a classe 0, já que argmax de NaN = 0)
            if not torch.isfinite(loss):
                n_nonfinite = all_details.get('n_nonfinite_batches', 0) + 1
                all_details['n_nonfinite_batches'] = n_nonfinite
                if n_nonfinite <= 3:
                    in_ok = all(torch.isfinite(t).all().item() for t in
                                (imgs if isinstance(imgs, list) else [imgs]))
                    print(f"\n  ⚠️  loss não-finita no batch {batch_idx} "
                          f"(entrada finita={in_ok}) — batch descartado.")
                continue

            if is_train:
                optimizer.zero_grad()
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

            total_loss += loss.item()

            # ── Metrics ───────────────────────────────────────
            if do_reg:
                valid_mask = (dists.squeeze(1) > 0.0).cpu()   # dist>0 = dentro de copa
                rmse, mae  = compute_regression_metrics(
                    reg_pred.detach().cpu(), dists.cpu(), valid_mask)
                all_rmse.append(rmse); all_mae.append(mae)
            if do_clf:
                miou = compute_segmentation_metrics(
                    seg_logits.detach(), labels, num_classes, ignore_index)
                all_miou.append(miou)

            # ── Save first batch for visualisation ────────────
            if batch_idx == 0:
                vis_img     = rgb_for_vis.detach()
                vis_reg_gt  = dists.detach().squeeze(1)    # (B, H, W)
                vis_seg_gt  = labels.detach()
                if do_reg:
                    vis_reg_pred = reg_pred.detach()
                if do_clf:
                    vis_seg_pred = seg_logits.detach()
                if _first_aug is not None:
                    vis_aug_target = _first_aug.squeeze(1)  # (B, H, W) real+pseudo

    n = len(loader)
    metrics = {'loss': total_loss / n}
    if do_reg:
        metrics['rmse'] = float(np.mean(all_rmse))
        metrics['mae']  = float(np.mean(all_mae))
    if do_clf:
        metrics['miou'] = float(np.mean(all_miou))
    if all_details:
        metrics.update({k: v / n for k, v in all_details.items()})

    vis_data = (vis_img, vis_reg_gt, vis_seg_gt, vis_reg_pred, vis_seg_pred,
                vis_aug_target)
    return metrics, vis_data


# ─────────────────────────────────────────────────────────────
# 8.  Main training orchestration
# ─────────────────────────────────────────────────────────────
def _maybe_resume(args, wgt_dir, model, optimizer, scheduler, scaler, criterion, device):
    """
    Retoma o treinamento se houver um checkpoint salvo em weights/.

    Procura, em ordem: o caminho explícito --resume_path (se dado), depois
    'best_model.pth' e 'last_model.pth' dentro de wgt_dir. Restaura pesos do
    modelo, optimizer, scheduler, scaler (AMP), os parâmetros de peso de tarefa
    do criterion (MultitaskLoss) e o histórico. Define a época de início.

    Verifica compatibilidade da TAREFA: se o checkpoint foi treinado para outra
    task (ex.: multitask vs classification), as cabeças diferem e os pesos não
    casam — nesse caso avisa e começa do zero (sem restaurar), em vez de quebrar.

    Retorna (start_epoch, best_val_loss, history).
    """
    start_epoch   = 1
    best_val_loss = float('inf')
    history       = {'train': [], 'val': []}

    if not getattr(args, 'resume', True):
        print("[resume] desativado (--resume False) → treino do zero.")
        return start_epoch, best_val_loss, history

    # localiza o checkpoint
    ckpt_path = None
    if getattr(args, 'resume_path', '') :
        if os.path.isfile(args.resume_path):
            ckpt_path = args.resume_path
        else:
            print(f"[resume] --resume_path não encontrado: {args.resume_path}")
    if ckpt_path is None:
        for name in ('best_model.pth', 'last_model.pth'):
            cand = os.path.join(wgt_dir, name)
            if os.path.isfile(cand):
                ckpt_path = cand
                break

    if ckpt_path is None:
        print(f"[resume] nenhum checkpoint em {wgt_dir} → treino do zero.")
        return start_epoch, best_val_loss, history

    print(f"[resume] checkpoint encontrado: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # ── compatibilidade de task ───────────────────────────────
    ckpt_args = ckpt.get('args', {})
    ckpt_task = ckpt_args.get('task', None)
    if ckpt_task is not None and ckpt_task != args.task:
        print(f"  ⚠️  Task do checkpoint ('{ckpt_task}') ≠ task atual ('{args.task}'). "
              f"As cabeças diferem; iniciando do zero para evitar incompatibilidade.")
        return start_epoch, best_val_loss, history

    # ── compatibilidade da normalização RGB ───────────────────
    # Pesos treinados com outra escala de entrada não servem; e "começar do
    # zero" aqui sobrescreveria o best_model.pth existente. Então para.
    for key, what in (('rgb_norm', 'RGB'), ('lidar_norm', 'LiDAR')):
        ckpt_norm = ckpt_args.get(key, None)
        if ckpt_norm is not None and ckpt_norm != getattr(args, key, None):
            sys.exit(f"[resume] o checkpoint {ckpt_path} foi treinado com normalização "
                     f"{what} '{ckpt_norm}', mas a atual é '{getattr(args, key, None)}'. "
                     f"Use outro --model_dir (ou --resume False para sobrescrevê-lo).")

    # ── pesos do modelo ───────────────────────────────────────
    state = ckpt.get('state_dict', ckpt)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [resume][WARN] chaves ausentes: {missing[:4]}{'...' if len(missing)>4 else ''}")
    if unexpected:
        print(f"  [resume][WARN] chaves inesperadas: {unexpected[:4]}{'...' if len(unexpected)>4 else ''}")

    # ── optimizer / scheduler / scaler ────────────────────────
    if 'optimizer' in ckpt and ckpt['optimizer'] is not None:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except Exception as e:
            print(f"  [resume][WARN] não restaurou optimizer ({e}).")
    if 'scheduler' in ckpt and ckpt['scheduler'] is not None:
        try:
            scheduler.load_state_dict(ckpt['scheduler'])
        except Exception as e:
            print(f"  [resume][WARN] não restaurou scheduler ({e}).")
    if scaler is not None and ckpt.get('scaler') is not None:
        try:
            scaler.load_state_dict(ckpt['scaler'])
        except Exception as e:
            print(f"  [resume][WARN] não restaurou scaler AMP ({e}).")

    # ── pesos de tarefa aprendíveis (MultitaskLoss) ───────────
    if ckpt.get('criterion') is not None and hasattr(criterion, 'load_state_dict'):
        try:
            criterion.load_state_dict(ckpt['criterion'])
        except Exception as e:
            print(f"  [resume][WARN] não restaurou estado do criterion ({e}).")

    # ── época / melhor perda / histórico ──────────────────────
    last_epoch    = int(ckpt.get('epoch', 0))
    start_epoch   = last_epoch + 1
    best_val_loss = float(ckpt.get('best_val_loss', ckpt.get('val_loss', float('inf'))))
    if isinstance(ckpt.get('history'), dict):
        history = ckpt['history']
    # Split de validação diferente (ex.: checkpoint antigo sem validação, cujo
    # "val_loss" era a loss de TREINO): o melhor valor salvo não é comparável.
    split_now  = (getattr(args, 'val_fraction', None), getattr(args, 'val_block_size', None))
    split_ckpt = (ckpt_args.get('val_fraction'), ckpt_args.get('val_block_size'))
    if getattr(args, 'on_the_fly', False) and split_ckpt != split_now:
        print(f"  ⚠️  split de validação mudou {split_ckpt} → {split_now}: "
              f"melhor val_loss reiniciado (o valor salvo não é comparável).")
        best_val_loss = float('inf')

    print(f"  ✓ Retomando da época {start_epoch} "
          f"(melhor val_loss até agora: {best_val_loss:.4f})")
    if start_epoch > args.epochs:
        print(f"  ⚠️  start_epoch ({start_epoch}) > --epochs ({args.epochs}); "
              f"nada a treinar. Aumente --epochs para continuar.")
    return start_epoch, best_val_loss, history


def _resolve_ignore(args):
    """
    Resolve o parâmetro booleano --ignore_index em valores efetivos usados pelo
    resto do pipeline (perdas, métricas, dataset). Os rótulos ficam SEMPRE no
    espaço original do raster (label_shift = 0).

    --ignore_index True  (comportamento clássico):
        • ignora a classe `--ignore_class_value` (default 0 = fundo) no CE/Dice/mIoU
        • num_classes = --num_classes (ex.: 5, abrangendo 0..4)
        • effective_ignore_index = ignore_class_value

    --ignore_index False (treinar TODAS as classes, inclusive o fundo 0):
        • o fundo (0) é uma CLASSE treinada como qualquer outra
        • classes = 0..num_classes-1 (ex.: --num_classes 2 → 0=fundo, 1=alvo)
        • effective_ignore_index = -100: só valores fora de [0, num_classes)
          (nodata/espúrios) são ignorados
        • no on-the-fly, também são sorteados centros de patch no fundo

    Antes, o modo False deslocava 1..K → 0..K-1 e jogava o fundo em -100; com
    rótulos {0, 1} isso deixava UMA única classe (tudo virava 0). Para treinar
    só as K classes sem o fundo, use --ignore_index True --num_classes K+1.

    Define em args: effective_ignore_index, label_shift, bg_is_class.
    """
    use_ignore = _as_bool(args.ignore_index)
    args.use_ignore  = use_ignore
    args.label_shift = 0
    if use_ignore:
        args.effective_ignore_index = int(getattr(args, 'ignore_class_value', 0))
        print(f"[classes] ignore ATIVADO: ignorando índice "
              f"{args.effective_ignore_index} | num_classes={args.num_classes}")
    else:
        args.effective_ignore_index = -100   # CrossEntropy: nenhuma classe real ignorada
        print(f"[classes] ignore DESATIVADO: treinando todas as {args.num_classes} "
              f"classes 0..{args.num_classes-1}, fundo (0) incluído | valores fora "
              f"desse intervalo (nodata) → ignorados (-100)")
    # o fundo (0) só é classe treinada quando não é ele o índice ignorado
    args.bg_is_class = (args.effective_ignore_index != 0)
    return args


def _as_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('1', 'true', 't', 'yes', 'y', 'sim', 's')


def compute_class_weights(label_dir, num_classes, label_shift, ignore_index,
                          scheme='inverse', manual=None, beta=0.999, counts=None):
    """
    Calcula pesos por classe para a CrossEntropy a partir das frequências de
    pixels no conjunto de TREINO, para mitigar o desbalanceamento.

    Os labels em disco têm valores ORIGINAIS (ex.: 0=fundo, 1..K=classes). Aqui
    aplica-se o MESMO `label_shift` usado no dataset, de modo que as contagens
    fiquem no espaço de índices que a loss enxerga (0..num_classes-1). Pixels de
    fundo/ignorados (mapeados para `ignore_index`, tipicamente -100) NÃO contam.

    schemes
    -------
    'inverse'  : w_c = N_total / (num_classes * N_c)            (inverse frequency)
    'median'   : w_c = mediana(freq) / freq_c                   (median frequency balancing)
    'effective': w_c = (1 - beta) / (1 - beta**N_c)             (Cui et al. 2019)
    'manual'   : usa o vetor `manual` (lista de num_classes floats), normalizado.

    Retorna torch.FloatTensor de tamanho num_classes (ou None se scheme inválido).
    Os pesos são normalizados para média 1 (não altera a escala global da loss).

    `counts` (opcional): contagens por classe já calculadas (ex.: on-the-fly,
    via OnTheFlyMangroveDataset.estimate_class_counts); dispensa `label_dir`.
    """
    import glob

    if manual is not None and scheme == 'manual':
        w = np.asarray(manual, dtype=np.float64)
        if w.shape[0] != num_classes:
            raise ValueError(f"--class_weights_manual deve ter {num_classes} valores.")
        w = w / w.mean()
        return torch.tensor(w, dtype=torch.float32)

    if counts is not None:
        counts = np.asarray(counts, dtype=np.float64).copy()
    else:
        counts = np.zeros(num_classes, dtype=np.float64)
        files = sorted(glob.glob(os.path.join(label_dir, '*.npy')))
        if not files:
            print(f"⚠️  Sem labels .npy em {label_dir}; pesos de classe desativados.")
            return None

        for f in files:
            lab = np.load(f).astype(np.int64)
            if label_shift != 0:
                bg = (lab == 0)
                lab = lab + label_shift
                lab[bg] = ignore_index
            # conta apenas índices válidos [0, num_classes)
            valid = lab[(lab >= 0) & (lab < num_classes)]
            if valid.size:
                binc = np.bincount(valid.ravel(), minlength=num_classes)
                counts += binc[:num_classes]

    # a classe ignorada pela loss não entra nas frequências (nem na normalização)
    if 0 <= ignore_index < num_classes:
        counts[ignore_index] = 0

    total = counts.sum()
    if total == 0:
        print("⚠️  Nenhum pixel rotulado contado; pesos de classe desativados.")
        return None

    # evita divisão por zero em classes ausentes do treino
    safe_counts = np.where(counts > 0, counts, 1.0)
    freq = safe_counts / total

    if scheme == 'inverse':
        w = total / (num_classes * safe_counts)
    elif scheme == 'median':
        w = np.median(freq) / freq
    elif scheme == 'effective':
        eff = 1.0 - np.power(beta, safe_counts)
        w = (1.0 - beta) / eff
    else:
        print(f"⚠️  scheme de pesos desconhecido '{scheme}'; pesos desativados.")
        return None

    # classes ausentes (count 0) recebem peso 0 (não há o que aprender delas)
    w = np.where(counts > 0, w, 0.0)
    # normaliza para média 1 sobre as classes presentes
    present = w[counts > 0]
    if present.size:
        w = w / present.mean()

    # relatório
    print("[class weights] esquema:", scheme)
    for c in range(num_classes):
        pct = 100.0 * counts[c] / total
        print(f"   classe {c}: {int(counts[c]):>12d} px ({pct:5.2f}%) → peso {w[c]:.4f}")
    return torch.tensor(w, dtype=torch.float32)


def train(args):
    if not hasattr(args, 'task'):
        args = _resolve_task(args)
    args = _resolve_ignore(args)
    set_seed(args.seed, cudnn_benchmark=getattr(args, 'cudnn_benchmark', False))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── Directories ───────────────────────────────────────────
    os.makedirs(args.model_dir, exist_ok=True)
    fig_dir  = os.path.join(args.model_dir, 'figs')
    wgt_dir  = os.path.join(args.model_dir, 'weights')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(wgt_dir, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────
    # TRAIN/ and VAL/ are already separated by the patch extraction script.
    # DIST só é necessária quando há ramo de regressão (multitask/regression).
    # Em classificação pura a pasta DIST é dispensável.
    need_dist = args.task in ('multitask', 'regression')

    on_the_fly = bool(getattr(args, 'on_the_fly', False))
    latefusion = bool(getattr(args, 'latefusion', False))
    lidar_norm = getattr(args, 'lidar_norm', 'minmax')
    if latefusion and lidar_norm == 'minmax' and not on_the_fly:
        sys.exit("--lidar_norm minmax calcula min/max no raster --chm_path e exige "
                 "--on_the_fly True. Com patches pré-extraídos use --lidar_norm values.")

    # Validação do --norm_values (só no modo legado --lidar_norm values): o nº
    # de tetos deve casar com --lidar_channels (1 valor por banda do LiDAR).
    norm_values = None
    if latefusion and lidar_norm == 'values':
        norm_values = getattr(args, 'norm_values', None)
        if norm_values is not None:
            if len(norm_values) != args.lidar_channels:
                sys.exit(f"--norm_values tem {len(norm_values)} valores, mas "
                         f"--lidar_channels={args.lidar_channels}. Forneça um teto "
                         f"por banda (ex.: --lidar_channels 2 --norm_values 30 10).")
            print(f"[Norm] LiDAR normalizado por banda: {list(norm_values)} "
                  f"(clamp [0,1])")
    elif latefusion:
        print(f"[Norm] LiDAR: min-max {getattr(args, 'lidar_minmax_scope', 'band')} "
              f"calculado no raster (--lidar_norm minmax)")

    if on_the_fly:
        # ── Modo ON-THE-FLY: recorta patches dos rasters grandes ──
        missing = [k for k in ('image_path', 'ref_path', 'depth_path')
                   if not getattr(args, k, None)]
        if missing:
            sys.exit(f"--on_the_fly True exige {', '.join('--'+m for m in missing)}.")
        if getattr(args, 'latefusion', False) and not getattr(args, 'chm_path', None):
            sys.exit("--on_the_fly com --latefusion True exige --chm_path "
                     "(raster CHM/LiDAR).")

        train_set = OnTheFlyMangroveDataset(
            image_path   = args.image_path,
            ref_path     = args.ref_path,
            depth_path   = args.depth_path,
            patch_size   = args.patch_size,
            latefusion   = getattr(args, 'latefusion', False),
            chm_path     = getattr(args, 'chm_path', None),
            lidar_channels = getattr(args, 'lidar_channels', 1),
            norm_values  = norm_values,
            lidar_norm   = lidar_norm,
            lidar_minmax_scope = getattr(args, 'lidar_minmax_scope', 'band'),
            label_shift  = args.label_shift,
            bg_ignore_value = args.effective_ignore_index,
            samples      = getattr(args, 'samples', None),
            augment      = True,
            oversample   = getattr(args, 'oversample', True),
            seed         = getattr(args, 'seed', 42),
            rgb_res      = getattr(args, 'rgb_resolution', 0.25),
            chm_res      = getattr(args, 'chm_resolution', 1.0),
            require_full_label = getattr(args, 'require_full_label', False),
            num_classes  = args.num_classes,
            bg_as_class  = args.bg_is_class,
            bg_center_ratio = getattr(args, 'bg_center_ratio', 1.0),
            rgb_normalize   = getattr(args, 'rgb_normalize', False),
            rgb_norm_method = getattr(args, 'rgb_norm_method', 'minmax'),
            rgb_norm_const  = getattr(args, 'rgb_norm_const', 255.0),
            val_fraction    = getattr(args, 'val_fraction', 0.2),
            val_block_size  = getattr(args, 'val_block_size', 1024),
            val_samples     = getattr(args, 'val_samples', 5000),
        )
        # validação = blocos espaciais reservados do mesmo raster (mesma
        # memória, sem recarregar). None se --val_fraction 0.
        val_set = train_set.validation_view()
        if val_set is None:
            print("  ⚠️  --val_fraction 0: SEM validação — o 'best' será escolhido "
                  "pela loss de TREINO (não mede generalização).")
        # registra a normalização efetiva em args → vai para o checkpoint e o
        # predict aplica exatamente a mesma
        args.rgb_norm = train_set.rgb_norm
        args.rgb_min = (train_set.rgb_min.tolist()
                        if train_set.rgb_min is not None else None)
        args.rgb_max = (train_set.rgb_max.tolist()
                        if train_set.rgb_max is not None else None)
        # idem p/ o LiDAR (min/max por banda vão para o checkpoint)
        args.lidar_norm = lidar_norm if latefusion else None
        args.lidar_min = (train_set.lidar_min.tolist()
                          if train_set.lidar_min is not None else None)
        args.lidar_max = (train_set.lidar_max.tolist()
                          if train_set.lidar_max is not None else None)
        args.norm_values = norm_values
        if latefusion:
            # patch CHM medido nos rasters → build_model monta o braço CHM p/
            # ele; chm_resolution acompanha a razão medida (checkpoint/predict)
            args.chm_patch = int(train_set.chm_patch)
            args.chm_resolution = float(args.rgb_resolution / train_set.chm_ratio)

    else:
        args.rgb_norm = 'constant'   # MangroveDataset divide por 255
        args.rgb_norm_const = 255.0
        args.rgb_min = args.rgb_max = None
        args.lidar_norm = 'values' if latefusion else None
        args.lidar_min = args.lidar_max = None
        train_set = MangroveDataset(
            root_dir   = os.path.join(args.data_dir, 'TRAIN'),
            multitask  = True,    # sempre carrega img+dist+label; a task escolhe o uso
            augment    = True,    # augmentation only on train
            latefusion = getattr(args, 'latefusion', False),
            label_shift= args.label_shift,
            require_dist = need_dist,
            norm_values = getattr(args, 'norm_values', None),
        )
        val_set = MangroveDataset(
            root_dir   = os.path.join(args.data_dir, 'VAL'),
            multitask  = True,
            augment    = False,
            latefusion = getattr(args, 'latefusion', False),
            label_shift= args.label_shift,
            require_dist = need_dist,
            norm_values = getattr(args, 'norm_values', None),
        )

    # Opções de performance do DataLoader. persistent_workers evita recriar os
    # workers a cada época (reduz o "engasgo" no início de cada época e ao
    # retomar). prefetch_factor antecipa batches. Ambos só valem com workers>0.
    loader_kw = {}
    if args.num_workers > 0:
        loader_kw['persistent_workers'] = True
        loader_kw['prefetch_factor'] = getattr(args, 'prefetch_factor', 4)

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        **loader_kw
    )
    if val_set is not None:
        # on-the-fly: a validação roda no processo principal (os patches já
        # estão em RAM, recortar é barato). Workers extras copiariam os rasters
        # inteiros (vários GB) para mais processos no Windows.
        val_workers = 0 if on_the_fly else args.num_workers
        val_loader = DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False,
            num_workers=val_workers, pin_memory=True,
            **(loader_kw if val_workers > 0 else {})
        )
    else:
        val_loader = None
    print(f"Train: {len(train_set)} | Val: "
          f"{len(val_set) if val_set is not None else 'desativada'}")
    if args.num_workers == 0:
        print("  ⚠️  [perf] --num_workers 0: carregamento no processo principal "
              "(lento, sem leitura antecipada). Para acelerar — especialmente a "
              "primeira época ao retomar — use --num_workers 4 (ou 8).")

    # ── Nº de bandas da imagem: lido dos dados, prevalece sobre --channels ──
    if on_the_fly:
        n_img = int(train_set.image.shape[0])
    else:
        x_amostra = train_set[0][0]
        n_img = int((x_amostra[0] if isinstance(x_amostra, (list, tuple)) else x_amostra).shape[0])
    if n_img != args.channels:
        print(f"  ⚠️  [channels] a imagem tem {n_img} banda(s), mas --channels={args.channels}; "
              f"usando {n_img}.")
        args.channels = n_img

    # ── Model ─────────────────────────────────────────────────
    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # ── Professor para pseudo-rotulagem online (self-training) ──
    # Carrega um checkpoint de REGRESSÃO congelado. A cada batch detecta copas
    # de alta confiança no fundo NÃO anotado e as injeta no alvo de DISTÂNCIA com
    # peso reduzido (ver generate_pseudo_targets). Funciona com aluno de task
    # 'regression' OU 'multitask' — em multitask, afeta só a distância; a
    # classificação (espécie) usa apenas rótulos reais.
    teacher = None
    if getattr(args, 'pseudo_teacher', None):
        if args.task not in ('regression', 'multitask'):
            print(f"  [pseudo] AVISO: pseudo-rotulagem vale p/ task regression|multitask; "
                  f"task atual={args.task}. Ignorando professor.")
        else:
            print(f"  [pseudo] carregando professor (regressão): {args.pseudo_teacher}")
            # O professor é SEMPRE um modelo de REGRESSÃO, independentemente da
            # task do aluno. Construímos com um args clonado e task='regression'
            # para que a arquitetura case com o checkpoint do professor.
            import copy as _copy
            t_args = _copy.copy(args)
            t_args.task = 'regression'
            t_args.task_multitask = False
            t_args.task_classification = False
            t_args.task_regression = True
            teacher = build_model(t_args).to(device)
            tck = torch.load(args.pseudo_teacher, map_location=device, weights_only=False)
            tstate = tck.get('state_dict', tck)
            tstate = {k.replace('module.', ''): v for k, v in tstate.items()}
            missing, unexpected = teacher.load_state_dict(tstate, strict=False)
            if missing or unexpected:
                print(f"  [pseudo][WARN] load_state_dict: {len(missing)} missing / "
                      f"{len(unexpected)} unexpected — confira se o professor é um "
                      f"modelo de REGRESSÃO com o mesmo --arch/--latefusion/--fusion "
                      f"e mesmas dimensões do aluno.")
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            print(f"  [pseudo] professor congelado | aluno={args.task} "
                  f"| conf≥{getattr(args,'pseudo_conf',0.5)} "
                  f"| peso={getattr(args,'pseudo_weight',0.3)} "
                  f"| sigma={getattr(args,'pseudo_sigma',2.0)}")

    # ── Pesos de classe (desbalanceamento) ───────────────────
    # Calculados a partir das frequências de pixels no TRAIN, no espaço de
    # índices que a loss enxerga (após label_shift). Não se aplica à regressão.
    class_weights = None
    if args.task in ('classification', 'multitask') and \
       getattr(args, 'class_weights', 'none') != 'none':
        manual = getattr(args, 'class_weights_manual', None)
        # on-the-fly: não há TRAIN/LABEL; estima as frequências nos patches
        # sorteados do próprio raster de referência.
        counts = None
        if on_the_fly and args.class_weights != 'manual':
            counts = train_set.estimate_class_counts(args.num_classes,
                                                     seed=getattr(args, 'seed', 42))
        class_weights = compute_class_weights(
            label_dir    = os.path.join(args.data_dir, 'TRAIN', 'LABEL'),
            num_classes  = args.num_classes,
            label_shift  = args.label_shift,
            ignore_index = args.effective_ignore_index,
            scheme       = args.class_weights,
            manual       = manual,
            counts       = counts,
        )

    # ── Loss ──────────────────────────────────────────────────
    optimizer_params = list(model.parameters())
    if args.task == 'multitask':
        criterion = MultitaskLoss(
            num_classes     = args.num_classes,
            ignore_index    = args.effective_ignore_index,
            w_reg           = args.w_reg,
            w_seg           = args.w_seg,
            learn_weights   = args.learn_task_weights,
            label_smoothing = args.label_smoothing,
            class_weights   = class_weights,
            reg_loss_type   = args.reg_loss,
            reg_include_bg  = getattr(args, 'reg_include_bg', False),
            reg_bg_weight   = getattr(args, 'reg_bg_weight', 0.0),
            reg_mask_source = getattr(args, 'reg_mask_source', 'dist'),
        ).to(device)
        # Include loss task weights in optimiser if learnable
        if args.learn_task_weights:
            optimizer_params += [criterion.log_var_reg, criterion.log_var_seg]
    elif args.task == 'classification':
        criterion = SegmentationLoss(
            num_classes     = args.num_classes,
            ignore_index    = args.effective_ignore_index,
            label_smoothing = args.label_smoothing,
            class_weights   = class_weights,
        ).to(device)
    else:  # regression
        criterion = MaskedMSELoss(ignore_index=args.effective_ignore_index,
                                  loss_type=args.reg_loss).to(device)

    # ── Optimiser & LR scheduler ──────────────────────────────
    #optimizer = torch.optim.AdamW(
    #    optimizer_params,
    #    lr           = args.lr,
    #    weight_decay = args.weight_decay,
    #)
    
    # region OPTIMIZER
    # Very important! Only with SGD the regression model is capable of capturing multiple crowns
    
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=0.9,
        weight_decay=args.weight_decay
    )

    # Warm-up + cosine decay
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # Mixed precision scaler
    use_amp = args.amp and device.type == 'cuda'
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    # ── Resume (se houver checkpoint em weights/) ─────────────
    start_epoch, best_val_loss, history = _maybe_resume(
        args, wgt_dir, model, optimizer, scheduler, scaler, criterion, device)

    # ── Critério de "melhor modelo" ───────────────────────────
    # Para REGRESSÃO pura, a 'loss' de validação é dominada pelo FUNDO
    # (a maioria dos pixels é dist==0), então o "melhor por loss" tende a
    # cair nas primeiras épocas, quando o modelo só prevê fundo — e as copas
    # ficam fracas. Selecionamos pelo MAE calculado SÓ DENTRO das copas
    # (dist>0), que mede o que realmente importa. Para classificação e
    # multitask, mantemos a 'loss' como critério.
    select_by_reg = (args.task == 'regression')
    best_reg_mae  = float('inf')
    # Em resume de regressão, recupera o melhor MAE (dentro das copas) já
    # observado, lendo do histórico salvo — assim o critério de "best" não
    # reinicia e não sobrescreve um best melhor por um pior.
    if select_by_reg and isinstance(history.get('val'), list):
        maes = [m['mae'] for m in history['val']
                if isinstance(m, dict) and m.get('mae') is not None]
        if maes:
            best_reg_mae = float(min(maes))

    # ── Training loop ─────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{args.epochs}  |  LR: {scheduler.get_last_lr()[0]:.2e}")

        # ── Train ─────────────────────────────────────────────
        train_metrics, train_vis = run_epoch(
            model, train_loader, criterion, optimizer, device,
            args.num_classes, args.effective_ignore_index, is_train=True,
            task=args.task, scaler=scaler, teacher=teacher,
        )
        scheduler.step()

        # ── Validate ──────────────────────────────────────────
        if val_loader is not None:
            val_metrics, val_vis = run_epoch(
                model, val_loader, criterion, None, device,
                args.num_classes, args.effective_ignore_index, is_train=False,
                task=args.task, scaler=None,
            )
        else:
            # Sem validação (modo on-the-fly): usa as métricas de TREINO como
            # referência para print e seleção de "best". A figura de validação
            # também passa a usar a do treino.
            val_metrics, val_vis = train_metrics, train_vis

        history['train'].append(train_metrics)
        history['val'].append(val_metrics)

        # ── Print metrics ─────────────────────────────────────
        def _fmt(m):
            s = f"loss={m['loss']:.4f}"
            if 'rmse' in m: s += f"  rmse={m['rmse']:.4f}  mae={m['mae']:.4f}"
            if 'miou' in m: s += f"  mIoU={m['miou']:.4f}"
            return s

        print(f"  Train → {_fmt(train_metrics)}")
        print(f"  Val   → {_fmt(val_metrics)}")
        if args.task == 'multitask' and args.learn_task_weights:
            wr = float(np.exp(-criterion.log_var_reg.item()))
            ws = float(np.exp(-criterion.log_var_seg.item()))
            print(f"  Task weights → reg={wr:.3f}  seg={ws:.3f}")

        # ── Checkpoint: best weights ──────────────────────────
        # Regressão → seleciona pelo MAE dentro das copas (dist>0).
        # Demais tasks → seleciona pela loss de validação.
        if select_by_reg and ('mae' in val_metrics):
            cur_metric  = val_metrics['mae']
            is_best     = cur_metric < best_reg_mae
            metric_name = 'val_mae(copa)'
        else:
            cur_metric  = val_metrics['loss']
            is_best     = cur_metric < best_val_loss
            metric_name = 'val_loss'

        if is_best:
            if select_by_reg and ('mae' in val_metrics):
                best_reg_mae = cur_metric
            else:
                best_val_loss = cur_metric
            best_path = os.path.join(wgt_dir, 'best_model.pth')
            torch.save({
                'epoch'        : epoch,
                'state_dict'   : model.state_dict(),
                'optimizer'    : optimizer.state_dict(),
                'scheduler'    : scheduler.state_dict(),
                'scaler'       : scaler.state_dict() if scaler is not None else None,
                'criterion'    : criterion.state_dict() if hasattr(criterion, 'state_dict') else None,
                'val_loss'     : val_metrics['loss'],
                'val_mae'      : val_metrics.get('mae'),
                'val_rmse'     : val_metrics.get('rmse'),
                'best_metric'  : cur_metric,
                'best_metric_name': metric_name,
                'best_val_loss': best_val_loss,
                'history'      : history,
                'args'         : vars(args),
            }, best_path)
            print(f"  ✓ Best weights saved → weights/best_model.pth  ({metric_name}={cur_metric:.4f})")

        # ── Checkpoint: periodic weights ──────────────────────
        # Salva estado COMPLETO para permitir retomar o treino fielmente.
        if epoch % args.save_every == 0:
            period_path = os.path.join(wgt_dir, f'epoch_{epoch:04d}.pth')
            torch.save({
                'epoch'        : epoch,
                'state_dict'   : model.state_dict(),
                'optimizer'    : optimizer.state_dict(),
                'scheduler'    : scheduler.state_dict(),
                'scaler'       : scaler.state_dict() if scaler is not None else None,
                'criterion'    : criterion.state_dict() if hasattr(criterion, 'state_dict') else None,
                'val_loss'     : val_metrics['loss'],
                'best_val_loss': best_val_loss,
                'history'      : history,
                'args'         : vars(args),
            }, period_path)
            print(f"  · Periodic weights saved → weights/epoch_{epoch:04d}.pth")

        # ── Checkpoint: last weights (todo epoch) ─────────────
        # Salva o estado COMPLETO da última época treinada a cada epoch,
        # garantindo um checkpoint resumível atualizado mesmo se o treino
        # for interrompido antes do fim.
        last_path = os.path.join(wgt_dir, 'last_model.pth')
        torch.save({
            'epoch'        : epoch,
            'state_dict'   : model.state_dict(),
            'optimizer'    : optimizer.state_dict(),
            'scheduler'    : scheduler.state_dict(),
            'scaler'       : scaler.state_dict() if scaler is not None else None,
            'criterion'    : criterion.state_dict() if hasattr(criterion, 'state_dict') else None,
            'val_loss'     : val_metrics['loss'],
            'best_val_loss': best_val_loss,
            'history'      : history,
            'args'         : vars(args),
        }, last_path)

        # ── Figures every N epochs ────────────────────────────
        if epoch % args.plot_every == 0 or epoch == 1:
            _save_vis(train_vis, fig_dir, epoch, 'train', args)
            _save_vis(val_vis,   fig_dir, epoch, 'val',   args)

    # ── Final checkpoint → weights/ ───────────────────────────
    # Salva estado COMPLETO usando a última época efetivamente treinada
    # (não necessariamente args.epochs, p. ex. se retomado já no fim).
    # Observação: last_model.pth já é salvo ao fim de cada época; aqui
    # apenas garantimos um salvamento final consistente.
    last_epoch_done = epoch if 'epoch' in dir() else start_epoch - 1
    last_path = os.path.join(wgt_dir, 'last_model.pth')
    torch.save({
        'epoch'        : last_epoch_done,
        'state_dict'   : model.state_dict(),
        'optimizer'    : optimizer.state_dict(),
        'scheduler'    : scheduler.state_dict(),
        'scaler'       : scaler.state_dict() if scaler is not None else None,
        'criterion'    : criterion.state_dict() if hasattr(criterion, 'state_dict') else None,
        'val_loss'     : (history['val'][-1]['loss'] if history['val'] else float('inf')),
        'best_val_loss': best_val_loss,
        'history'      : history,
        'args'         : vars(args),
    }, last_path)

    # ── Loss curves (saved directly in model_dir) ─────────────
    _plot_loss_curves(history, args.model_dir, args.task)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Model directory: {args.model_dir}")


# ─────────────────────────────────────────────────────────────
# 9.  Visualisation helpers
# ─────────────────────────────────────────────────────────────
def _save_vis(vis_data, fig_dir, epoch, split, args):
    """Desempacota os tensores de visualização e chama plot_figures conforme a task.

    Cada task plota só o que lhe diz respeito:
      • regression     → RGB, CHM referência, CHM previsto
      • classification → RGB, labels referência, labels previstos
      • multitask      → as 5 linhas
    """
    vis_img, vis_reg_gt, vis_seg_gt, vis_reg_pred, vis_seg_pred, vis_aug = vis_data
    if vis_img is None:
        return

    plot_figures(
        img       = vis_img,
        ref       = vis_seg_gt,     # usado só se a task incluir classificação
        pred      = vis_seg_pred,
        depth     = vis_reg_gt,     # usado só se a task incluir regressão
        dist      = vis_reg_pred,
        model_dir = fig_dir,
        epoch     = epoch,
        set_name  = split,
        task      = args.task,
        aug_target = vis_aug,       # alvo de distância aumentado (real + pseudo)
    )


def _plot_loss_curves(history, model_dir, task):
    """Save training/validation loss curves (métricas conforme a task)."""
    epochs = range(1, len(history['train']) + 1)
    keys   = ['loss']
    if task in ('multitask', 'regression'):
        keys += ['rmse', 'mae']
    if task in ('multitask', 'classification'):
        keys.append('miou')

    fig, axes = plt.subplots(1, len(keys), figsize=(5 * len(keys), 4))
    if len(keys) == 1: axes = [axes]

    for ax, key in zip(axes, keys):
        tr = [m.get(key, np.nan) for m in history['train']]
        vl = [m.get(key, np.nan) for m in history['val']]
        ax.plot(epochs, tr, label='train')
        ax.plot(epochs, vl, label='val')
        ax.set_title(key); ax.legend(); ax.set_xlabel('Epoch')

    plt.tight_layout()
    path = os.path.join(model_dir, 'loss_curves.pdf')
    plt.savefig(path, dpi=150, bbox_inches='tight', format='pdf')
    plt.close()
    print(f"Loss curves saved: {path}")


# ─────────────────────────────────────────────────────────────
# region ARG parser
# ─────────────────────────────────────────────────────────────
def get_args():
    p = argparse.ArgumentParser(
        description='Multitask training: CHM regression + species segmentation')

    # Paths
    p.add_argument('--data_dir',  type=str,
                   default=r'E:\MANGROVE_SPECIES_MAPPING\DATASET_PATCHES_ITCs_SPECIES_CLASS_RGB_CHM_GROUND',
                   help='Root dataset directory (deve conter TRAIN/ e VAL/, cada um '
                        'com IMAGE/, DIST/, LABEL/ e — em modo LateFusion — LIDAR/)')
    p.add_argument('--model_dir', type=str, default=r"F:\Consultorias\Bioflore\Marajo\deteccao_macicos\codes\multitask_PAN_MULTI_FUSION_without_background_macicos_only_with_dist_map",
                   help='Output directory for checkpoints and figures')

    # ── Extração de patches ON-THE-FLY (estilo main_FINAL) ────
    # Quando --on_the_fly True, o treino NÃO usa patches pré-extraídos de
    # --data_dir; em vez disso recorta patches sob demanda dos rasters
    # grandes abaixo, centrados em pixels rotulados, com oversampling.
    p.add_argument('--on_the_fly', type=_as_bool, default=True,
                   help='Se True, extrai patches on-the-fly dos rasters grandes '
                        '(--image_path/--ref_path/--depth_path[/--chm_path]) em vez '
                        'de ler patches pré-extraídos de --data_dir.')
    p.add_argument('--image_path', type=str, default=r"F:\Consultorias\Bioflore\Marajo\deteccao_macicos\images\CBERS_2025\CBERS_4A_WPM_20250617_212_114_L4_BAND0_CUT.tif",
                   help='[on-the-fly] Ortofoto RGB (raster grande).')
    p.add_argument('--ref_path', type=str, default=r"F:\Consultorias\Bioflore\Marajo\deteccao_macicos\images\TRAIN_TEST_MASKS_PAN_2025\ITCs_labels_train.tif",
                   help='[on-the-fly] Raster de rótulos de espécie (0=fundo).')
    p.add_argument('--depth_path', type=str, default=r"F:\Consultorias\Bioflore\Marajo\deteccao_macicos\images\TRAIN_TEST_MASKS_PAN_2025\ITCs_distance_train.tif",
                   help='[on-the-fly] Raster de distância/altura (alvo de regressão).')
    p.add_argument('--chm_path', type=str, default=r"F:\Consultorias\Bioflore\Marajo\deteccao_macicos\images\CBERS_2025\composicao_colorida_2025_cut.tif",
                   help='[on-the-fly + LateFusion] Raster CHM/LiDAR (pode ter '
                        'resolução diferente do RGB; recortado por geo-referência).')
    p.add_argument('--samples', type=int, default=50000,
                   help='[on-the-fly] Nº de patches por época. None = usa todos os '
                        'centros após oversampling. Se menor que o total, sorteia '
                        'esse nº de centros a cada época (com reposição).')
    p.add_argument('--oversample', type=_as_bool, default=True,
                   help='[on-the-fly] Oversampling das classes minoritárias '
                        '(equilibra os centros de patch por classe).')
    p.add_argument('--rgb_normalize', type=_as_bool, default=True,
                   help='[on-the-fly] True = normaliza a imagem --image_path pelo '
                        'método --rgb_norm_method. False (padrão) = usa os valores '
                        'como estão (imagem já normalizada 0–1). Em ambos os casos '
                        'NaN → 0. Salvo no checkpoint e reaplicado no predict.')
    p.add_argument('--rgb_norm_method', type=str, default='minmax',
                   choices=['minmax', 'constant'],
                   help="[on-the-fly, só com --rgb_normalize True] 'minmax' = "
                        "(x - min)/(max - min) por banda, min/max calculados no "
                        "raster ignorando NaN (resultado em 0–1). 'constant' = "
                        "x / --rgb_norm_const.")
    p.add_argument('--rgb_norm_const', type=float, default=255.0,
                   help="[on-the-fly] Constante divisora quando --rgb_norm_method "
                        "constant (ex.: 255 p/ imagem uint8, 10000 p/ reflectância "
                        "Sentinel-2 inteira).")
    p.add_argument('--val_fraction', type=float, default=0.2,
                   help='[on-the-fly] Fração dos blocos espaciais do raster de '
                        'treino reservada para VALIDAÇÃO (sorteada à parte entre '
                        'blocos com e sem rótulo). O melhor modelo é escolhido pela '
                        'loss nesses blocos, que a rede nunca vê no treino. 0 = sem '
                        'validação (best pela loss de treino — não recomendado).')
    p.add_argument('--val_block_size', type=int, default=1024,
                   help='[on-the-fly] Lado (px) dos blocos do split espacial. Deve '
                        'ser >= --patch_size. Blocos maiores = validação mais '
                        'independente (menos autocorrelação espacial com o treino).')
    p.add_argument('--val_samples', type=int, default=2000,
                   help='[on-the-fly] Nº máximo de patches de validação (conjunto '
                        'FIXO, o mesmo em todas as épocas).')
    p.add_argument('--bg_center_ratio', type=float, default=1.0,
                   help='[on-the-fly, só com o fundo como classe (--ignore_index '
                        'False)] Nº de centros de patch sorteados no FUNDO (0) em '
                        'relação ao nº de centros rotulados (!= 0). 1.0 = mesma '
                        'quantidade. Aumente se a rede gerar muitos falsos positivos.')
    p.add_argument('--require_full_label', type=_as_bool, default=False,
                   help='[on-the-fly] Se True, EXCLUI do treino qualquer patch que '
                        'contenha pixels de classe 0 (não-rotulado). O patch inteiro '
                        'é descartado — a classe 0 não entra no treino nem é ignorada '
                        'via ignore_index; apenas patches TOTALMENTE rotulados são '
                        'usados. Útil quando você não quer treinar com a classe 0.')
    p.add_argument('--reg_mask_source', type=str, default='label',
                   choices=['dist', 'label'],
                   help="Fonte da máscara de validade da REGRESSÃO. 'dist' (padrão): "
                        "pixels com mapa de distância > 0 — captura copas com espécie "
                        "não anotada, mas pode incluir SOMBRA se o mapa de distância "
                        "vazar para áreas escuras. 'label' (estilo main_FINAL): só "
                        "pixels dentro do polígono de ESPÉCIE anotado (label != fundo) "
                        "— imune a sombra, pois sombra não está na anotação de espécie.")

    # Architecture
    p.add_argument('--arch', type=str, default='manet',
                   choices=['unet', 'unetplusplus', 'manet', 'linknet', 'BPATUNet', 'resunet'],
                   help='Encoder-decoder backbone (usado em cada braço na LateFusion)')
    p.add_argument('--pretrained', action='store_true',
                   help='Use ImageNet-pretrained encoder (smp models)')
    p.add_argument('--channels', type=int, default=1,
                   help='Nº de bandas da imagem de entrada. Detectado automaticamente '
                        'dos dados (prevalece sobre este valor se divergir). Com < 3 '
                        'bandas as figuras mostram só a banda 1 em tons de cinza.')

    # ── Seleção de tarefa (escolha UMA; ver resolução abaixo) ──
    # Conforme pedido: três flags booleanas. Se mais de uma for True a
    # prioridade é multitask > classification > regression; se nenhuma, multitask.
    p.add_argument('--task_multitask', type=_as_bool, default=True,
                   help='Treina regressão (distância) + classificação simultâneas.')
    p.add_argument('--task_classification', type=_as_bool, default=False,
                   help='Treina APENAS a classificação (single-task).')
    p.add_argument('--task_regression', type=_as_bool, default=False,
                   help='Treina APENAS a regressão de distância (single-task).')

    # ── LateFusion (RGB + CHM/LiDAR) ──────────────────────────
    p.add_argument('--latefusion', type=_as_bool, default=True,
                   help='Quando True treina a rede multitask em modo LateFusion: '
                        'um braço processa os patches RGB (IMAGE/) e outro '
                        'braço processa os patches CHM/LiDAR (LIDAR/) em sua resolução '
                        'nativa. As features são fundidas tardiamente e enviadas a duas '
                        'cabeças (mapa de distância + imagem classificada). '
                        'Requer que o dataset tenha sido gerado com latefusion=True '
                        '(pasta LIDAR/ presente).')
    p.add_argument('--fusion', type=str, default='concat',
                   choices=['concat', 'gated', 'xattn'],
                   help="Estratégia de fusão RGB⊕LiDAR (só vale com --latefusion): "
                        "'concat' = late fusion clássica (concatenação); "
                        "'gated' = gating cross-modal (squeeze-excite condicionado "
                        "na outra modalidade, pondera quanto cada fonte contribui); "
                        "'xattn' = atenção espacial cruzada bidirecional (cada "
                        "modalidade atende à outra). Todas alimentam as mesmas "
                        "cabeças; 'gated' e 'xattn' adicionam parâmetros de fusão.")
    p.add_argument('--lidar_channels', type=int, default=4,
                   help='Número de canais do raster CHM/LiDAR.')
    p.add_argument('--lidar_norm', type=str, default='minmax',
                   choices=['minmax', 'values'],
                   help="Normalização das --lidar_channels bandas do raster "
                        "--chm_path (LateFusion). 'minmax' (padrão) = (x - min)/"
                        "(max - min), com min/max calculados no raster (área que "
                        "cobre o RGB, ignorando NaN) e salvos no checkpoint p/ o "
                        "predict; NaN → 0. 'values' = legado: divide pelos tetos "
                        "--norm_values.")
    p.add_argument('--lidar_minmax_scope', type=str, default='band',
                   choices=['band', 'global'],
                   help="Com --lidar_norm minmax: 'band' = min/max próprios de cada "
                        "banda (cada banda vira 0–1); 'global' = um único min/max "
                        "sobre todas as bandas (preserva as escalas relativas entre "
                        "bandas, ex.: embeddings).")
    p.add_argument('--norm_values', type=float, nargs='+', default=[350],
                   help='[só com --lidar_norm values] Teto de normalização POR BANDA '
                        'do raster CHM/LiDAR. Cada banda i é dividida por '
                        'norm_values[i] e clampada em [0, 1]. Ex.: --norm_values 30 10 '
                        '→ 1ª banda (CHM) /30, 2ª banda /10. O nº de valores deve ser '
                        'igual a --lidar_channels.')
    p.add_argument('--patch_size', type=int, default=64,
                   help='Tamanho do patch RGB (px). Usado só para validar a '
                        'compatibilidade do patch CHM com o backbone na LateFusion.')
    p.add_argument('--rgb_resolution', type=float, default=2,
                   help='Resolução da imagem --image_path (m/px). No on-the-fly a '
                        'razão entre as resoluções é medida nos rasters (este valor '
                        'só gera aviso se não conferir).')
    p.add_argument('--chm_resolution', type=float, default=8,
                   help='Resolução do --chm_path (m/px). No on-the-fly é medida nos '
                        'rasters e o patch CHM = patch_size × res_img/res_chm.')
    p.add_argument('--dist_guided', type=_as_bool, default=False,
                   help='Distance-guided segmentation: usa sigmoid(dist_pred) como '
                        'mapa de atenção espacial que pondera as features antes da '
                        'classificação. Recomendado quando o target de regressão é '
                        'mapa de distância euclidiana (pico no centro da copa).')
    p.add_argument('--num_classes', type=int, default=6,
                   help='Número de classes de segmentação, contando o fundo 0 '
                        '(rótulos 0..num_classes-1). Ex.: rótulos {0,1} → 2. Com '
                        '--ignore_index True a classe ignorada também entra na '
                        'contagem (ex.: 5 = fundo + 4 espécies).')
    p.add_argument('--ignore_index', type=_as_bool, default=True,
                   help='Se True, ignora a classe --ignore_class_value (padrão 0 = '
                        'fundo/no-data) na loss e nas métricas, mantendo os rótulos '
                        'como estão. Se False (padrão), treina TODAS as classes '
                        '0..num_classes-1, INCLUINDO o fundo 0 (ex.: --num_classes 2 '
                        '→ 0=fundo, 1=alvo). Só valores fora desse intervalo '
                        '(nodata) são ignorados. No on-the-fly, também sorteia '
                        'centros de patch no fundo (ver --bg_center_ratio).')
    p.add_argument('--ignore_class_value', type=int, default=0,
                   help='Qual índice de classe ignorar quando --ignore_index True '
                        '(tipicamente 0 = fundo/no-data).')

    # Loss weights
    p.add_argument('--w_reg',  type=float, default=1.5, help='Regression loss weight')
    p.add_argument('--w_seg',  type=float, default=1.0, help='Segmentation loss weight')
    p.add_argument('--reg_loss', type=str, default='l1',
                   choices=['l1', 'mse', 'smooth_l1'],
                   help="Função de perda da regressão de distância. 'l1' (padrão) dá "
                        "gradiente constante mesmo p/ alvos pequenos, então copas de "
                        "sinal fraco (bordas, árvores menores) são aprendidas — gera "
                        "mapas que 'pegam todas as árvores'. 'mse' penaliza pouco erros "
                        "pequenos e tende a ignorar copas de distância baixa. "
                        "'smooth_l1' é o híbrido de Huber.")
    p.add_argument('--reg_bias_init', type=float, default=0.0,
                   help="Bias inicial da última conv da cabeça de regressão. "
                        "CRÍTICO: valores muito negativos (ex.: -4) fazem sigmoid≈0 e "
                        "saturam o gradiente (derivada da sigmoid ≈0), impedindo a rede "
                        "de 'levantar' as copas — muitas árvores nunca são marcadas. "
                        "Default 0.0 (sigmoid=0.5) dá gradiente saudável.")
    p.add_argument('--reg_include_bg', type=_as_bool, default=False,
                   help="Inclui o FUNDO (dist=0) no treino da regressão. Default "
                        "False = mascara o fundo, treinando SÓ dentro das copas "
                        "(dist>0). MANTENHA False quando a anotação é INCOMPLETA "
                        "(nem toda copa do patch está rotulada): aí o fundo contém "
                        "copas não anotadas, e treinar nele ensinaria a rede a "
                        "suprimir árvores. Só use True se TODAS as copas estiverem "
                        "anotadas em cada patch.")
    p.add_argument('--reg_bg_weight', type=float, default=0.0,
                   help="SOFT-MASK do fundo na regressão: peso dado aos pixels de "
                        "fundo (dist=0); as copas sempre pesam 1.0. 0.0 (padrão) = "
                        "fundo totalmente ignorado. Um valor pequeno (ex.: 0.1) dá um "
                        "leve sinal de contraste 'fundo→0' sem punir fortemente as "
                        "copas NÃO anotadas. CUIDADO: com anotação incompleta (~50%), "
                        "o fundo contém muitas copas reais; use peso baixo (0.05–0.1) "
                        "e suba aos poucos — peso alto suprime copas não anotadas. "
                        "Ignorado se --reg_include_bg True.")

    # ── Pseudo-rotulagem online (self-training) ───────────────
    p.add_argument('--pseudo_teacher', type=str, default=None,
                   help="Caminho de um checkpoint de regressão treinado (PROFESSOR). "
                        "Se fornecido, ativa a pseudo-rotulagem online: a cada batch "
                        "o professor detecta copas de alta confiança no fundo NÃO "
                        "anotado e as injeta no alvo com peso reduzido. Deixe vazio "
                        "para desativar. Use o fluxo: treine 1 modelo normalmente, "
                        "depois treine outro passando esse checkpoint aqui.")
    p.add_argument('--pseudo_conf', type=float, default=0.5,
                   help="Confiança mínima (valor do mapa de distância predito, [0,1]) "
                        "para um pico virar pseudo-copa. IMPORTANTE p/ tamanhos diversos: "
                        "copas GRANDES têm pico predito mais BAIXO/espalhado, então um "
                        "limiar alto (0.5) filtra justamente as grandes. Default 0.35 "
                        "deixa copas grandes passarem. Suba se injetar lixo; baixe "
                        "(0.25–0.3) se ainda perder copas grandes.")
    p.add_argument('--pseudo_weight', type=float, default=0.75,
                   help="Peso dos pseudo-rótulos na loss (copas reais = 1.0). <1 "
                        "garante que a verdade real domine o gradiente. Default 0.3.")
    p.add_argument('--pseudo_sigma', type=float, default=1.25,
                   help="Sigma MÍNIMO (px) da gaussiana plantada (tamanho da menor "
                        "pseudo-copa). O tamanho real é ADAPTATIVO: copas maiores "
                        "ganham sigma maior (ver --pseudo_sigma_scale).")
    p.add_argument('--pseudo_sigma_scale', type=float, default=1.0,
                   help="Quanto a pseudo-copa pode CRESCER conforme o tamanho estimado "
                        "da copa. sigma final ∈ [pseudo_sigma, pseudo_sigma*(1+scale)]. "
                        "Default 2.0 → copas grandes ganham até 3x o sigma mínimo. "
                        "0 = tamanho fixo (comportamento antigo).")
    p.add_argument('--pseudo_peak_radii', type=int, nargs='+', default=[2, 4, 8],
                   help="Raios (px) da detecção de picos MULTI-ESCALA. Vários raios "
                        "capturam copas de tamanhos diferentes (pequenas em janelas "
                        "pequenas, grandes em janelas grandes). Default [2 4 8]. "
                        "Era a causa de só pegar copas pequenas (raio único).")
    
    
    p.add_argument('--learn_task_weights', type=_as_bool, default=True,
                   help='Use homoscedastic uncertainty to learn task weights automatically')
    p.add_argument('--label_smoothing', type=float, default=0.01,
                   help='Label smoothing for CE loss')
    p.add_argument('--class_weights', type=str, default='inverse',
                   choices=['none', 'inverse', 'median', 'effective', 'manual'],
                   help='Pondera a CrossEntropy para mitigar desbalanceamento de '
                        'classes. "inverse"=freq. inversa; "median"=median frequency '
                        'balancing; "effective"=effective number of samples (Cui 2019); '
                        '"manual"=usa --class_weights_manual; "none"=sem pesos (padrão). '
                        'Os pesos são calculados a partir do TRAIN e normalizados p/ média 1.')
    p.add_argument('--class_weights_manual', type=float, nargs='+', default=None,
                   help='Lista de num_classes pesos (usado só com --class_weights manual). '
                        'Ex.: --class_weights_manual 1.0 2.5 3.0 1.2')

    # Training hyper-parameters
    p.add_argument('--epochs',       type=int,   default=50)
    p.add_argument('--batch_size',   type=int,   default=100)
    p.add_argument('--base_lr',      type=float, default=0.01)
    p.add_argument('--lr',           type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--warmup_epochs',type=int,   default=5)
    p.add_argument('--num_workers',  type=int,   default=5)
    p.add_argument('--prefetch_factor', type=int, default=4,
                   help="Batches pré-carregados por worker (só com num_workers>0). "
                        "Maior = mais leitura antecipada, suaviza engasgos de I/O.")
    p.add_argument('--cudnn_benchmark', type=_as_bool, default=False,
                   help="Ativa cudnn.benchmark (patch de tamanho fixo): o cuDNN "
                        "escolhe os kernels de convolução mais rápidos após a 1ª "
                        "iteração — acelera o treino, mas perde o determinismo exato. "
                        "Default False (reprodutível).")
    p.add_argument('--amp',          action='store_true',
                   help='Use automatic mixed precision (CUDA only)')
    p.add_argument('--seed',         type=int,   default=42)

    # Logging
    p.add_argument('--plot_every', type=int, default=1,
                   help='Save prediction figures every N epochs')
    p.add_argument('--save_every', type=int, default=25,
                   help='Save periodic weight checkpoint every N epochs')

    # ── Resume ────────────────────────────────────────────────
    def _str2bool(v):
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in ('1', 'true', 't', 'yes', 'y', 'sim', 's')

    p.add_argument('--resume', type=_str2bool, nargs='?', const=True, default=True,
                   help='Se True (padrão), retoma automaticamente o treino a partir '
                        'de weights/best_model.pth (ou last_model.pth) se existir na '
                        'pasta do modelo. Use --resume False para treinar do zero.')
    p.add_argument('--resume_path', type=str, default='',
                   help='Caminho explícito de um checkpoint para retomar. Se vazio, '
                        'procura best_model.pth / last_model.pth em <model_dir>/weights.')

    return p.parse_args()


# ─────────────────────────────────────────────────────────────
# 11.  Entry point
# ─────────────────────────────────────────────────────────────
def _resolve_task(args):
    """
    Resolve as três flags booleanas em um único args.task canônico:
    'multitask' | 'classification' | 'regression'.

    Regra: se múltiplas forem True, prioridade multitask > classification >
    regression; se nenhuma, assume multitask. Define também args.multitask
    (bool) para retrocompatibilidade com código que ainda o consulte.
    """
    if getattr(args, 'task_multitask', False):
        task = 'multitask'
    elif getattr(args, 'task_classification', False):
        task = 'classification'
    elif getattr(args, 'task_regression', False):
        task = 'regression'
    else:
        task = 'multitask'
    args.task = task
    args.multitask = (task == 'multitask')
    return args


if __name__ == '__main__':
    args = get_args()
    args = _resolve_task(args)

    print("=" * 60)
    print(f"Mangrove Species Mapping – {args.task.capitalize()} Training")
    print("=" * 60)
    print(f"  >>> TASK: {args.task.upper()} <<<")
    for k, v in vars(args).items():
        print(f"  {k:25s}: {v}")
    print("=" * 60)

    train(args)