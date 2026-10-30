"""
Inference Script — Mangrove Species Mapping
============================================
Produz dois rasters georreferenciados a partir de uma imagem grande:
  • pred_class.tif   — mapa de classes (Byte)
  • pred_prob.tif    — probabilidade máxima por pixel (Float32)
  • pred_dist.tif    — mapa de distância previsto (Float32)

Estratégia de predição:
  Para cada overlap em [0.3, 0.5, 0.7] roda uma passagem de janela
  deslizante sobre a imagem inteira. A predição final é a média das
  três passagens — reduz artefatos de borda e aumenta a estabilidade.

Modo LateFusion (--latefusion):
  Quando ativado, o modelo tem dois braços (RGB + CHM/LiDAR). Para cada
  janela RGB, o tile correspondente do CHM é lido em RESOLUÇÃO NATIVA,
  alinhado por georreferência (não por divisão de índices), exatamente
  como no dataloader_construction.py. O CHM é redimensionado internamente
  pela rede apenas no momento da fusão. Áreas sem cobertura de CHM recebem
  zeros nesse braço (a predição continua válida pelo braço RGB).

Eficiência de memória:
  • A imagem de entrada é lida banda a banda via GDAL (nunca carregada
    inteira em RAM).
  • Os acumuladores de predição são float16 em CPU — tipicamente
    ~2 × rows × cols × n_classes bytes.
  • Cada tile é processado em mini-batches na GPU; o acumulador vive
    em CPU e recebe os resultados incrementalmente.

Uso:
  python predict.py \
      --image_path  <ortofoto.tif> \
      --checkpoint  <exp/checkpoint.pth.tar> \
      --out_dir     <pasta/saida> \
      --arch        BPATUNet \
      --n_classes   4 \
      --patch_size  128 \
      --batch_size  32 \
      --overlaps    0.3 0.5 0.7
"""

import os


    
import sys
import argparse
import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from osgeo import gdal, osr
from tqdm import tqdm

# Removido os.chdir hard-coded do ambiente do professor (E:\MANGROVE_...).
# Os caminhos vêm dos argumentos CLI / wrappers deste experimento.

import segmentation_models_pytorch as smp

# ── suppress GDAL warnings ────────────────────────────────────
for _v in ('PROJ_LIB', 'PROJ_DATA', 'GDAL_DATA'):
    os.environ.pop(_v, None)
gdal.UseExceptions()
gdal.PushErrorHandler('CPLQuietErrorHandler')


# ═════════════════════════════════════════════════════════════
# 1.  Arquitecturas — cópia fiel do train_CLAUDE.py
# ═════════════════════════════════════════════════════════════

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
    """ATENÇÃO: atributo é 'net', não 'double_conv' — deve ser igual ao treino."""
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

class MultitaskHead(nn.Module):
    """
    Wrapper multitask — idêntico ao train_CLAUDE.py.
    Recebe features do backbone e produz (dist_pred, seg_logits).
    O sigmoid já é aplicado INTERNAMENTE no reg_head — não aplicar novamente
    na inferência.
    """
    def __init__(self, feature_extractor: nn.Module,
                 dec_out_channels: int, num_classes: int,
                 dist_guided: bool = False, task: str = 'multitask'):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.dist_guided       = dist_guided
        self.task              = task

        if task in ('multitask', 'regression'):
            reg_hidden = max(32, dec_out_channels // 2)
            self.reg_head = nn.Sequential(
                nn.Conv2d(dec_out_channels, reg_hidden, 3, padding=1),
                nn.BatchNorm2d(reg_hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(reg_hidden, 1, 1),
            )
            nn.init.constant_(self.reg_head[-1].bias, -4.0)

        if task in ('multitask', 'classification'):
            clf_hidden = max(32, dec_out_channels)
            clf_in = dec_out_channels + 1 if task == 'multitask' else dec_out_channels
            self.clf_head = nn.Sequential(
                nn.Conv2d(clf_in, clf_hidden, 3, padding=1),
                nn.BatchNorm2d(clf_hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(clf_hidden, num_classes, 1),
            )

    def forward(self, x):
        feat = self.feature_extractor(x)

        if self.task == 'regression':
            return torch.sigmoid(self.reg_head(feat))
        if self.task == 'classification':
            return self.clf_head(feat)

        # multitask
        dist_pred = torch.sigmoid(self.reg_head(feat))
        if self.dist_guided:
            attn        = dist_pred
            feat_guided = feat * attn
            clf_input   = torch.cat([feat_guided, attn], dim=1)
        else:
            clf_input   = torch.cat([feat, dist_pred], dim=1)
        seg_logits = self.clf_head(clf_input)
        return dist_pred, seg_logits


# ═════════════════════════════════════════════════════════════
# 1b.  LateFusion (RGB + CHM/LiDAR) — cópia fiel do train_CLAUDE.py
# ═════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Bloco residual full pre-activation (igual ao train_CLAUDE.py)."""
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
    """Encoder ResU-Net com skip-connections (igual ao train_CLAUDE.py)."""
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
    """Decoder ResU-Net com skip-connections (igual ao train_CLAUDE.py)."""
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
    """ResU-Net de braço único (igual ao train_CLAUDE.py)."""
    def __init__(self, input_depth, depths, out_ch):
        super().__init__()
        self.encoder = ResUnetEncoder(input_depth, depths)
        self.decoder = ResUnetDecoder(depths)
        self.out_conv = nn.Conv2d(depths[0], out_ch, kernel_size=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.out_conv(x)


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
                 task: str = 'multitask', fusion: str = 'concat'):
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
            nn.init.constant_(self.reg_head[-1].bias, -4.0)

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

# ═════════════════════════════════════════════════════════════
# 2.  Função de construção do modelo — idêntica ao train_CLAUDE.py
# ═════════════════════════════════════════════════════════════

def _safe_depth(side, cap=4, min_bottleneck=4):
    """Profundidade de downsampling segura: maior d com side/2**d >= min_bottleneck.
    IDÊNTICA à do treino — assegura que o predict reconstrua os mesmos braços
    (em especial o braço LiDAR com depth reduzido quando o patch CHM é pequeno)."""
    d = 0
    s = side
    while s % 2 == 0 and s // 2 >= min_bottleneck and d < cap:
        s //= 2
        d += 1
    return max(1, d)


def build_fusion_backbone(arch: str, in_channels: int, out_ch: int,
                          depth: int = 4) -> nn.Module:
    """
    Constrói UM braço da LateFusion. `depth` só afeta a BPATUNet (nº de
    downsamplings) e DEVE casar com o do treino p/ os pesos carregarem.
    `encoder_weights=None` porque os pesos vêm do checkpoint.
    """
    SMP_COMMON = dict(
        encoder_name    = "resnet50",
        encoder_weights = None,
        in_channels     = in_channels,
        activation      = None,
        encoder_depth   = 4,
    )
    if arch == "unet":
        return smp.Unet(**SMP_COMMON, classes=out_ch,
                        decoder_channels=(512, 256, 128, 64),
                        decoder_use_batchnorm=True, decoder_attention_type="scse")
    elif arch == "unetplusplus":
        return smp.UnetPlusPlus(**SMP_COMMON, classes=out_ch,
                                decoder_channels=(512, 256, 128, 64),
                                decoder_use_batchnorm=True, decoder_attention_type="scse")
    elif arch == "manet":
        return smp.MAnet(**SMP_COMMON, classes=out_ch,
                         decoder_channels=(512, 256, 128, 64),
                         decoder_use_batchnorm=True, decoder_pab_channels=64)
    elif arch == "linknet":
        return smp.Linknet(**SMP_COMMON, classes=out_ch,
                           decoder_channels=(256, 128, 64, 32))
    elif arch == "BPATUNet":
        return BPATUNet(in_channels=in_channels, n_classes=out_ch,
                        base_channels=64, depth=depth)
    elif arch == "resunet":
        return ResUnetBackbone(in_channels, [64, 128, 256, 512], out_ch)
    else:
        sys.exit(f"Arquitetura desconhecida para LateFusion: {arch}")


def build_model(args, n_channels: int, n_classes: int) -> nn.Module:
    """
    Reconstrói o modelo com a mesma arquitetura do treino, segundo args.task
    ∈ {multitask, regression, classification}. Os pesos são carregados depois.

    LateFusion (--latefusion):
      dois backbones (RGB + CHM) → LateFusionGeneric(task)
    Single-source:
      backbone (64 features) → MultitaskHead(task)
    (deeplabv3_resnet50 e resunet mantêm a interface dict {'out','aux'} e só
     fazem sentido para multitask — checkpoints legados.)
    """
    feature_channels = 64
    arch = args.arch
    task = args.task

    # ── LateFusion (RGB + CHM/LiDAR) ──────────────────────────
    if getattr(args, 'latefusion', False):
        # Profundidade adaptativa por braço — IDÊNTICA ao treino. O patch CHM
        # nativo é menor que o RGB (scale = chm_res/rgb_res); o braço LiDAR usa
        # menos downsamplings p/ não degenerar o bottleneck (BPATUNet).
        scale = int(round(getattr(args, 'chm_resolution', 1.0) /
                          getattr(args, 'rgb_resolution', 0.25)))
        chm_patch  = getattr(args, 'patch_size', 64) // max(1, scale)
        rgb_depth   = _safe_depth(getattr(args, 'patch_size', 64), cap=4)
        lidar_depth = _safe_depth(chm_patch, cap=4)
        fusion = getattr(args, 'fusion', 'concat')

        if arch == 'BPATUNet' and lidar_depth < 4:
            print(f"  ℹ️  Braço LiDAR (BPATUNet): patch CHM = {chm_patch}px → "
                  f"depth={lidar_depth} (RGB depth={rgb_depth}).")

        backbone_rgb   = build_fusion_backbone(arch, n_channels,          feature_channels,
                                               depth=rgb_depth)
        backbone_lidar = build_fusion_backbone(arch, args.lidar_channels, feature_channels,
                                               depth=lidar_depth)
        model = LateFusionGeneric(
            backbone_rgb   = backbone_rgb,
            backbone_lidar = backbone_lidar,
            feat_channels  = feature_channels,
            num_classes    = n_classes,
            dist_guided    = args.dist_guided,
            task           = task,
            fusion         = fusion,
        )
        extra = f"| clf {n_classes} classes" if task != 'regression' else "| regressão"
        print(f"  [Model] LateFusion ({fusion}) {task} | arch={arch} | "
              f"RGB({n_channels}ch) + CHM({args.lidar_channels}ch) {extra}")
        return model

    # ── Single-source de checkpoint do train_fusion_..._on_the_fly.py ──
    # Esse treino usa no single-source o MESMO backbone da LateFusion
    # (build_backbone: resnet50, encoder_depth=4, decoder 512-256-128-64;
    # BPATUNet base 64 / depth 4; ResUnetBackbone) e não salva encoder_depth/
    # backbone no checkpoint. O caminho legado abaixo (train_CLAUDE.py) usa
    # --encoder_depth/--backbone do CLI e não casa com esses pesos.
    if getattr(args, 'ckpt_treino_atual', False) and arch != "deeplabv3_resnet50":
        backbone = build_fusion_backbone(arch, n_channels, feature_channels)
        model = MultitaskHead(
            feature_extractor = backbone,
            dec_out_channels  = feature_channels,
            num_classes       = n_classes,
            dist_guided       = args.dist_guided,
            task              = task,
        )
        extra = f"| clf {n_classes} classes" if task != 'regression' else "| regressão"
        print(f"  [Model] Single-source {task} | arch={arch} {extra} "
              f"| backbone idêntico ao do treino")
        return model

    # ── Legado (train_CLAUDE.py): decoder_channels para smp ──
    _dec_base = [512, 256, 128, 64, 32]
    decoder_ch = tuple(_dec_base[:args.encoder_depth][::-1]
                       if arch == 'linknet'
                       else _dec_base[:args.encoder_depth])

    SMP_COMMON = dict(
        encoder_name    = args.backbone,
        encoder_weights = None,             # pesos carregados do checkpoint
        in_channels     = n_channels,
        activation      = None,
        encoder_depth   = args.encoder_depth,
        classes         = feature_channels,
    )

    if arch == "unet":
        backbone = smp.Unet(**SMP_COMMON, decoder_channels=decoder_ch,
                            decoder_use_batchnorm=True, decoder_attention_type="scse")
    elif arch == "unetplusplus":
        backbone = smp.UnetPlusPlus(**SMP_COMMON, decoder_channels=decoder_ch,
                                    decoder_use_batchnorm=True, decoder_attention_type="scse")
    elif arch == "manet":
        backbone = smp.MAnet(**SMP_COMMON, decoder_channels=decoder_ch,
                             decoder_use_batchnorm=True, decoder_pab_channels=64)
    elif arch == "linknet":
        backbone = smp.Linknet(**SMP_COMMON, decoder_channels=decoder_ch)
    elif arch == "BPATUNet":
        backbone = BPATUNet(in_channels=n_channels, n_classes=feature_channels,
                            base_channels=args.base_channels,
                            depth=args.encoder_depth)
    elif arch == "deeplabv3_resnet50":
        import torchvision.models.segmentation as seg_models
        model = seg_models.deeplabv3_resnet50(weights=None, aux_loss=True)
        model.backbone.conv1 = nn.Conv2d(
            n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.aux_classifier[4] = nn.Conv2d(256, 1,         1)
        model.classifier[4]     = nn.Conv2d(256, n_classes,  1)
        return model   # mantém interface dict {'out','aux'} (legado multitask)
    elif arch == "resunet":
        from src.resnet import ResUnet
        class _ResUnetWrapper(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base
                self.dist_head = nn.Conv2d(args.filters[-1], 1, 1)
            def forward(self, x):
                out = self.base(x)
                return {'out': out, 'aux': self.dist_head(out)}
        return _ResUnetWrapper(ResUnet(channel=n_channels, nb_classes=n_classes,
                                       filters=args.filters))
    else:
        sys.exit(f"Arquitetura desconhecida: {arch}")

    # ── Single-source: backbone (64 features) → MultitaskHead(task) ──
    model = MultitaskHead(
        feature_extractor = backbone,
        dec_out_channels  = feature_channels,
        num_classes       = n_classes,
        dist_guided       = args.dist_guided,
        task              = task,
    )
    extra = f"| clf {n_classes} classes" if task != 'regression' else "| regressão"
    print(f"  [Model] Single-source {task} | arch={arch} {extra}")
    return model


# ═════════════════════════════════════════════════════════════
# 3.  Leitura eficiente da imagem grande
# ═════════════════════════════════════════════════════════════

class LazyTiffReader:
    """
    Lê a imagem grande tile a tile — nunca a carrega inteira em RAM.

    Usa rasterio para ler metadados (preserva CRS, transform e todos os
    metadados corretamente — igual ao labeled_dataset_construction.py).
    Usa GDAL para leitura de tiles (mais eficiente para janelas arbitrárias).

    Pickle-safe para Windows (multiprocessing spawn):
    guarda apenas o path; reabre arquivos sob demanda.
    """
    def __init__(self, path: str, norm: bool = True, rgb_norm: str = 'div255',
                 rgb_mean=None, rgb_std=None, rgb_min=None, rgb_max=None,
                 rgb_norm_const=255.0):
        self.path = path
        self.norm = norm
        # mesma normalização do treino (restaurada do checkpoint). NaN → 0 sempre.
        #   'none'     → valores como estão
        #   'minmax'   → (x - min)/(max - min) por banda, clip [0,1]
        #   'constant' → x / rgb_norm_const
        #   'zscore'   → (x - mean)/std por banda   (checkpoints de 09/10/2026)
        #   'div255'   → x / 255                    (checkpoints antigos)
        self.rgb_norm = rgb_norm or 'div255'
        self.rgb_norm_const = float(rgb_norm_const or 255.0)

        def _band(v):
            return (np.asarray(v, dtype=np.float32).reshape(-1, 1, 1)
                    if v is not None else None)
        self.rgb_mean, self.rgb_std = _band(rgb_mean), _band(rgb_std)
        self.rgb_min,  self.rgb_max = _band(rgb_min),  _band(rgb_max)
        need = {'zscore': (self.rgb_mean, self.rgb_std),
                'minmax': (self.rgb_min, self.rgb_max)}.get(self.rgb_norm, ())
        if any(v is None for v in need):
            raise ValueError(f"rgb_norm='{self.rgb_norm}' exige as estatísticas "
                             f"por banda salvas no checkpoint.")
        if self.rgb_norm not in ('none', 'minmax', 'constant', 'zscore', 'div255'):
            raise ValueError(f"rgb_norm desconhecido no checkpoint: {self.rgb_norm}")
        # Lê metadados via rasterio — preserva CRS e transform fielmente
        import rasterio as rio
        with rio.open(path) as src:
            self.bands  = src.count
            self.rows   = src.height
            self.cols   = src.width
            self._meta  = src.meta.copy()   # contém crs, transform, driver, dtype…

    def read_tile(self, r: int, c: int, h: int, w: int) -> np.ndarray:
        """Lê um tile (r,c) de tamanho (h,w) e retorna (C,H,W) float32.
        Reabre via GDAL a cada chamada — seguro para multiprocessing.
        """
        ds  = gdal.Open(self.path)
        out = np.zeros((self.bands, h, w), dtype=np.float32)
        for b in range(self.bands):
            data = ds.GetRasterBand(b + 1).ReadAsArray(c, r, w, h)
            if data is None:
                data = np.full((h, w), np.nan, dtype=np.float32)   # → 0 abaixo
            out[b] = data.astype(np.float32)
        ds = None   # fecha
        if self.norm:
            if self.rgb_norm == 'minmax':
                out = np.clip((out - self.rgb_min)
                              / np.maximum(self.rgb_max - self.rgb_min, 1e-12), 0.0, 1.0)
            elif self.rgb_norm == 'constant':
                out /= self.rgb_norm_const
            elif self.rgb_norm == 'zscore':
                out = (out - self.rgb_mean) / self.rgb_std
            elif self.rgb_norm == 'div255':
                out /= 255.0
        # NaN (nodata) → 0, como no treino; senão a rede devolve NaN no tile todo
        np.nan_to_num(out, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        return out

    @property
    def meta(self):
        """Metadados rasterio completos — use para salvar rasters de saída."""
        return self._meta


def _poly2(r, c):
    """Cópia fiel do treino: termos do polinômio de 2º grau em (row, col)."""
    r = np.asarray(r, dtype=np.float64) * 1e-4
    c = np.asarray(c, dtype=np.float64) * 1e-4
    return np.stack([np.ones_like(r), r, c, r * r, r * c, c * c], axis=-1)


def _rgb_to_chm_map(rgb_tf, rgb_crs, chm_tf, chm_crs, H, W, n=21):
    """Cópia fiel do treino: mapeamento (row, col) do grid RGB → (row, col)
    fracionário do grid CHM. Se os CRS diferem (ex.: RGB em graus WGS84 e CHM
    em UTM), reprojeta uma grade n×n de pontos de controle e ajusta um
    polinômio de 2º grau. Retorna (M (2, 6), erro_max_px)."""
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


class LazyChmReader:
    """
    Lê tiles do CHM (LiDAR) em RESOLUÇÃO NATIVA, alinhados a janelas da imagem RGB
    via GEORREFERÊNCIA (não por divisão de índices) — mesma lógica do treino.

    Dado um tile RGB no pixel (r, c), o canto vai para o grid do CHM pelo afim
    georreferenciado (CRS-aware, ver _rgb_to_chm_map) e lê-se uma janela de
    `chm_patch_size = patch_size // round(chm_res / rgb_res)` px — MESMA fórmula
    e arredondamento do treino.

    Se a janela cair (parcial ou totalmente) fora da cobertura do CHM, devolve um
    patch preenchido com zeros do tamanho correto — assim a predição não quebra
    em áreas sem LiDAR (o braço CHM simplesmente recebe zeros ali).
    """
    def __init__(self, chm_path: str, rgb_meta: dict,
                 patch_size: int, rgb_resolution: float, chm_resolution: float,
                 norm: bool = True, lidar_channels: int = 1, norm_values=None,
                 lidar_norm=None, lidar_min=None, lidar_max=None):
        import rasterio as rio
        self.path = chm_path
        self.norm = norm
        self.patch_size = patch_size
        self.lidar_channels = int(lidar_channels)
        # lidar_norm (do checkpoint): 'minmax' → (x-min)/(max-min) com os min/max
        # do TREINO; 'values'/None (ckpt antigo) → tetos norm_values (legado)
        self.lidar_norm = lidar_norm
        self.lidar_min = self.lidar_max = None
        if lidar_norm == 'minmax':
            if lidar_min is None or lidar_max is None:
                raise ValueError("lidar_norm='minmax' exige lidar_min/lidar_max "
                                 "(vêm do checkpoint).")
            self.lidar_min = np.asarray(lidar_min, dtype=np.float32).reshape(-1, 1, 1)
            self.lidar_max = np.asarray(lidar_max, dtype=np.float32).reshape(-1, 1, 1)
        self.norm_values = (np.asarray(norm_values, dtype=np.float32)
                            if norm_values is not None and lidar_norm != 'minmax'
                            else None)
        if self.norm_values is not None and \
           self.norm_values.shape[0] != self.lidar_channels:
            raise ValueError(
                f"--norm_values tem {self.norm_values.shape[0]} valores, mas "
                f"--lidar_channels={self.lidar_channels}.")

        scale = max(1, int(round(chm_resolution / rgb_resolution)))
        self.chm_patch_size = max(1, patch_size // scale)

        with rio.open(chm_path) as src:
            self.chm_rows = src.height
            self.chm_cols = src.width
            chm_tf, chm_crs = src.transform, src.crs
            n_bands = src.count
        # banda a banda, o raster precisa casar com o treino (senão canais
        # faltantes viram zero ou os min/max são aplicados às bandas erradas)
        if n_bands < self.lidar_channels:
            raise ValueError(f"[CHM] raster tem {n_bands} bandas, mas o modelo foi "
                             f"treinado com lidar_channels={self.lidar_channels}.")
        if self.lidar_min is not None and self.lidar_min.shape[0] != n_bands:
            raise ValueError(f"[CHM] raster tem {n_bands} bandas, mas o checkpoint "
                             f"tem min/max de {self.lidar_min.shape[0]} bandas "
                             f"(raster diferente do treino?).")
        rgb_crs = rgb_meta.get('crs', None)
        self.M, err = _rgb_to_chm_map(rgb_meta['transform'], rgb_crs, chm_tf, chm_crs,
                                      rgb_meta['height'], rgb_meta['width'])
        if rgb_crs is not None and chm_crs is not None and rgb_crs != chm_crs:
            print(f"  [CHM] CRS diferente do RGB → reprojeção da localização "
                  f"(erro máx. do ajuste: {err:.4f} px do CHM)")
        # handle GDAL aberto uma vez por processo: reabrir a cada tile descarta o
        # cache de blocos (com 128 bandas, cada bloco de 512² descomprime ~134 MB)
        self._ds = None

        print(f"  [CHM] {self.chm_rows}x{self.chm_cols} px | scale={scale} "
              f"| chm_patch={self.chm_patch_size}px | norm={lidar_norm or 'legado'}")

    def __getstate__(self):
        # o handle GDAL não é serializável; cada worker reabre o seu
        state = self.__dict__.copy()
        state['_ds'] = None
        return state

    def read_tile(self, r: int, c: int) -> np.ndarray:
        """
        Lê o tile CHM/LiDAR correspondente ao tile RGB que começa em (r, c).
        Retorna (C, chm_patch_size, chm_patch_size) float32, com C = lidar_channels.
        """
        cps = self.chm_patch_size
        C = self.lidar_channels
        out = np.zeros((C, cps, cps), dtype=np.float32)

        # canto do tile RGB → índice no grid CHM (mesmo mapeamento/arredondamento
        # do treino)
        rc_f, cc_f = self.M @ _poly2(r, c)
        crow, ccol = int(round(rc_f)), int(round(cc_f))

        # interseção da janela [crow, crow+cps) × [ccol, ccol+cps) com o raster CHM
        r0 = max(crow, 0)
        c0 = max(ccol, 0)
        r1 = min(crow + cps, self.chm_rows)
        c1 = min(ccol + cps, self.chm_cols)
        if r1 <= r0 or c1 <= c0:
            return out   # totalmente fora da cobertura → zeros

        if self._ds is None:
            self._ds = gdal.Open(self.path)
        dr = r0 - crow
        dc = c0 - ccol
        # Lê TODAS as bandas da janela de uma vez → (bandas, h, w)
        data = self._ds.ReadAsArray(c0, r0, c1 - c0, r1 - r0)
        if data is None:
            return out
        if data.ndim == 2:
            data = data[None]
        data = data[:C].astype(np.float32)

        if self.lidar_norm == 'minmax':
            # idêntico ao treino: min-max com as estatísticas do treino, clip
            # [0,1], NaN → 0 e a mesma precisão float16 em que o treino guarda
            nb = data.shape[0]
            data = (data - self.lidar_min[:nb]) / np.maximum(
                self.lidar_max[:nb] - self.lidar_min[:nb], 1e-12)
            data = np.clip(data, 0.0, 1.0)
            np.nan_to_num(data, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
            data = data.astype(np.float16).astype(np.float32)
            out[:nb, dr:dr + (r1 - r0), dc:dc + (c1 - c0)] = data
            return out

        data = np.nan_to_num(data, nan=0.0)
        out[:data.shape[0], dr:dr + (r1 - r0), dc:dc + (c1 - c0)] = data

        if self.norm:
            if self.norm_values is not None:
                # Normalização POR BANDA: divide cada banda pelo teto fixo e clampa
                # em [0, 1]. IDÊNTICA à do treino → preserva a escala física e a
                # consistência entre tiles (evita o problema do "máximo por patch").
                nv = self.norm_values.reshape(C, 1, 1)
                out = out / (nv + 1e-8)
                out = np.clip(out, 0.0, 1.0)
            else:
                # Comportamento legado (banda única): divide pelo máximo do patch.
                lmax = float(out.max())
                if lmax > 1.0:
                    out /= (lmax + 1e-8)
        return out


# ═════════════════════════════════════════════════════════════
# 4.  Dataset de tiles para um único overlap
# ═════════════════════════════════════════════════════════════

class SlidingWindowDataset(Dataset):
    """
    Enumera todos os patches de uma imagem com um dado overlap.
    Não carrega nada em RAM — cada __getitem__ lê via LazyTiffReader.
    Retorna (patch_float32, row_start, col_start).
    """
    def __init__(self, reader: LazyTiffReader, patch_size: int, overlap: float,
                 chm_reader=None, mask=None):
        self.reader     = reader
        self.chm_reader = chm_reader          # LazyChmReader ou None
        self.mask       = mask                # MangroveMask ou None
        self.patch_size = patch_size
        p               = patch_size
        stride          = max(1, int(round(p * (1 - overlap))))
        self.stride     = stride

        rows, cols = reader.rows, reader.cols

        # coordenadas de todos os patches (top-left corner)
        r_starts = list(range(0, rows - p + 1, stride))
        c_starts = list(range(0, cols - p + 1, stride))

        # garante que o último patch chega até a borda
        if not r_starts or r_starts[-1] + p < rows:
            r_starts.append(max(0, rows - p))
        if not c_starts or c_starts[-1] + p < cols:
            c_starts.append(max(0, cols - p))

        # remove duplicatas mantendo ordem
        r_starts = sorted(set(r_starts))
        c_starts = sorted(set(c_starts))

        all_coords = [(r, c) for r in r_starts for c in c_starts]

        # ── Filtro por máscara: mantém só tiles que tocam o alvo (mangue) ──
        # Pular tiles 100% fora do mangue economiza inferência e evita que
        # contexto irrelevante (água/não-mangue) contamine as bordas via overlap.
        if self.mask is not None:
            kept = [(r, c) for (r, c) in all_coords
                    if self.mask.tile_has_target(r, c, p, p)]
            print(f"  [máscara] tiles mantidos: {len(kept)}/{len(all_coords)} "
                  f"(pulados {len(all_coords) - len(kept)} fora do mangue)")
            self.coords = kept
        else:
            self.coords = all_coords

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        r, c        = self.coords[idx]
        p           = self.patch_size
        tile        = self.reader.read_tile(r, c, p, p)   # (C, H, W)
        if self.chm_reader is not None:
            chm_tile = self.chm_reader.read_tile(r, c)    # (1, h, w) nativo
            return torch.from_numpy(tile), torch.from_numpy(chm_tile), r, c
        return torch.from_numpy(tile), r, c


# ═════════════════════════════════════════════════════════════
# 5b. Máscara de mangue (foca a predição nos pixels da classe-alvo)
# ═════════════════════════════════════════════════════════════

class MangroveMask:
    """
    Carrega um raster de classes (ex.: pred_class.tif com 1=mangue,
    2=não-mangue, 3=água) e produz uma máscara booleana dos pixels que
    pertencem à(s) classe(s)-alvo (por padrão, classe 1 = mangue).

    Usada para:
      (a) PULAR tiles que não tocam nenhum pixel-alvo (economia + evita
          contaminar bordas com contexto irrelevante na janela deslizante);
      (b) CORTAR a predição final, zerando tudo fora do mangue.

    O raster da máscara DEVE estar alinhado ao grid da imagem de entrada
    (mesmas linhas/colunas e mesmo transform) — é o caso aqui, pois o
    pred_class.tif foi gerado a partir da mesma imagem RGB.
    """
    def __init__(self, mask_path: str, expected_shape, target_values=(1,)):
        import rasterio as rio
        self.path = mask_path
        self.target_values = tuple(int(v) for v in target_values)
        with rio.open(mask_path) as src:
            arr = src.read(1)
            self.mask_rows, self.mask_cols = arr.shape
        exp_r, exp_c = expected_shape
        if (self.mask_rows, self.mask_cols) != (exp_r, exp_c):
            raise ValueError(
                f"Máscara {mask_path} tem shape {(self.mask_rows, self.mask_cols)} "
                f"mas a imagem de entrada tem {(exp_r, exp_c)}. A máscara precisa "
                f"estar no MESMO grid/recorte da imagem (mesmas dimensões). "
                f"Reprojete/recorte a máscara para casar com a imagem.")
        # máscara booleana dos pixels-alvo (True = mangue)
        self.mask = np.isin(arr, self.target_values)
        n_target = int(self.mask.sum())
        pct = 100.0 * n_target / self.mask.size if self.mask.size else 0.0
        print(f"  [máscara] {os.path.basename(mask_path)} | alvo={self.target_values} "
              f"| {n_target} px ({pct:.1f}%) dentro do mangue")
        if n_target == 0:
            print("  ⚠️  [máscara] NENHUM pixel-alvo encontrado! Verifique "
                  "--mask_target_values e o conteúdo do raster de máscara.")

    def tile_has_target(self, r: int, c: int, h: int, w: int) -> bool:
        """True se o tile (r,c,h,w) contém ao menos 1 pixel-alvo."""
        sub = self.mask[r:r+h, c:c+w]
        return bool(sub.any())

    def apply(self, array: np.ndarray):
        """Zera tudo fora da máscara. Aceita (H,W) ou (H,W,bandas)."""
        if array.ndim == 2:
            out = array.copy()
            out[~self.mask] = 0
            return out
        out = array.copy()
        out[~self.mask, ...] = 0
        return out


# ═════════════════════════════════════════════════════════════
# 5c. Motor de predição — acumulação multi-overlap
# ═════════════════════════════════════════════════════════════

def predict_one_overlap(
    model:      nn.Module,
    reader:     LazyTiffReader,
    patch_size: int,
    overlap:    float,
    n_classes:  int,
    batch_size: int,
    device:     torch.device,
    desc:       str,
    task:       str = 'multitask',
    chm_reader=None,
    mask=None,
) -> tuple:
    """
    Roda uma passagem completa com um dado overlap.

    Se `chm_reader` for fornecido, o modelo é LateFusion e recebe [rgb, chm].

    Conforme `task`:
      • multitask      → acumula segmentação E distância
      • classification → acumula apenas segmentação (acc_dist fica None)
      • regression     → acumula apenas distância    (acc_seg  fica None)

    Retorna (acc_seg, acc_dist) — qualquer um pode ser None se não se aplica.
    """
    rows, cols = reader.rows, reader.cols
    do_seg = task in ('multitask', 'classification')
    do_reg = task in ('multitask', 'regression')

    acc_seg  = np.zeros((rows, cols, n_classes), dtype=np.float32) if do_seg else None
    acc_dist = np.zeros((rows, cols),            dtype=np.float32) if do_reg else None
    count    = np.zeros((rows, cols),            dtype=np.float32)

    dataset = SlidingWindowDataset(reader, patch_size, overlap,
                                   chm_reader=chm_reader, mask=mask)
    loader  = DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = False,
        drop_last   = False,
    )

    fusion = chm_reader is not None

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            if fusion:
                tiles, chm_tiles, r_batch, c_batch = batch
                tiles     = tiles.to(device)
                chm_tiles = chm_tiles.to(device)
                model_input = [tiles, chm_tiles]
                n_in_batch  = tiles.size(0)
            else:
                tiles, r_batch, c_batch = batch
                tiles = tiles.to(device)
                model_input = tiles
                n_in_batch  = tiles.size(0)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                out = model(model_input)

            # ── Interpretação da saída conforme a task ─────────────
            # multitask → tupla (dist_pred, seg_logits) [dist já tem sigmoid]
            #             ou dict {'out','aux'} (deeplab/resunet legados)
            # classification → tensor de logits (B, C, H, W)
            # regression     → tensor de distância (B, 1, H, W), já com sigmoid
            dist_np = prob_np = None
            if task == 'multitask':
                if isinstance(out, tuple):
                    dist_raw, seg_raw = out
                    dist_np = dist_raw.squeeze(1).cpu().numpy()
                    prob_np = torch.softmax(seg_raw, dim=1).cpu().numpy()
                else:  # dict legado
                    dist_np = torch.sigmoid(out['aux']).squeeze(1).cpu().numpy()
                    prob_np = torch.softmax(out['out'], dim=1).cpu().numpy()
            elif task == 'classification':
                seg_raw = out['out'] if isinstance(out, dict) else out
                prob_np = torch.softmax(seg_raw, dim=1).cpu().numpy()
            else:  # regression
                dist_raw = out['aux'] if isinstance(out, dict) else out
                # se vier sem sigmoid (dict legado) aplica; o caminho novo já tem
                if isinstance(out, dict):
                    dist_raw = torch.sigmoid(dist_raw)
                dist_np = dist_raw.squeeze(1).cpu().numpy()

            p = patch_size
            for b in range(n_in_batch):
                r = int(r_batch[b]); c = int(c_batch[b])
                if do_seg:
                    acc_seg[r:r+p, c:c+p, :] += prob_np[b].transpose(1, 2, 0)
                if do_reg:
                    acc_dist[r:r+p, c:c+p]   += dist_np[b]
                count[r:r+p, c:c+p] += 1.0

    # divide pelo contador (evita divisão por zero nas bordas)
    mask = count > 0
    if do_seg:
        for k in range(n_classes):
            acc_seg[mask, k] /= count[mask]
    if do_reg:
        acc_dist[mask] /= count[mask]

    return acc_seg, acc_dist


# ═════════════════════════════════════════════════════════════
# 6.  Salvar rasters georreferenciados
# ═════════════════════════════════════════════════════════════

def save_raster(path: str, array: np.ndarray, meta: dict,
                dtype: str = 'float32', force_crs=None) -> None:
    """
    Salva um array numpy como GeoTIFF preservando CRS e transform
    da imagem de referência — usa rasterio com meta.copy(), idêntico
    ao labeled_dataset_construction.py.

    Parameters
    ----------
    path  : caminho de saída
    array : (rows, cols) ou (rows, cols, bands)
    meta  : rasterio meta da imagem de referência (src.meta.copy())
    dtype : 'uint8', 'int32' ou 'float32'
    force_crs : CRS a aplicar quando o meta NÃO tiver CRS definido (None/vazio),
                ou para sobrescrever o CRS do meta. Aceita 'EPSG:31983', um
                inteiro EPSG (31983) ou um objeto rasterio.crs.CRS. Se None,
                mantém o CRS do meta como está.
    """
    import rasterio as rio
    from rasterio.crs import CRS

    if array.ndim == 2:
        n_bands = 1
        rows, cols = array.shape
    else:
        rows, cols, n_bands = array.shape

    out_meta = meta.copy()
    out_meta.update({
        'count'    : n_bands,
        'dtype'    : dtype,
        'nodata'   : 0,
        'compress' : 'lzw',       # compressão sem perda
        'bigtiff'  : 'IF_SAFER',  # suporta arquivos > 4 GB
    })

    # ── Garantia de CRS ───────────────────────────────────────
    # Sintoma clássico de "saída deslocada": o GeoTIFF é gravado SEM CRS
    # (meta['crs'] is None) — o transform posiciona os pixels em coordenadas
    # UTM, mas sem dizer QUAL é o sistema, o visualizador interpreta errado.
    # Aqui resolvemos isso explicitamente.
    def _as_crs(c):
        if c is None:
            return None
        if isinstance(c, CRS):
            return c
        if isinstance(c, int):
            return CRS.from_epsg(c)
        return CRS.from_string(str(c))      # ex.: 'EPSG:31983'

    meta_crs  = out_meta.get('crs', None)
    force_crs = _as_crs(force_crs)
    if force_crs is not None:
        # força/sobrescreve (usado p/ corrigir rasters sem CRS ou com CRS errado)
        if meta_crs is not None and meta_crs != force_crs:
            print(f"  [CRS] sobrescrevendo {meta_crs} → {force_crs}")
        elif meta_crs is None:
            print(f"  [CRS] meta sem CRS → aplicando {force_crs}")
        out_meta['crs'] = force_crs
    elif meta_crs is None:
        # nenhum CRS no meta e nenhum forçado: avisa (vai sair sem georref!)
        print(f"  ⚠️  [CRS] AVISO: imagem de referência SEM CRS e --output_crs não "
              f"definido. O raster '{os.path.basename(path)}' sairá sem "
              f"georreferenciamento e pode aparecer deslocado. "
              f"Use --output_crs EPSG:31983 (ou o EPSG correto).")

    with rio.open(path, 'w', **out_meta) as dst:
        if array.ndim == 2:
            dst.write(array.astype(dtype), 1)
        else:
            for b in range(n_bands):
                dst.write(array[:, :, b].astype(dtype), b + 1)

    print(f"  Salvo: {path}")


# ═════════════════════════════════════════════════════════════
# 6b. Inferência de dimensões a partir do state_dict
# ═════════════════════════════════════════════════════════════

def _infer_n_classes(state: dict, args) -> int:
    """
    Determina n_classes a partir dos pesos da camada de saída no checkpoint.

    Chaves por arquitetura (train_CLAUDE.py):
      MultitaskHead (BPATUNet/smp) → clf_head.3.weight    shape[0]
      DeepLabV3                    → classifier.4.weight  shape[0]
      ResUnet wrapper              → base.output_layer.0.weight shape[0]
    """
    candidates = [
        'clf_head.3.weight',              # MultitaskHead (nova arquitetura)
        'clf_head.weight',                # variante direta
        'classifier.4.weight',            # DeepLabV3
        'base.output_layer.0.weight',     # ResUnet
        'seg_head.weight',                # BPATUNet antigo (single-task)
        'seg_head.2.weight',              # SMPMultitaskWrapper antigo
        'outc.weight',                    # genérico
    ]
    for key in candidates:
        if key in state:
            n = int(state[key].shape[0])
            print(f"  [INFO] n_classes inferido de '{key}': {n}")
            return n

    # fallback genérico: busca qualquer Conv2d de saída
    for key, val in state.items():
        if val.ndim == 4 and val.shape[2] == 1 and val.shape[3] == 1:
            if 'clf' in key or 'seg' in key or 'classifier' in key:
                if key.endswith('.weight') and val.shape[0] > 1:
                    n = int(val.shape[0])
                    print(f"  [INFO] n_classes inferido (fallback) de '{key}': {n}")
                    return n

    print(f"  [WARN] Não foi possível inferir n_classes. Usando --n_classes={args.n_classes}")
    return args.n_classes


def _infer_n_channels(state: dict, args, image_bands: int) -> int:
    """
    Determina n_channels (bandas de entrada) a partir do checkpoint.
    Se o checkpoint foi treinado com BandAdaptationLayer, o número real
    de canais de entrada pode diferir do que o encoder espera.

    Para BPATUNet o band_adapt.adapt.weight tem shape (target, in, 1, 1)
    — usamos in como n_channels.
    Para smp o primeiro conv do encoder tem shape (out, in, k, k).
    """
    candidates = [
        # ── LateFusion: braço RGB (prioridade) ────────────────
        ('backbone_rgb.encoder.conv1.weight', 1),          # smp ResNet (fusão RGB)
        ('backbone_rgb.encoder.features.0.weight', 1),     # smp EfficientNet (fusão RGB)
        ('backbone_rgb.band_adapt.adapt.weight', 1),       # BPATUNet (fusão RGB)
        ('backbone_rgb.encoder.first_res_cov.weight', 1),  # resunet (fusão RGB)
        # ── Single-source ─────────────────────────────────────
        ('band_adapt.adapt.weight', 1),          # BPATUNet adapt: (target, IN, 1, 1)
        ('feature_extractor.band_adapt.adapt.weight', 1),  # BPATUNet sob MultitaskHead
        ('backbone.encoder.conv1.weight', 1),     # smp ResNet encoder
        ('feature_extractor.encoder.conv1.weight', 1),     # smp sob MultitaskHead
        ('backbone.encoder.features.0.weight', 1),# smp EfficientNet
        ('backbone.conv1.weight', 1),             # DeepLabV3
        ('base.input_layer.0.weight', 1),         # ResUnet
    ]
    for key, dim in candidates:
        if key in state:
            n = int(state[key].shape[dim])
            return n

    # Se BandAdaptationLayer não existe (in_channels == target_channels),
    # o número de canais é o mesmo da imagem
    print(f"  [INFO] n_channels inferido da imagem: {image_bands}")
    return image_bands


def _infer_bpat_config(state: dict, args) -> tuple[int, int]:
    """
    Infere base_channels e encoder_depth do BPATUNet a partir do state_dict.

    base_channels : shape[0] do primeiro DoubleConv do encoder
                    chave 'feature_extractor.inc.net.0.weight' → shape (base_ch, 4, 3, 3)

    encoder_depth : deduzido pela presença das chaves de cada nível
                    down1 → depth ≥ 2 (sempre presente)
                    down2 / be3 → depth ≥ 3
                    down3 / be4 → depth == 4
    """
    # ── base_channels ─────────────────────────────────────────
    base_ch = args.base_channels   # fallback
    for key in ('feature_extractor.inc.net.0.weight',   # MultitaskHead
                'inc.net.0.weight'):                     # single-task
        if key in state:
            base_ch = int(state[key].shape[0])
            print(f"  [INFO] base_channels inferido de '{key}': {base_ch}")
            break
    else:
        print(f"  [WARN] base_channels não inferido. Usando --base_channels={base_ch}")

    # ── encoder_depth ─────────────────────────────────────────
    # BPATUNet nova (ModuleList): chaves 'downs.0','downs.1',... Conta o maior
    # índice. Compat: nomes antigos 'down2'/'down3'.
    down_idx = [int(m.group(1))
                for k in state
                for m in [re.search(r'downs\.(\d+)\.', k)] if m]
    if down_idx:
        enc_depth = max(down_idx) + 1
    else:
        has_down3 = any('down3' in k for k in state)
        has_down2 = any('down2' in k for k in state)
        enc_depth = 4 if has_down3 else (3 if has_down2 else 2)
    print(f"  [INFO] encoder_depth inferido do state_dict: {enc_depth}")

    return base_ch, enc_depth


def _remap_state_dict(state: dict) -> dict:
    """
    Renomeia chaves do checkpoint para corresponder aos nomes de atributo
    atuais do BPATUNet.

    Versão antiga (treinada com nomes longos) → versão atual (nomes curtos):
      boundary_enhance_2  →  be2
      boundary_enhance_3  →  be3
      boundary_enhance_4  →  be4
      boundary_enhance_5  →  be5
      channel_att_3       →  ca3
      channel_att_4       →  ca4

    Aplica tanto em chaves diretas (single-task) quanto em chaves com
    prefixo 'feature_extractor.' (MultitaskHead).
    """
    renames = {
        'boundary_enhance_2': 'be2',
        'boundary_enhance_3': 'be3',
        'boundary_enhance_4': 'be4',
        'boundary_enhance_5': 'be5',
        'channel_att_3'     : 'ca3',
        'channel_att_4'     : 'ca4',
    }
    new_state = {}
    renamed   = 0
    for key, val in state.items():
        new_key = key
        for old_name, new_name in renames.items():
            if old_name in new_key:
                new_key = new_key.replace(old_name, new_name)
                renamed += 1
                break
        new_state[new_key] = val

    if renamed:
        print(f"  [INFO] {renamed} chaves renomeadas no state_dict "
              f"(boundary_enhance_* → be*, channel_att_* → ca*)")
    return new_state

def _as_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ('1', 'true', 't', 'yes', 'y', 'sim', 's')


def restore_args_from_checkpoint(args, ckpt):
    """
    Sobrescreve em `args` TODOS os campos de arquitetura/dados a partir de
    ckpt['args'] (vars(args) salvos pelo treino). O checkpoint é a FONTE DA
    VERDADE — assim o predict precisa apenas dos caminhos (imagem, CHM, modelo,
    saída); todo o resto (arch, task, fusion, patch_size, normalização,
    nº de classes, profundidades, etc.) é extraído automaticamente.

    Campos restaurados (se presentes no checkpoint):
      arquitetura : arch, fusion, dist_guided, base_channels, num_classes
      tarefa      : task, task_multitask, task_classification, task_regression,
                    multitask, ignore_index, ignore_class_value,
                    effective_ignore_index, label_shift
      fusão/lidar : latefusion, lidar_channels, norm_values
      geometria   : patch_size, rgb_resolution, chm_resolution
    """
    ckpt_args = ckpt.get('args', {}) or {}
    if not ckpt_args:
        print("  ⚠️  [ckpt] checkpoint SEM 'args' salvos (treino antigo?). "
              "Usarei os defaults/flags do predict — confira manualmente "
              "arch/task/fusion/patch_size/norm_values.")
        return args, False

    restored = [
        'arch', 'fusion', 'dist_guided', 'base_channels', 'num_classes',
        'task', 'task_multitask', 'task_classification', 'task_regression',
        'multitask', 'ignore_index', 'ignore_class_value',
        'effective_ignore_index', 'label_shift',
        'latefusion', 'lidar_channels', 'norm_values',
        'patch_size', 'rgb_resolution', 'chm_resolution',
        'rgb_norm', 'rgb_mean', 'rgb_std', 'rgb_min', 'rgb_max', 'rgb_norm_const',
        'lidar_norm', 'lidar_minmax_scope', 'lidar_min', 'lidar_max',
    ]
    changes = []
    for key in restored:
        if key in ckpt_args and ckpt_args[key] is not None:
            old = getattr(args, key, None)
            setattr(args, key, ckpt_args[key])
            if old is not None and str(old) != str(ckpt_args[key]):
                changes.append(f"{key}: {old} → {ckpt_args[key]}")

    # 'n_classes' do predict é sinônimo de 'num_classes' do treino
    if 'num_classes' in ckpt_args and ckpt_args['num_classes'] is not None:
        args.n_classes = ckpt_args['num_classes']

    # Origem do checkpoint: o train_fusion_..._on_the_fly.py não tem
    # --encoder_depth/--backbone (o backbone single-source é fixo, igual ao da
    # LateFusion); o treino legado (train_CLAUDE.py) salva encoder_depth.
    args.ckpt_treino_atual = 'encoder_depth' not in ckpt_args

    print("  [ckpt] arquitetura restaurada do checkpoint:")
    print(f"         arch={getattr(args,'arch','?')} | task={getattr(args,'task','?')} "
          f"| fusion={getattr(args,'fusion','?')} | latefusion={getattr(args,'latefusion','?')}")
    print(f"         patch_size={getattr(args,'patch_size','?')} | "
          f"num_classes={getattr(args,'num_classes','?')} | "
          f"dist_guided={getattr(args,'dist_guided','?')}")
    if getattr(args, 'latefusion', False):
        print(f"         lidar_channels={getattr(args,'lidar_channels','?')} | "
              f"norm_values={getattr(args,'norm_values','?')} | "
              f"chm_res={getattr(args,'chm_resolution','?')} | "
              f"rgb_res={getattr(args,'rgb_resolution','?')}")
    if changes:
        print("  [ckpt] (defaults do CLI sobrescritos: " + " | ".join(changes) + ")")
    return args, True


def predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # ── 7b. Carrega checkpoint PRIMEIRO e restaura a arquitetura ──
    # O checkpoint é a fonte da verdade: dele extraímos arch, task, fusion,
    # patch_size, normalização, nº de classes, etc. O usuário só informa os
    # caminhos (imagem, CHM, modelo, saída).
    print(f"Carregando checkpoint: {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    args, _have_ckpt_args = restore_args_from_checkpoint(args, ckpt)

    # out_dir default: <pasta_do_checkpoint>/../prediction
    if not getattr(args, 'out_dir', None):
        ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        args.out_dir = os.path.join(os.path.dirname(ckpt_dir), 'prediction')
        print(f"  [out_dir] não informado → usando {args.out_dir}")

    # Resolve a task a partir do que foi restaurado (task já vem pronto do
    # checkpoint; _resolve_task é fallback caso o ckpt seja antigo).
    if not hasattr(args, 'task') or args.task is None:
        args = _resolve_task(args)

    # ── Remapeamento inverso de classes ───────────────────────
    # Checkpoints antigos treinados com ignore_index False deslocaram os
    # rótulos 1..K → 0..K-1 (label_shift=-1); os atuais não deslocam
    # (label_shift=0, fundo 0 é classe). O label_shift salvo no checkpoint é
    # a fonte da verdade; sem ele, cai na regra antiga baseada em ignore_index.
    if getattr(args, 'label_shift', None) is not None:
        args.label_unshift = -int(args.label_shift)
    else:
        args.label_unshift = 0 if _as_bool(args.ignore_index) else 1
    if args.label_unshift:
        print(f"[classes] treino deslocou os rótulos → pred_class será remapeado "
              f"+{args.label_unshift} (0..K-1 → 1..K) para casar com os rótulos originais.")
    else:
        print("[classes] pred_class mantido como está (rótulos originais).")

    # ── 7a. Lê metadados da imagem (sem carregar pixels) ──────
    # checkpoints antigos não têm rgb_norm → /255 (comportamento anterior)
    reader = LazyTiffReader(args.image_path, norm=True,
                            rgb_norm=getattr(args, 'rgb_norm', None) or 'div255',
                            rgb_mean=getattr(args, 'rgb_mean', None),
                            rgb_std=getattr(args, 'rgb_std', None),
                            rgb_min=getattr(args, 'rgb_min', None),
                            rgb_max=getattr(args, 'rgb_max', None),
                            rgb_norm_const=getattr(args, 'rgb_norm_const', None))
    print(f"Imagem: {reader.rows} × {reader.cols} px  |  {reader.bands} bandas "
          f"| RGB norm={reader.rgb_norm}")

    # ── 7a-mask. Máscara de mangue (foca a predição na classe-alvo) ──
    mangrove_mask = None
    if _as_bool(getattr(args, 'use_mangrove_mask', False)):
        if not getattr(args, 'mask_path', None):
            sys.exit("--use_mangrove_mask requer --mask_path (raster de classes, "
                     "ex.: pred_class.tif com 1=mangue).")
        mangrove_mask = MangroveMask(
            mask_path      = args.mask_path,
            expected_shape = (reader.rows, reader.cols),
            target_values  = getattr(args, 'mask_target_values', [1]),
        )

    # ── 7a-bis. Leitor do CHM (apenas em modo LateFusion) ─────
    chm_reader = None
    if getattr(args, 'latefusion', False):
        if not args.chm_path:
            sys.exit("O modelo foi treinado com LateFusion (latefusion=True no "
                     "checkpoint), então --chm_path é obrigatório (raster CHM/LiDAR).")
        # os min/max (e a escala) vêm do raster do TREINO → avisa se for outro
        ckpt_chm = (ckpt.get('args') or {}).get('chm_path')
        if ckpt_chm and os.path.normcase(os.path.abspath(ckpt_chm)) != \
                os.path.normcase(os.path.abspath(args.chm_path)):
            print(f"  ⚠️  [CHM] --chm_path ({args.chm_path}) difere do raster do "
                  f"treino ({ckpt_chm}). A normalização usa os min/max do TREINO: "
                  f"confira se é a mesma fonte, bandas e escala.")
        lidar_norm = getattr(args, 'lidar_norm', None)   # None = checkpoint antigo
        # norm_values vem do checkpoint; valida contra lidar_channels (também do ckpt)
        if lidar_norm != 'minmax' and getattr(args, 'norm_values', None) is not None:
            if len(args.norm_values) != args.lidar_channels:
                sys.exit(f"Inconsistência no checkpoint: norm_values tem "
                         f"{len(args.norm_values)} valores mas lidar_channels="
                         f"{args.lidar_channels}.")
            print(f"[Norm] LiDAR normalizado por banda: {list(args.norm_values)} "
                  f"(clamp [0,1]) | RGB {reader.rgb_norm}")
        elif lidar_norm == 'minmax':
            print(f"[Norm] LiDAR min-max com os min/max do treino "
                  f"({len(args.lidar_min)} bandas) | RGB {reader.rgb_norm}")
        chm_reader = LazyChmReader(
            chm_path       = args.chm_path,
            rgb_meta       = reader.meta,
            patch_size     = args.patch_size,
            rgb_resolution = args.rgb_resolution,
            chm_resolution = args.chm_resolution,
            norm           = True,
            lidar_channels = args.lidar_channels,
            norm_values    = getattr(args, 'norm_values', None),
            lidar_norm     = lidar_norm,
            lidar_min      = getattr(args, 'lidar_min', None),
            lidar_max      = getattr(args, 'lidar_max', None),
        )

    # ── 7c. Prepara state_dict e infere dimensões do modelo ───
    state = ckpt.get('state_dict', ckpt)
    # remove prefixo "module." de modelos treinados com DataParallel
    state = {k.replace('module.', ''): v for k, v in state.items()}
    # ── Infere n_classes a partir da camada de saída ───────────
    # Só faz sentido quando a task inclui classificação. Para regressão
    # pura não há clf_head; usamos 1 (placeholder, não é usado).
    if args.task == 'regression':
        n_classes = 1
        print("  task=regression → sem classificação (n_classes=1 placeholder)")
    else:
        n_classes = _infer_n_classes(state, args)
        print(f"  n_classes inferido do checkpoint: {n_classes}")

    # ── Infere n_channels a partir da primeira camada conv ─────
    n_channels = _infer_n_channels(state, args, reader.bands)
    print(f"  n_channels inferido do checkpoint: {n_channels}")

    # ── Infere base_channels e encoder_depth (BPATUNet) ───────
    # Estes parâmetros definem o tamanho do modelo — devem ser
    # idênticos ao treino, senão load_state_dict falha com size mismatch.
    if args.arch == 'BPATUNet' and not getattr(args, 'latefusion', False):
        base_channels, encoder_depth = _infer_bpat_config(state, args)
        # sobrescreve os argumentos com os valores do checkpoint
        args.base_channels = base_channels
        args.encoder_depth = encoder_depth

    # ── Constrói modelo com as dimensões corretas ──────────────
    model = build_model(args, n_channels=n_channels, n_classes=n_classes)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [WARN] Chaves ausentes: {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"  [WARN] Chaves inesperadas: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")
    if not missing and not unexpected:
        print("  ✓ State dict carregado sem discrepâncias")

    model = model.to(device)
    model.eval()

    # ── 7c. Acumuladores multi-overlap (conforme a task) ──────
    rows, cols = reader.rows, reader.cols
    do_seg = args.task in ('multitask', 'classification')
    do_reg = args.task in ('multitask', 'regression')
    final_seg  = np.zeros((rows, cols, n_classes), dtype=np.float32) if do_seg else None
    final_dist = np.zeros((rows, cols),            dtype=np.float32) if do_reg else None

    overlaps = sorted(set(args.overlaps))   # ex: [0.3, 0.5, 0.7]
    n_passes = len(overlaps)

    for ov in overlaps:
        desc = f"overlap={ov:.1f}"
        print(f"\nPassagem com {desc}  "
              f"(stride={max(1, int(round(args.patch_size * (1 - ov))))} px)")

        seg_ov, dist_ov = predict_one_overlap(
            model      = model,
            reader     = reader,
            patch_size = args.patch_size,
            overlap    = ov,
            n_classes  = n_classes,
            batch_size = args.batch_size,
            device     = device,
            desc       = desc,
            task       = args.task,
            chm_reader = chm_reader,
            mask       = mangrove_mask,
        )
        # acumula: cada overlap contribui igualmente para a média final
        if do_seg:
            final_seg  += seg_ov  / n_passes
        if do_reg:
            final_dist += dist_ov / n_passes

    # ── 7d/7e. Decisão final + salvamento (conforme a task) ───
    os.makedirs(args.out_dir, exist_ok=True)
    meta = reader.meta   # crs + transform + driver preservados via rasterio

    # ── Resolução do CRS de saída ─────────────────────────────
    # Prioridade: --output_crs (se passado) > CRS da imagem de referência.
    # Se a imagem de referência não tiver CRS (causa do "deslocamento"), e
    # --output_crs não for passado, avisamos. Default de --output_crs é
    # EPSG:31983 (SIRGAS 2000 / UTM 23S) — o CRS do seu projeto.
    out_crs = getattr(args, 'output_crs', None)
    ref_crs = meta.get('crs', None)
    if out_crs:
        force_crs = out_crs
        if ref_crs is None:
            print(f"  [CRS] imagem de referência SEM CRS → saída forçada p/ {out_crs}")
        elif str(ref_crs) != str(out_crs):
            print(f"  [CRS] --output_crs={out_crs} difere do CRS da referência "
                  f"({ref_crs}). Vou GRAVAR {out_crs} (assumindo que o transform "
                  f"está nesse sistema). Se a referência já estava certa, omita "
                  f"--output_crs.")
    else:
        force_crs = None
        if ref_crs is None:
            print("  ⚠️  [CRS] imagem de referência SEM CRS e --output_crs não "
                  "informado → a saída pode aparecer deslocada. Re-rode com "
                  "--output_crs EPSG:31983 (ou o EPSG correto).")
        else:
            print(f"  [CRS] preservando CRS da referência: {ref_crs}")
    print("\nSalvando rasters...")

    if do_seg:
        pred_class = np.argmax(final_seg, axis=2).astype(np.int32)   # 0..K-1
        pred_prob  = np.max(final_seg,    axis=2).astype(np.float32)  # (rows, cols)

        # desfaz o shift do treino (0..K-1 → 1..K) quando aplicável
        if args.label_unshift:
            pred_class = pred_class + args.label_unshift
        pred_class = pred_class.astype(np.uint8)

        # ── Corte pela máscara de mangue (zera tudo fora do alvo) ──
        # Mesmo pulando tiles fora do mangue, tiles que CRUZAM a borda
        # mangue/não-mangue ainda predizem espécie em pixels não-mangue;
        # o corte garante que a saída seja exatamente 0 fora do mangue.
        if mangrove_mask is not None:
            pred_class = mangrove_mask.apply(pred_class)   # fundo=0
            pred_prob  = mangrove_mask.apply(pred_prob)
            final_seg  = mangrove_mask.apply(final_seg)    # afeta probs por classe
            print("  [máscara] saída de classificação cortada para o mangue (fora=0).")

        save_raster(os.path.join(args.out_dir, 'pred_class.tif'),
                    pred_class, meta, dtype='uint8', force_crs=force_crs)
        #save_raster(os.path.join(args.out_dir, 'pred_prob.tif'),
        #            pred_prob,  meta, dtype='float32', force_crs=force_crs)
        
        if args.save_class_probs:
            # Mantém o salvamento do arquivo multibanda (opcional, pode apagar se quiser economizar espaço)
            #save_raster(os.path.join(args.out_dir, 'pred_prob_per_class_multiband.tif'),
            #            final_seg, meta, dtype='float32')
            
            # --- NOVO: Salva um TIF individual para cada classe ---
            print("\n  Salvando mapas de probabilidade individuais por classe...")
            for k in range(n_classes):
                # Extrai apenas as probabilidades da classe K (2D array)
                prob_map_k = final_seg[:, :, k]
                
                # Calcula o ID real da classe (adicionando o unshift, se houver)
                class_id = k + (args.label_unshift if args.label_unshift else 0)
                
                # Gera o nome do arquivo dinamicamente
                file_name = f'pred_prob_class_{class_id}.tif'
                
                # Salva o raster
                save_raster(os.path.join(args.out_dir, file_name),
                            prob_map_k, meta, dtype='float32', force_crs=force_crs)

    if do_reg:
        if mangrove_mask is not None:
            final_dist = mangrove_mask.apply(final_dist)
            print("  [máscara] saída de distância cortada para o mangue (fora=0).")
        save_raster(os.path.join(args.out_dir, 'pred_dist.tif'),
                    final_dist, meta, dtype='float32', force_crs=force_crs)

    print("\n✓ Predição concluída.")
    print(f"  Task                    : {args.task}")
    if do_seg:
        uniq, cnts = np.unique(pred_class, return_counts=True)
        total = pred_class.size
        print(f"  Classes únicas preditas : {uniq.tolist()}")
        print(f"  n_classes do checkpoint : {n_classes}")
        # distribuição por classe — revela se alguma classe é rara/ausente
        print("  Distribuição de classes :")
        for u, c in zip(uniq.tolist(), cnts.tolist()):
            print(f"      classe {u:>3} : {c:>12,} px ({100.0*c/total:5.2f}%)")
        # alerta se o nº de classes previstas < nº de classes do modelo
        n_expected = n_classes
        n_seen = len([u for u in uniq.tolist() if (u != 0 or not args.label_unshift)])
        if n_seen < n_expected:
            print(f"  ⚠️  Apenas {n_seen} de {n_expected} classes apareceram na predição. "
                  f"Se isso não for esperado, verifique balanceamento de treino / "
                  f"se --ignore_index e --n_classes batem com o treino.")
        print(f"  Prob. média máx./pixel  : {pred_prob.mean():.3f}")
    if do_reg:
        print(f"  Dist. média predita     : {final_dist.mean():.3f}")
    print(f"  Resultados em           : {args.out_dir}")


# ═════════════════════════════════════════════════════════════
# region Argumentos
# ═════════════════════════════════════════════════════════════

# Remove-Item Env:PROJ_LIB -ErrorAction SilentlyContinue
# Remove-Item Env:PROJ_DATA -ErrorAction SilentlyContinue
# Remove-Item Env:GDAL_DATA -ErrorAction SilentlyContinue

def get_args():
    p = argparse.ArgumentParser(
        description='Predição multi-overlap para mapeamento de espécies de mangue')

    # ── ESSENCIAIS: o que você realmente precisa informar ─────
    p.add_argument('--image_path', type=str, required=False,
                   default=r"F:\Consultorias\Bioflore\Marajo\deteccao_macicos\images\CBERS_2025\CBERS_4A_WPM_20250617_212_114_L4_BAND0_CUT.tif",
                   help='[ESSENCIAL] Imagem RGB de entrada (.tif).')
    p.add_argument('--checkpoint', type=str, required=False,
                   default=r"F:\Consultorias\Bioflore\Marajo\deteccao_macicos\codes\multitask_PAN_MULTI_FUSION_without_background_macicos_only_with_dist_map\weights\last_model.pth",
                   help='[ESSENCIAL] Checkpoint do modelo (.pth). Toda a arquitetura '
                        '(arch, task, fusion, patch_size, normalização, nº de classes…) '
                        'é extraída daqui automaticamente.')
    p.add_argument('--chm_path', type=str,
                   default=r"F:\Consultorias\Bioflore\Marajo\deteccao_macicos\images\CBERS_2025\composicao_colorida_2025_cut.tif",
                   help='[ESSENCIAL se o modelo usa LateFusion] Raster CHM/LiDAR — o '
                        'MESMO usado no treino (--chm_path do treino). O código detecta '
                        'pelo checkpoint se a fusão é necessária e aplica a mesma '
                        'normalização (min-max com os min/max do treino).')
    p.add_argument('--out_dir', type=str, default=r"F:\Consultorias\Bioflore\Marajo\deteccao_macicos\codes\multitask_PAN_MULTI_FUSION_without_background_macicos_only_with_dist_map\prediction",
                   help='[ESSENCIAL] Pasta de saída. Se omitido, usa <pasta_do_'
                        'checkpoint>/../prediction.')
    p.add_argument('--output_crs', type=str, default='EPSG:32722',
                   help="CRS gravado nos rasters de saída. Default 'EPSG:31983' "
                        "(SIRGAS 2000 / UTM 23S). Resolve saída deslocada quando a "
                        "imagem de referência não tem CRS. Passe '' p/ preservar o "
                        "CRS da imagem; ou um EPSG p/ forçar outro (ex.: EPSG:32723).")

    # ── Opcionais de inferência (têm defaults sensatos) ───────
    p.add_argument('--overlaps', type=float, nargs='+', default=[0.3, 0.5, 0.7],
                   help='Overlaps das passagens da janela deslizante; a predição '
                        'final é a média. Mais/maiores overlaps suavizam costuras.')
    p.add_argument('--batch_size', type=int, default=1500,
                   help='Patches por batch na GPU (ajuste à sua memória).')

    # ── Máscara de mangue (foca a predição na classe-alvo) ────
    p.add_argument('--use_mangrove_mask', default=False,
                   help="Se True, restringe a predição de espécies aos pixels de "
                        "mangue de um raster de classes (--mask_path). Pula tiles "
                        "fora do mangue E zera a saída fora dele. Se False, prediz "
                        "a imagem inteira (comportamento padrão).")
    p.add_argument('--mask_path', type=str,
                   default=r"E:\MANGROVE_SPECIES_MAPPING\exp\train_MANGROVE_MASK_256_BPATUNet_RGB_CHM_GROUND\prediction\pred_class_BERTIOGA_WITH_GAP.tif",
                   help="Raster de classes usado como máscara (mesmo grid da imagem "
                        "de entrada). Ex.: pred_class.tif com 1=mangue, 2=não-mangue, "
                        "3=água. Só usado se --use_mangrove_mask.")
    p.add_argument('--mask_target_values', type=int, nargs='+', default=[1],
                   help="Valor(es) de pixel da máscara considerados ALVO (mangue). "
                        "Default [1]. Aceita múltiplos (ex.: --mask_target_values 1 4).")

    # ═══════════════════════════════════════════════════════════
    #  PARÂMETROS AUTO-EXTRAÍDOS DO CHECKPOINT
    # ═══════════════════════════════════════════════════════════
    # Todos os campos abaixo são lidos automaticamente de ckpt['args'] (o treino
    # salva vars(args)). Você NÃO precisa informá-los — eles existem só como
    # fallback para checkpoints antigos que não tenham 'args'. Default None =
    # "use o que vier do checkpoint".
    g = p.add_argument_group(
        'fallback (auto-extraído do checkpoint — normalmente não use)')
    g.add_argument('--arch', type=str, default=None,
                   choices=['deeplabv3_resnet50', 'unet', 'unetplusplus',
                            'manet', 'linknet', 'BPATUNet', 'resunet'])
    g.add_argument('--backbone', type=str, default='resnet50')
    g.add_argument('--encoder_depth', type=int, default=2, choices=[2, 3, 4, 5])
    g.add_argument('--base_channels', type=int, default=32, choices=[32, 64])
    g.add_argument('--task_multitask',  type=_as_bool, default=None)
    g.add_argument('--task_classification', type=_as_bool, default=None)
    g.add_argument('--task_regression', type=_as_bool, default=None)
    g.add_argument('--dist_guided', type=_as_bool, default=None)
    g.add_argument('--n_classes', type=int, default=None)
    g.add_argument('--num_classes', type=int, default=None)
    g.add_argument('--filters', type=int, nargs='+', default=[32, 32, 32, 32])
    g.add_argument('--ignore_index', type=_as_bool, default=True)
    g.add_argument('--ignore_class_value', type=int, default=0)
    g.add_argument('--latefusion', type=_as_bool, default=None)
    g.add_argument('--fusion', type=str, default=None,
                   choices=['concat', 'gated', 'xattn'])
    g.add_argument('--lidar_channels', type=int, default=None)
    g.add_argument('--norm_values', type=float, nargs='+', default=None)
    g.add_argument('--rgb_resolution', type=float, default=None)
    g.add_argument('--chm_resolution', type=float, default=None)
    g.add_argument('--patch_size', type=int, default=None)

    # ── Saída extra ───────────────────────────────────────────
    p.add_argument('--save_class_probs', default=False,
                   help='Se ativado, salva também as probabilidades por classe '
                        '(pred_prob_per_class.tif, n_classes bandas)')

    return p.parse_args()

# Remove-Item Env:PROJ_LIB -ErrorAction SilentlyContinue
# Remove-Item Env:PROJ_DATA -ErrorAction SilentlyContinue
# Remove-Item Env:GDAL_DATA -ErrorAction SilentlyContinue

# ═════════════════════════════════════════════════════════════
# 9.  Entry point
# ═════════════════════════════════════════════════════════════

def _resolve_task(args):
    """Resolve as três flags booleanas em args.task canônico.
    Prioridade: multitask > classification > regression; nenhuma → multitask.
    Mantém args.multitask por retrocompatibilidade.
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

    print("=" * 60)
    print("Mangrove Species Mapping — Predição multi-overlap")
    print("=" * 60)
    print("  Parâmetros de arquitetura serão extraídos do checkpoint.")
    print(f"  image_path : {args.image_path}")
    print(f"  checkpoint : {args.checkpoint}")
    print(f"  chm_path   : {args.chm_path}")
    print(f"  out_dir    : {args.out_dir or '(derivado do checkpoint)'}")
    print("=" * 60)

    predict(args)