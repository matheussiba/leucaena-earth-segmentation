"""
run.py - orquestrador unico do fluxo PhD (pool de tiles + AOI vetorial + registry).

UM motor para todas as redes (02_train.py --arch ...), configurado por um
experiment.yaml nomeado, sobre um POOL central de tiles. Cada fase escreve
timestamps/metricas no run.yaml do run.

Fases:
  lidar     (opcional) CHM+INT via run_lidar (Engine python por default)
  resolve   AOI.gpkg -> VRT rgbnir + VRT lidar + annotations recortado
  labels    (proc 1) rasteriza rotulos/distancia na grade do AOI
  train     (proc 2) treina o motor on-the-fly lendo os VRTs do AOI
  predict   (proc 3) aplica pesos de um run a um AOI (NAO cria run novo)
  post      (proc 4) pos-processa os rasters de predicao
  report    agrega todos os run.yaml em reports/summary.{md,csv}
  eval      preenche predictions[].metrics (F1/IoU/P/R/%cena cls foco) nos run.yaml
  all       lidar(python) -> resolve -> labels -> train -> predict -> post

Config:
  .env (por PC)            PHD_RUN_ROOT, TRAIN_PROFILE=home|lab, R_BIN (opcional)
  experiment.yaml          rascunho editavel em experiments/<rede>/<nome>/
  run-00N/experiment.yaml  snapshot imutavel da config usada naquele run

Exemplos:
  python scripts/phd-run/run.py train   -aoi 261002-riodaspedras -experiment manet/latefusion-multitask -run run-001
  python scripts/phd-run/run.py predict -aoi 260515-piracicaba   -experiment manet/latefusion-multitask -run run-001
  python scripts/phd-run/run.py report
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))  # pipeline.aoi_resolve etc.
import _miniyaml as yaml  # noqa: E402

MANET_DIR = REPO / "scripts" / "manet-latefusion-multitask"
COMPOSE = HERE / "docker-compose.yml"

# host(sob o root) -> mount no container
MOUNTS = {
    "tiles": "/data/tiles",
    "aois": "/data/aois",
    "annotations": "/data/annotations",
    "datasets": "/data/datasets",
    "experiments": "/data/experiments",
    "reports": "/data/reports",
}


def _now() -> str:
    return _dt.datetime.now().replace(microsecond=0).isoformat()


# --------------------------------------------------------------------------- env
def load_env(repo: Path) -> dict:
    env = {}
    f = repo / ".env"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    # .env tem prioridade sobre o ambiente do processo
    for k in ("PHD_RUN_ROOT", "TRAIN_PROFILE", "R_BIN"):
        if k not in env and os.environ.get(k):
            env[k] = os.environ[k]
    return env


class Layout:
    """Deriva todas as pastas a partir de PHD_RUN_ROOT."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.tiles = self.root / "tiles"
        self.opt_raw_rgb = self.tiles / "optical" / "raw" / "rgb"
        self.opt_raw_ir = self.tiles / "optical" / "raw" / "ir"
        self.rgbnir = self.tiles / "optical" / "rgbnir"
        self.lidar_raw = self.tiles / "lidar" / "raw"
        self.lidar_products = self.tiles / "lidar" / "products"
        self.lidar_rasters = self.tiles / "lidar" / "rasters"
        self.aois = self.root / "aois"
        self.annotations = self.root / "annotations"
        self.datasets = self.root / "datasets"
        self.experiments = self.root / "experiments"
        self.reports = self.root / "reports"

    def ensure(self):
        for p in (self.opt_raw_rgb, self.opt_raw_ir, self.rgbnir,
                  self.lidar_raw, self.lidar_products, self.lidar_rasters, self.aois,
                  self.annotations, self.datasets, self.experiments, self.reports):
            p.mkdir(parents=True, exist_ok=True)

    def to_container(self, host_path: Path) -> str:
        p = Path(host_path).resolve()
        for name, mount in MOUNTS.items():
            base = (self.root / name).resolve()
            try:
                rel = p.relative_to(base)
                return f"{mount}/{rel.as_posix()}" if rel.parts else mount
            except ValueError:
                continue
        return str(p).replace("\\", "/")


# --------------------------------------------------------------------------- exec
def _run(cmd, cwd=None, dry=False):
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}\n")
    if dry:
        return 0
    return subprocess.call(cmd, cwd=str(cwd) if cwd else None)


def _ensure_local_cuda_image() -> None:
    """Garante imagem local leucaena-segmentation:cuda (nao puxa do Hub)."""
    tip = (
        "Imagem Docker local ausente: leucaena-segmentation:cuda\n"
        "  Na raiz do repo, com Docker Desktop ligado:\n"
        "    docker build -t leucaena-segmentation:cuda .\n"
        "  Depois rode de novo o .\\run.cmd ..."
    )
    try:
        r = subprocess.run(
            ["docker", "image", "inspect", "leucaena-segmentation:cuda"],
            capture_output=True, text=True,
        )
    except FileNotFoundError:
        sys.exit("[ABORT] docker nao encontrado no PATH.\n" + tip)
    if r.returncode != 0:
        sys.exit("[ABORT] " + tip)


def docker_run(lay: Layout, profile: str, script_rel: str, args: list, dry=False):
    """Roda `python <script_rel> <args>` dentro do container unico."""
    if not dry:
        _ensure_local_cuda_image()
    env = os.environ.copy()
    env["TILES_HOST"] = str(lay.tiles)
    env["AOIS_HOST"] = str(lay.aois)
    env["ANNOT_HOST"] = str(lay.annotations)
    env["DATASETS_HOST"] = str(lay.datasets)
    env["EXPERIMENTS_HOST"] = str(lay.experiments)
    env["REPORTS_HOST"] = str(lay.reports)
    env["DOCKER_SHM_SIZE"] = "16gb" if profile == "lab" else "8gb"
    cmd = ["docker", "compose", "-f", str(COMPOSE), "run", "--rm", "phd-run",
           "python", script_rel] + [str(a) for a in args]
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}\n")
    if dry:
        return 0
    return subprocess.call(cmd, cwd=str(REPO), env=env)


def host_run(script_abs: Path, args: list, dry=False):
    return _run([sys.executable, str(script_abs)] + [str(a) for a in args], cwd=REPO, dry=dry)


# --------------------------------------------------------------------------- cfg
def experiment_yaml_path(lay: Layout, experiment: str) -> Path:
    return lay.experiments / experiment / "experiment.yaml"


def load_experiment(lay: Layout, experiment: str) -> dict:
    exp_file = experiment_yaml_path(lay, experiment)
    if not exp_file.exists():
        sys.exit(f"[ABORT] experiment.yaml nao encontrado: {exp_file}\n"
                 f"        crie experiments/{experiment}/experiment.yaml "
                 f"(veja scripts/phd-run/templates/experiment.yaml).")
    cfg = yaml.load(exp_file)
    cfg.setdefault("network", experiment.split("/")[0])
    return cfg


def _preflight_annotation_dns(lay: Layout, cfg: dict) -> None:
    """ABORT imediato se o GPKG global tem DN fora de [1, num_classes)."""
    ann = annotations_path(lay, cfg)
    if ann is None or not ann.is_file():
        return
    ncls = int(cfg.get("num_classes", 6) or 6)
    try:
        import geopandas as gpd  # noqa: PLC0415
        gdf = gpd.read_file(ann)
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] preflight annotations: nao leu {ann} ({exc})")
        return
    if "DN" not in gdf.columns:
        sys.exit(f"[ABORT] annotations sem coluna DN: {ann}")
    dns = sorted({int(v) for v in gdf["DN"].dropna().unique()})
    bad = [d for d in dns if d < 1 or d >= ncls]
    if bad:
        sys.exit(
            f"[ABORT] {ann.name} tem DN {bad} fora de [1, {ncls - 1}] "
            f"(num_classes={ncls}).\n"
            f"        DNs no arquivo: {dns}\n"
            f"        Corrija o GPKG (ex. 9→5) ANTES de lidar/labels/train."
        )
    print(f"[preflight] annotations OK: DN={dns} (num_classes={ncls})")


def config_summary(cfg: dict) -> dict:
    """Subset plano da config p/ run.yaml / report (nao substitui o snapshot)."""
    keys = (
        "network", "task", "latefusion", "fusion", "channels", "lidar_channels",
        "num_classes", "report_class", "patch_size", "on_the_fly", "save_patches",
        "epochs", "lr", "pretrained", "rgb_norm_method", "lidar_norm",
        "class_weights", "reg_loss", "rgb_resolution", "chm_resolution",
    )
    out = {k: cfg[k] for k in keys if k in cfg}
    labels = cfg.get("labels")
    if isinstance(labels, dict):
        out["labels_version"] = labels.get("version")
        out["chm_min_m"] = labels.get("chm_min_m")
        out["chm_filter_class"] = labels.get("chm_filter_class")
    return out


def snapshot_experiment(lay: Layout, experiment: str, rd: Path, *, dry: bool = False) -> Path:
    """Copia o rascunho experiments/.../experiment.yaml -> run-00N/experiment.yaml.

    Imutavel: se o snapshot ja existe, nao sobrescreve (preserva o que o run usou).
    """
    src = experiment_yaml_path(lay, experiment)
    dst = rd / "experiment.yaml"
    if dry:
        print(f"[dry-run] snapshot {src} -> {dst}")
        return dst
    rd.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"[snapshot] ja existe (imutavel): {dst}")
        return dst
    if not src.is_file():
        sys.exit(f"[ABORT] experiment.yaml nao encontrado p/ snapshot: {src}")
    shutil.copy2(src, dst)
    print(f"[snapshot] {src} -> {dst}")
    return dst


def dataset_id(aoi: str, cfg: dict, masked: bool = False) -> str:
    v = cfg.get("labels", {}).get("version", 1) if isinstance(cfg.get("labels"), dict) else 1
    did = f"{aoi}__lbl-v{v}"
    if masked:
        did += "__mask"
    if cfg.get("save_patches"):
        did += f"__p{cfg.get('patch_size', 256)}"
    return did


def aoi_mask_path(lay: Layout, aoi: str, cfg: dict, *,
                  want_mask: bool = False,
                  mask_gpkg_cli: str | None = None) -> Path | None:
    """Resolve mask so sob pedido explicito (-mask / -mask-gpkg).

    Nome: aois/<aoi>-mask.gpkg  (ex.: -aoi 261002-riodaspedras
    → aois/261002-riodaspedras-mask.gpkg). Outro AOI = outro arquivo.
    Sem -mask, nao recorta mesmo se o arquivo existir.
    """
    if not want_mask and not mask_gpkg_cli and not cfg.get("mask_gpkg"):
        return None
    explicit = mask_gpkg_cli or cfg.get("mask_gpkg") or cfg.get("aoi_mask")
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = lay.aois / p
        if not p.is_file():
            sys.exit(f"[ABORT] mask nao encontrada: {p}")
        return p
    from pipeline.aoi_resolve import find_aoi_mask  # noqa: PLC0415
    cand = find_aoi_mask(lay.aois, aoi, None)
    if cand is None:
        sys.exit(
            f"[ABORT] -mask pedido mas nao existe aois/{aoi}-mask.gpkg "
            f"(nem .geojson/.shp)\n"
            f"        Exporte no QGIS como GeoPackage ou use -mask-gpkg PATH"
        )
    return cand


def annotations_path(lay: Layout, cfg: dict) -> Path | None:
    name = cfg.get("annotations")
    if name:
        p = lay.annotations / name if not os.path.isabs(name) else Path(name)
        return p
    gpkgs = sorted(lay.annotations.glob("*.gpkg"))
    return gpkgs[0] if gpkgs else None


# --------------------------------------------------------------------------- run.yaml
def run_dir(lay: Layout, experiment: str, run: str) -> Path:
    return lay.experiments / experiment / run


def load_run_yaml(rd: Path) -> dict:
    f = rd / "run.yaml"
    return yaml.load(f) if f.exists() else {}


def save_run_yaml(rd: Path, data: dict):
    rd.mkdir(parents=True, exist_ok=True)
    yaml.dump(data, rd / "run.yaml")


def stamp(rd: Path, phase: str, field: str, extra: dict | None = None):
    data = load_run_yaml(rd)
    ts = data.setdefault("timestamps", {})
    if phase in ("build_labels", "train", "predict", "postprocess"):
        ts.setdefault(phase, {})[field] = _now()
    else:
        ts[field] = _now()
    if extra:
        data.update(extra)
    save_run_yaml(rd, data)


# --------------------------------------------------------------------------- phases
def phase_lidar(aoi: str, engine: str = "python", force: bool = False, dry=False) -> int:
    """Roda run_lidar.ps1 (default Engine=python). Skip se rasters ja existem (sem -Force)."""
    ps1 = REPO / "run_lidar.ps1"
    if not ps1.is_file():
        sys.exit(f"[ABORT] nao encontrado: {ps1}")
    cmd = [
        "powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", str(ps1), "-aoi", aoi, "-Engine", engine,
    ]
    if force:
        cmd.append("-Force")
    print(f"[lidar] Engine={engine} aoi={aoi}")
    return _run(cmd, cwd=REPO, dry=dry)


def _resolve_raster(out: Path, stem: str) -> Path | None:
    """Prefere .tif (mask crop); senao .vrt."""
    for ext in (".tif", ".vrt"):
        p = out / f"{stem}{ext}"
        if p.is_file():
            return p
    return None


def phase_resolve(lay: Layout, cfg: dict, aoi: str, runtime: str, dry=False) -> dict:
    want_mask = bool(cfg.get("_use_mask"))
    mask_cli = cfg.get("_mask_gpkg")
    mask_p = aoi_mask_path(lay, aoi, cfg, want_mask=want_mask, mask_gpkg_cli=mask_cli)
    did = dataset_id(aoi, cfg, masked=mask_p is not None)
    out = lay.datasets / did / "_resolve"
    annot = annotations_path(lay, cfg)
    src_prefix = MOUNTS["tiles"] if runtime == "docker" else str(lay.tiles).replace("\\", "/")
    resolver = REPO / "pipeline" / "aoi_resolve.py"
    args = ["--root", str(lay.root), "--aoi", aoi, "--out-dir", str(out),
            "--src-prefix", src_prefix]
    if annot:
        args += ["--annotations", str(annot)]
    nf = cfg.get("nomenc_field")
    if nf:
        args += ["--nomenc-field", nf]
    if mask_p is not None:
        args += ["--mask-gpkg", str(mask_p)]
    else:
        args += ["--no-mask"]
    needs_lidar = bool(cfg.get("latefusion", False)) or int(cfg.get("lidar_channels", 0) or 0) > 0
    if needs_lidar:
        args += ["--require-lidar"]
    if cfg.get("_force_resolve"):
        args += ["--force"]
    # resolve no host; se saidas ja existem, aoi_resolve faz skip (sem --force)
    rc = host_run(resolver, args, dry=dry)
    if rc != 0 and not dry:
        sys.exit(f"[ABORT] resolve falhou (rc={rc}).")
    rgb_vrt = _resolve_raster(out, "aoi_rgbnir")
    lid_vrt = _resolve_raster(out, "aoi_lidar")
    if not dry:
        if rgb_vrt is None:
            sys.exit(f"[ABORT] resolve nao gerou aoi_rgbnir.tif/.vrt em {out}")
        meta = out / "resolve.json"
        info = {}
        if meta.is_file():
            try:
                info = json.loads(meta.read_text(encoding="utf-8"))
                if not (info.get("rgbnir") or {}).get("ok"):
                    sys.exit(
                        f"[ABORT] aoi_rgbnir invalido: {info.get('rgbnir')}. "
                        "Coloque o tile do AOI em tiles/optical/rgbnir/ e rode de novo."
                    )
                # paths do resolve.json (apos mask apontam p/ .tif)
                for key, var in (("rgbnir_vrt", "rgb"), ("lidar_vrt", "lid")):
                    raw = info.get(key)
                    if raw:
                        p = Path(raw)
                        if p.is_file():
                            if var == "rgb":
                                rgb_vrt = p
                            else:
                                lid_vrt = p
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] nao li resolve.json: {exc}")
        if mask_p is not None:
            print(f"[resolve] mask ativa -> dataset_id={did}")
        needs_lidar = bool(cfg.get("latefusion", False)) or int(cfg.get("lidar_channels", 0) or 0) > 0
        lidar_ok = lid_vrt is not None and lid_vrt.is_file()
        if needs_lidar and not lidar_ok:
            missing = (info.get("lidar") or {}).get("missing") or info.get("tiles")
            rasters = lay.tiles / "lidar" / "rasters"
            sys.exit(
                f"[ABORT] latefusion precisa de LiDAR, mas aoi_lidar falhou "
                f"(missing={missing}).\n"
                f"        Esperado: {rasters}/<tile>.tif  [CHM, INTENSITY]\n"
                f"        Gere com: .\\run_lidar.cmd -aoi {aoi} -Engine python -Force\n"
                f"        Depois rode de novo: .\\run.cmd all ..."
            )
    return {
        "dataset_id": did,
        "resolve_dir": out,
        "rgbnir_vrt": rgb_vrt,
        "lidar_vrt": lid_vrt,
        "annotations_clip": out / "aoi_annotations.gpkg",
        "masks": lay.datasets / did / "masks",
        "masked": mask_p is not None,
    }


def _p(lay, runtime, host_path):
    return lay.to_container(host_path) if runtime == "docker" else str(host_path).replace("\\", "/")


def phase_labels(lay, cfg, aoi, res, runtime, dry=False):
    lab = cfg.get("labels", {}) if isinstance(cfg.get("labels"), dict) else {}
    res["masks"].mkdir(parents=True, exist_ok=True)
    args = [
        "--img_path", _p(lay, runtime, res["rgbnir_vrt"]),
        "--shp_path", _p(lay, runtime, res["annotations_clip"]),
        "--out_base_dir", _p(lay, runtime, res["masks"]),
        "--classes_dist_map"] + [str(c) for c in lab.get("classes_dist_map", [1])] + [
        "--dist_map_as_cluster", str(lab.get("dist_map_as_cluster", True)),
        "--cluster_edge_m", str(lab.get("cluster_edge_m", 10.0)),
        "--gaussian_sigma", str(lab.get("gaussian_sigma", 2)),
        "--chm_min_m", str(lab.get("chm_min_m", 4.0)),
        "--chm_filter_class", str(lab.get("chm_filter_class", 1)),
        "--train_ratio", str(lab.get("train_ratio", 0.8)),
        "--num_classes", str(int(cfg.get("num_classes", 6))),
    ]
    if res.get("lidar_vrt") and Path(res["lidar_vrt"]).is_file():
        args += ["--chm_path", _p(lay, runtime, res["lidar_vrt"])]
    elif cfg.get("latefusion"):
        sys.exit(
            f"[ABORT] labels com latefusion precisa de aoi_lidar.vrt. "
            f"Rode: .\\run_lidar.cmd -aoi {aoi} -Engine python -Force"
        )
    script_rel = "scripts/manet-latefusion-multitask/01_build_labels.py"
    if runtime == "docker":
        return docker_run(lay, cfg.get("_profile", "home"), script_rel, args, dry=dry)
    return host_run(MANET_DIR / "01_build_labels.py", args, dry=dry)


def phase_train(lay, cfg, aoi, res, run, runtime, dry=False):
    prof = cfg.get("_profile", "home")
    pconf = (cfg.get("profiles", {}) or {}).get(prof, {}) or {}
    rd = run_dir(lay, cfg["_experiment"], run)
    model_dir = rd / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    masks = res["masks"]
    task = cfg.get("task", "multitask")
    args = [
        "--on_the_fly", "True",
        "--image_path", _p(lay, runtime, res["rgbnir_vrt"]),
        "--ref_path", _p(lay, runtime, masks / "ITCs_labels_train.tif"),
        "--depth_path", _p(lay, runtime, masks / "ITCs_distance_train.tif"),
        "--chm_path", _p(lay, runtime, res["lidar_vrt"]),
        "--model_dir", _p(lay, runtime, model_dir),
        "--arch", str(cfg.get("network", "manet")),
        "--channels", str(cfg.get("channels", 4)),
        "--lidar_channels", str(cfg.get("lidar_channels", 2)),
        "--latefusion", str(cfg.get("latefusion", True)),
        "--fusion", str(cfg.get("fusion", "concat")),
        "--task_multitask", str(task == "multitask"),
        "--task_classification", str(task == "classification"),
        "--task_regression", str(task == "regression"),
        "--num_classes", str(cfg.get("num_classes", 6)),
        # Classe reportada por época (IoU/F1); default = chm_filter_class / 1 (leucaena)
        "--report_class", str(
            cfg.get("report_class",
                    (cfg.get("labels") or {}).get("chm_filter_class", 1))
        ),
        "--rgb_norm_method", str(cfg.get("rgb_norm_method", "minmax")),
        "--lidar_norm", str(cfg.get("lidar_norm", "minmax")),
        "--class_weights", str(cfg.get("class_weights", "inverse")),
        "--reg_loss", str(cfg.get("reg_loss", "l1")),
        "--rgb_resolution", str(cfg.get("rgb_resolution", 0.25)),
        "--chm_resolution", str(cfg.get("chm_resolution", 0.25)),
        "--patch_size", str(cfg.get("patch_size", 512)),
        "--epochs", str(cfg.get("epochs", 60)),
        # SGD usa --base_lr (peak do warmup+cosine). yaml `lr` (e opcional
        # `base_lr`) alimenta os dois; só --lr era ignorado e caía no default 0.01.
        "--lr", str(cfg.get("base_lr", cfg.get("lr", 1e-4))),
        "--base_lr", str(cfg.get("base_lr", cfg.get("lr", 1e-4))),
        "--batch_size", str(pconf.get("batch", 2)),
        "--num_workers", str(pconf.get("workers", 2)),
        "--prefetch_factor", str(pconf.get("prefetch", 2)),
        "--resume", "True",
        "--amp",
    ]
    if cfg.get("pretrained", True):
        args += ["--pretrained"]
    # init_from (finetune): injeta pesos de outro run
    if cfg.get("_init_from"):
        init_host = Path(cfg["_init_from"])
        if not init_host.is_absolute():
            # aceita relativo a PHD_RUN_ROOT ou a experiments/
            cand = [lay.root / init_host, lay.experiments / init_host]
            init_host = next((c for c in cand if c.is_file()), lay.root / init_host)
        args += ["--init_from", _p(lay, runtime, init_host)]
    extra = cfg.get("train_extra_args") or []
    if not isinstance(extra, (list, tuple)):
        sys.exit(f"[ABORT] train_extra_args deve ser lista no experiment.yaml, veio: {type(extra).__name__}")
    args += [str(a) for a in extra]
    script_rel = "scripts/manet-latefusion-multitask/02_train.py"
    if runtime == "docker":
        return docker_run(lay, prof, script_rel, args, dry=dry)
    return host_run(MANET_DIR / "02_train.py", args, dry=dry)


def _find_ckpt(model_dir: Path) -> Path | None:
    for sub in (model_dir / "weights", model_dir):
        for name in ("best_model.pth", "last_model.pth"):
            if (sub / name).exists():
                return sub / name
    return None


def phase_predict(lay, cfg, aoi, res, run, runtime, dry=False):
    rd = run_dir(lay, cfg["_experiment"], run)
    ckpt = _find_ckpt(rd / "models")
    if not ckpt and not dry:
        sys.exit(f"[ABORT] checkpoint nao encontrado em {rd/'models'}. Treine antes (phase train).")
    ckpt = ckpt or (rd / "models" / "weights" / "best_model.pth")
    out = rd / "predictions" / aoi
    out.mkdir(parents=True, exist_ok=True)
    pr = cfg.get("predict", {}) if isinstance(cfg.get("predict"), dict) else {}
    overlaps = pr.get("overlaps", [0.3, 0.5, 0.7])
    # default 4: AOIs grandes (piracicaba) com batch 8+ derrubam o container (EOF/OOM)
    args = [
        "--image_path", _p(lay, runtime, res["rgbnir_vrt"]),
        "--chm_path", _p(lay, runtime, res["lidar_vrt"]),
        "--checkpoint", _p(lay, runtime, ckpt),
        "--out_dir", _p(lay, runtime, out),
        "--output_crs", str(cfg.get("output_crs", "EPSG:31983")),
        "--batch_size", str(pr.get("batch_size", 4)),
        "--resume", str(pr.get("resume", True)),
        "--overlaps"] + [str(o) for o in overlaps]
    script_rel = "scripts/manet-latefusion-multitask/03_predict.py"
    if runtime == "docker":
        rc = docker_run(lay, cfg.get("_profile", "home"), script_rel, args, dry=dry)
    else:
        rc = host_run(MANET_DIR / "03_predict.py", args, dry=dry)
    return rc, out


def phase_post(lay, cfg, aoi, run, runtime, dry=False):
    rd = run_dir(lay, cfg["_experiment"], run)
    out = rd / "predictions" / aoi
    po = cfg.get("post", {}) if isinstance(cfg.get("post"), dict) else {}
    args = [
        "--input", _p(lay, runtime, out / "pred_dist.tif"),
        "--out_dir", _p(lay, runtime, out),
        "--sigma", str(po.get("sigma", 2)),
        "--threshold", str(po.get("threshold", 0.5)),
        "--min_size", str(po.get("min_size", 500)),
        "--connectivity", str(po.get("connectivity", 2)),
        "--save_filtered", str(po.get("save_filtered", True)),
    ]
    script_rel = "scripts/manet-latefusion-multitask/04_postprocess.py"
    if runtime == "docker":
        return docker_run(lay, cfg.get("_profile", "home"), script_rel, args, dry=dry)
    return host_run(MANET_DIR / "04_postprocess.py", args, dry=dry)


# --------------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["lidar", "resolve", "labels", "train", "predict",
                                      "post", "eval", "report", "all"])
    ap.add_argument("-aoi", dest="aoi", help="nome do AOI (=> aois/<aoi>.gpkg)")
    ap.add_argument("-experiment", dest="experiment", help="<rede>/<nome>")
    ap.add_argument("-run", dest="run", help="run-00N (train cria; predict reusa)")
    ap.add_argument("-profile", dest="profile", default=None, help="home|lab (sobrepoe .env)")
    ap.add_argument("-runtime", dest="runtime", choices=["docker", "host"], default="docker")
    ap.add_argument("-root", dest="root", default=None, help="sobrepoe PHD_RUN_ROOT")
    ap.add_argument("-lidar-engine", dest="lidar_engine", choices=["python", "r", "geotrees"],
                    default="python", help="engine do passo lidar (default: python)")
    ap.add_argument("--force-lidar", dest="force_lidar", action="store_true",
                    help="reprocessa LiDAR mesmo se rasters ja existirem")
    ap.add_argument("--skip-lidar", dest="skip_lidar", action="store_true",
                    help="no 'all', nao roda o passo lidar (assume rasters prontos)")
    ap.add_argument("--force-resolve", dest="force_resolve", action="store_true",
                    help="Refaz crop/VRT do resolve mesmo se ja existir")
    ap.add_argument("-mask", dest="use_mask", action="store_true",
                    help="Recorta aois/<aoi>-mask.gpkg (dataset ...__mask). "
                         "Sem esta flag, nao usa mask mesmo se o arquivo existir.")
    ap.add_argument("-mask-gpkg", dest="mask_gpkg", default=None,
                    help="Path da mask (implica -mask). Relativo a aois/ ou absoluto.")
    ap.add_argument("--init-from", dest="init_from", default=None,
                    help="caminho de pesos p/ finetune (grava parent/init_from no run.yaml)")
    ap.add_argument("--parent-run", dest="parent_run", default=None)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    env = load_env(REPO)
    root = a.root or env.get("PHD_RUN_ROOT")
    if not root:
        sys.exit("[ABORT] defina PHD_RUN_ROOT no .env (ou use -root).")
    lay = Layout(Path(root))
    lay.ensure()
    profile = (a.profile or env.get("TRAIN_PROFILE", "home")).lower()

    # report / eval nao precisam de experimento/aoi
    if a.phase == "report":
        rep = REPO / "reports" / "build-registry-report.py"
        return sys.exit(host_run(rep, ["--root", str(lay.root)], dry=a.dry_run))
    if a.phase == "eval":
        from eval_leucaena import backfill_root  # noqa: PLC0415
        n = backfill_root(lay.root)
        print(f"\n[OK] fases concluidas: eval ({n} run.yaml)")
        return

    if a.phase == "lidar":
        if not a.aoi:
            sys.exit("[ABORT] -aoi <nome> e obrigatorio.")
        rc = phase_lidar(a.aoi, engine=a.lidar_engine, force=a.force_lidar, dry=a.dry_run)
        if rc not in (0, None) and not a.dry_run:
            sys.exit(f"[ABORT] lidar rc={rc}")
        print("\n[OK] fases concluidas: lidar")
        return

    if not a.experiment:
        sys.exit("[ABORT] -experiment <rede>/<nome> e obrigatorio.")
    cfg = load_experiment(lay, a.experiment)
    cfg["_experiment"] = a.experiment
    cfg["_profile"] = profile
    if a.init_from:
        cfg["_init_from"] = a.init_from
    cfg["_use_mask"] = bool(a.use_mask or a.mask_gpkg)
    if a.mask_gpkg:
        cfg["_mask_gpkg"] = a.mask_gpkg
    cfg["_force_resolve"] = bool(a.force_resolve)
    if not a.aoi:
        sys.exit("[ABORT] -aoi <nome> e obrigatorio.")

    # Fail-fast ANTES de lidar/labels: DN do GPKG global vs num_classes.
    # Evita 10+ min de Docker/CHM para descobrir DN 9 com num_classes=6.
    if a.phase in ("labels", "train", "all") and not a.dry_run:
        _preflight_annotation_dns(lay, cfg)

    needs_lidar = bool(cfg.get("latefusion", False)) or int(cfg.get("lidar_channels", 0) or 0) > 0
    if a.phase == "all":
        phases = ["resolve", "labels", "train", "predict", "post"]
        if needs_lidar and not a.skip_lidar:
            phases = ["lidar"] + phases
    else:
        phases = [a.phase]

    if "lidar" in phases:
        rc = phase_lidar(a.aoi, engine=a.lidar_engine, force=a.force_lidar, dry=a.dry_run)
        if rc not in (0, None) and not a.dry_run:
            sys.exit(f"[ABORT] lidar rc={rc}")

    res = phase_resolve(lay, cfg, a.aoi, a.runtime, dry=a.dry_run)

    # run.yaml base (apenas quando ha run e fases de treino/predicao)
    run = a.run
    if run and any(p in phases for p in ("labels", "train", "predict", "post")):
        rd = run_dir(lay, a.experiment, run)
        snap = snapshot_experiment(lay, a.experiment, rd, dry=a.dry_run)
        data = load_run_yaml(rd)
        data.update({
            "run": run, "experiment": a.experiment, "network": cfg.get("network"),
            "profile": profile, "aoi_train": a.aoi, "dataset_id": res["dataset_id"],
            "experiment_snapshot": str(snap).replace("\\", "/"),
        })
        # congela config na 1a vez (nao sobrescreve se ja havia run.yaml)
        if "config" not in data:
            data["config"] = config_summary(cfg)
        if a.parent_run:
            data["parent_run"] = a.parent_run
        if a.init_from:
            data["init_from"] = a.init_from
        data.setdefault("timestamps", {}).setdefault("run_started", _now())
        if not a.dry_run:
            save_run_yaml(rd, data)

    for ph in phases:
        if ph == "lidar":
            continue  # ja executado antes do resolve
        if ph == "resolve":
            print(f"[resolve] dataset_id={res['dataset_id']} -> {res['resolve_dir']}")
        elif ph == "labels":
            if run:
                stamp(run_dir(lay, a.experiment, run), "build_labels", "started")
            rc = phase_labels(lay, cfg, a.aoi, res, a.runtime, dry=a.dry_run)
            if rc not in (0, None) and not a.dry_run:
                sys.exit(f"[ABORT] labels rc={rc}")
            if run:
                stamp(run_dir(lay, a.experiment, run), "build_labels", "finished")
        elif ph == "train":
            if not run:
                sys.exit("[ABORT] train exige -run run-00N.")
            stamp(run_dir(lay, a.experiment, run), "train", "started")
            rc = phase_train(lay, cfg, a.aoi, res, run, a.runtime, dry=a.dry_run)
            if rc not in (0, None) and not a.dry_run:
                sys.exit(f"[ABORT] train rc={rc}")
            stamp(run_dir(lay, a.experiment, run), "train", "finished",
                  {"status": "trained"})
        elif ph == "predict":
            if not run:
                sys.exit("[ABORT] predict exige -run run-00N (reusa pesos desse run).")
            stamp(run_dir(lay, a.experiment, run), "predict", "started")
            rc, out = phase_predict(lay, cfg, a.aoi, res, run, a.runtime, dry=a.dry_run)
            if rc not in (0, None) and not a.dry_run:
                sys.exit(f"[ABORT] predict rc={rc}")
            # anexa item em predictions: + metricas leucaena (cls foco vs masks)
            rd = run_dir(lay, a.experiment, run)
            data = load_run_yaml(rd)
            preds = data.setdefault("predictions", [])
            item = {"aoi": a.aoi, "profile": profile, "at": _now(),
                    "out": str(out).replace("\\", "/")}
            if not a.dry_run:
                try:
                    from eval_leucaena import eval_pred_vs_masks, masks_dir_for_dataset  # noqa: PLC0415
                    masks = masks_dir_for_dataset(lay.root, data.get("dataset_id") or "")
                    pred_tif = Path(out) / "pred_class.tif"
                    rc_cls = int(cfg.get("report_class", 1) or 1)
                    if masks and pred_tif.is_file():
                        m = eval_pred_vs_masks(pred_tif, masks, rc_cls)
                        if m:
                            item["metrics"] = m
                            if a.aoi == data.get("aoi_train"):
                                data["metrics_train"] = {
                                    "leucaena_f1": m["labeled_f1"],
                                    "iou": m["labeled_iou"],
                                    "precision": m["labeled_precision"],
                                    "recall": m["labeled_recall"],
                                    "scene_pct_class": m["scene_pct_class"],
                                    "report_class": m["report_class"],
                                }
                            print(
                                f"[eval] cls{rc_cls} F1={m['labeled_f1']:.4f} "
                                f"IoU={m['labeled_iou']:.4f} "
                                f"P={m['labeled_precision']:.4f} "
                                f"R={m['labeled_recall']:.4f} "
                                f"cena={m['scene_pct_class']:.2f}%"
                            )
                except Exception as exc:  # noqa: BLE001
                    print(f"[warn] eval leucaena falhou: {exc}")
            preds.append(item)
            save_run_yaml(rd, data)
            stamp(rd, "predict", "finished")
        elif ph == "post":
            if not run:
                sys.exit("[ABORT] post exige -run run-00N.")
            stamp(run_dir(lay, a.experiment, run), "postprocess", "started")
            rc = phase_post(lay, cfg, a.aoi, run, a.runtime, dry=a.dry_run)
            if rc not in (0, None) and not a.dry_run:
                sys.exit(f"[ABORT] post rc={rc}")
            stamp(run_dir(lay, a.experiment, run), "postprocess", "finished")

    print("\n[OK] fases concluidas:", ", ".join(phases))


if __name__ == "__main__":
    main()
