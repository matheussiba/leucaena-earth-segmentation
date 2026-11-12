"""
Avalia pred_class.tif vs labels (classe foco, default=1 leucaena) e devolve
metricas para gravar no run.yaml / report.

Uso:
  python scripts/phd-run/eval_leucaena.py --root PHD_RUN_ROOT
      # varre experiments/**/run-*/run.yaml e preenche predictions[].metrics
  python scripts/phd-run/eval_leucaena.py --pred PATH --masks-dir PATH [--class 1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import _miniyaml as yaml  # noqa: E402


def _scores(pred, gt, c: int):
    import numpy as np  # noqa: PLC0415
    p = pred == c
    g = gt == c
    tp = int((p & g).sum())
    fp = int((p & ~g).sum())
    fn = int((~p & g).sum())
    iou = tp / (tp + fp + fn + 1e-9)
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "iou": round(float(iou), 6),
        "precision": round(float(prec), 6),
        "recall": round(float(rec), 6),
        "f1": round(float(f1), 6),
    }


def eval_pred_vs_masks(
    pred_tif: Path,
    masks_dir: Path,
    report_class: int = 1,
) -> dict | None:
    """Compara pred_class com ITCs_labels_{train,test}.tif.

    Metricas principais: labeled_* (so pixels com gt>0) — mesma tabela do batch.
    """
    import numpy as np  # noqa: PLC0415
    import rasterio  # noqa: PLC0415

    if not pred_tif.is_file():
        return None
    train_p = masks_dir / "ITCs_labels_train.tif"
    test_p = masks_dir / "ITCs_labels_test.tif"
    if not train_p.is_file() and not test_p.is_file():
        return None

    with rasterio.open(pred_tif) as src:
        pred = src.read(1)
    gt_tr = None
    gt_te = None
    if train_p.is_file():
        with rasterio.open(train_p) as src:
            gt_tr = src.read(1)
    if test_p.is_file():
        with rasterio.open(test_p) as src:
            gt_te = src.read(1)

    h = pred.shape[0]
    w = pred.shape[1]
    if gt_tr is not None:
        h = min(h, gt_tr.shape[0])
        w = min(w, gt_tr.shape[1])
    if gt_te is not None:
        h = min(h, gt_te.shape[0])
        w = min(w, gt_te.shape[1])
    pred = pred[:h, :w]
    if gt_tr is not None:
        gt_tr = gt_tr[:h, :w]
    if gt_te is not None:
        gt_te = gt_te[:h, :w]

    if gt_tr is not None and gt_te is not None:
        labeled = (gt_tr > 0) | (gt_te > 0)
        gt_lab = np.where(gt_tr > 0, gt_tr, gt_te)
    elif gt_tr is not None:
        labeled = gt_tr > 0
        gt_lab = gt_tr
    else:
        labeled = gt_te > 0
        gt_lab = gt_te

    lab = _scores(pred[labeled], gt_lab[labeled], report_class)
    out = {
        "report_class": int(report_class),
        "labeled_f1": lab["f1"],
        "labeled_iou": lab["iou"],
        "labeled_precision": lab["precision"],
        "labeled_recall": lab["recall"],
        "leucaena_f1": lab["f1"],  # alias p/ report legado
        "iou": lab["iou"],
        "scene_pct_class": round(float(100.0 * (pred == report_class).mean()), 4),
    }
    if gt_tr is not None:
        tr = _scores(pred, gt_tr, report_class)
        out["train_recall"] = tr["recall"]
        out["train_f1"] = tr["f1"]
    if gt_te is not None:
        te = _scores(pred, gt_te, report_class)
        out["test_recall"] = te["recall"]
        out["test_f1"] = te["f1"]
        out["test_iou"] = te["iou"]
        out["test_precision"] = te["precision"]
    return out


def masks_dir_for_dataset(root: Path, dataset_id: str) -> Path | None:
    if not dataset_id:
        return None
    p = root / "datasets" / dataset_id / "masks"
    return p if p.is_dir() else None


def backfill_root(root: Path, report_class: int | None = None) -> int:
    """Atualiza predictions[].metrics em todos os run.yaml sob experiments/."""
    n = 0
    exp_root = root / "experiments"
    for run_yaml in sorted(exp_root.rglob("run.yaml")):
        try:
            data = yaml.load(run_yaml)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {run_yaml}: {exc}")
            continue
        if not isinstance(data, dict):
            continue
        preds = data.get("predictions") or []
        if not preds:
            continue
        ds = data.get("dataset_id") or ""
        masks = masks_dir_for_dataset(root, ds)
        cfg_rc = report_class
        if cfg_rc is None:
            cfg = data.get("config") if isinstance(data.get("config"), dict) else {}
            cfg_rc = int(cfg.get("report_class", 1) or 1)
        changed = False
        for p in preds:
            if not isinstance(p, dict):
                continue
            aoi = p.get("aoi") or ""
            out = p.get("out")
            pred_tif = None
            if out:
                cand = Path(str(out).replace("/", "\\")) / "pred_class.tif"
                if not cand.is_file():
                    cand = Path(out) / "pred_class.tif"
                if cand.is_file():
                    pred_tif = cand
            if pred_tif is None:
                # fallback: sibling predictions/<aoi>/pred_class.tif
                pred_tif = run_yaml.parent / "predictions" / aoi / "pred_class.tif"
            if masks is None or not pred_tif.is_file():
                continue
            m = eval_pred_vs_masks(pred_tif, masks, cfg_rc)
            if not m:
                continue
            p["metrics"] = m
            changed = True
            print(
                f"[eval] {run_yaml.parent.relative_to(exp_root)} aoi={aoi} "
                f"cls{cfg_rc} F1={m['labeled_f1']:.4f} IoU={m['labeled_iou']:.4f} "
                f"P={m['labeled_precision']:.4f} R={m['labeled_recall']:.4f} "
                f"cena={m['scene_pct_class']:.2f}%"
            )
        if changed:
            # espelha a 1a predicao da AOI de treino em metrics_train p/ o report
            aoi_train = data.get("aoi_train")
            for p in preds:
                if p.get("aoi") == aoi_train and isinstance(p.get("metrics"), dict):
                    mt = p["metrics"]
                    data["metrics_train"] = {
                        "leucaena_f1": mt.get("labeled_f1", mt.get("leucaena_f1")),
                        "iou": mt.get("labeled_iou", mt.get("iou")),
                        "precision": mt.get("labeled_precision"),
                        "recall": mt.get("labeled_recall"),
                        "scene_pct_class": mt.get("scene_pct_class"),
                        "report_class": mt.get("report_class", cfg_rc),
                    }
                    break
            yaml.dump(data, run_yaml)
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="PHD_RUN_ROOT — backfill todos os runs")
    ap.add_argument("--pred", help="pred_class.tif (modo unitario)")
    ap.add_argument("--masks-dir", help="datasets/<id>/masks")
    ap.add_argument("--class", dest="report_class", type=int, default=1)
    a = ap.parse_args()
    if a.root:
        n = backfill_root(Path(a.root), a.report_class)
        print(f"[OK] {n} run.yaml atualizado(s)")
        return
    if a.pred and a.masks_dir:
        m = eval_pred_vs_masks(Path(a.pred), Path(a.masks_dir), a.report_class)
        if not m:
            sys.exit("[ABORT] nao foi possivel avaliar (arquivos ausentes)")
        print(m)
        return
    sys.exit("[ABORT] use --root PHD_RUN_ROOT  ou  --pred + --masks-dir")


if __name__ == "__main__":
    main()
