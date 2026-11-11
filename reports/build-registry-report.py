"""
build-registry-report.py
=========================
Varre todos os experiments/<rede>/<nome>/<run>/run.yaml do PHD_RUN_ROOT e gera:
  reports/summary.csv  - uma linha por run (rede, experimento, AOI treino, config
                         congelada, metricas, changes, timestamps)
  reports/summary.md   - tabela legivel + evolucao run-001..run-00N por experimento
                         + MATRIZ treino-AOI x predicao-AOI (generalizacao cross-area)

Config por run: run.yaml -> config:  e/ou snapshot
  experiments/<rede>/<nome>/<run>/experiment.yaml  (copia imutavel do rascunho)

Sem dependencias externas (usa scripts/phd-run/_miniyaml.py). Reaproveita a ideia
do parser de reports/fill-resultados.py, mas lendo o registry novo (run.yaml).

Uso:
  python reports/build-registry-report.py --root C:/00_PHD_MSB_RUN
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts" / "phd-run"))
import _miniyaml as yaml  # noqa: E402


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _cfg(r: dict) -> dict:
    c = r.get("config")
    return c if isinstance(c, dict) else {}


def collect(root: Path) -> list[dict]:
    rows = []
    exp_root = root / "experiments"
    for run_yaml in sorted(exp_root.glob("*/*/*/run.yaml")):
        try:
            d = yaml.load(run_yaml)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {run_yaml}: {exc}")
            continue
        if not isinstance(d, dict):
            continue
        d["_path"] = run_yaml
        # snapshot ao lado do run.yaml (imutavel); preenche config se run antigo
        snap = run_yaml.parent / "experiment.yaml"
        d["_has_snapshot"] = snap.is_file()
        if "config" not in d and snap.is_file():
            try:
                sc = yaml.load(snap)
                if isinstance(sc, dict):
                    d["config"] = {
                        k: sc[k] for k in (
                            "network", "task", "latefusion", "fusion", "patch_size",
                            "epochs", "lr", "num_classes", "report_class",
                        ) if k in sc
                    }
            except Exception as exc:  # noqa: BLE001
                print(f"[warn] snapshot {snap}: {exc}")
        rows.append(d)
    return rows


def _metrics_flat(m: dict | None) -> dict:
    return {k: v for k, v in (m or {}).items()} if isinstance(m, dict) else {}


def write_csv(rows: list[dict], out: Path):
    cols = ["network", "experiment", "run", "profile", "aoi_train", "dataset_id",
            "status", "parent_run", "changes",
            "epochs", "lr", "patch_size", "task", "latefusion", "fusion",
            "num_classes", "report_class", "has_snapshot",
            "train_leucaena_f1", "train_iou", "train_best_val",
            "n_predictions", "predict_aois"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            mt = _metrics_flat(r.get("metrics_train"))
            c = _cfg(r)
            preds = r.get("predictions") or []
            w.writerow({
                "network": r.get("network", ""),
                "experiment": r.get("experiment", ""),
                "run": r.get("run", ""),
                "profile": r.get("profile", ""),
                "aoi_train": r.get("aoi_train", ""),
                "dataset_id": r.get("dataset_id", ""),
                "status": r.get("status", ""),
                "parent_run": r.get("parent_run", ""),
                "changes": r.get("changes", ""),
                "epochs": c.get("epochs", ""),
                "lr": c.get("lr", ""),
                "patch_size": c.get("patch_size", ""),
                "task": c.get("task", ""),
                "latefusion": c.get("latefusion", ""),
                "fusion": c.get("fusion", ""),
                "num_classes": c.get("num_classes", ""),
                "report_class": c.get("report_class", ""),
                "has_snapshot": r.get("_has_snapshot", False),
                "train_leucaena_f1": mt.get("leucaena_f1", ""),
                "train_iou": mt.get("iou", ""),
                "train_best_val": mt.get("best_val", ""),
                "n_predictions": len(preds),
                "predict_aois": ";".join(sorted({p.get("aoi", "") for p in preds})),
            })


def _pred_metric(p: dict):
    m = _metrics_flat(p.get("metrics"))
    for k in ("leucaena_f1", "f1", "iou"):
        if k in m:
            return _num(m[k])
    return None


def write_md(rows: list[dict], out: Path):
    L = ["# Registry - resumo de experimentos", "",
         f"Runs encontrados: **{len(rows)}**", ""]

    # 1) tabela por run
    L += ["## Runs", "",
          "| rede | experimento | run | AOI treino | ep | lr | patch | f1(treino) | status | changes |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x.get("network", ""), x.get("experiment", ""),
                                         str(x.get("run", "")))):
        mt = _metrics_flat(r.get("metrics_train"))
        c = _cfg(r)
        L.append(f"| {r.get('network','')} | {r.get('experiment','')} | {r.get('run','')} "
                 f"| {r.get('aoi_train','')} | {c.get('epochs','')} | {c.get('lr','')} "
                 f"| {c.get('patch_size','')} | {mt.get('leucaena_f1','')} "
                 f"| {r.get('status','')} "
                 f"| {str(r.get('changes','')).replace('|','/')} |")

    # 2) evolucao por experimento (run-001 -> run-00N)
    L += ["", "## Evolucao por experimento", ""]
    by_exp: dict[str, list[dict]] = {}
    for r in rows:
        key = f"{r.get('network','?')}/{(r.get('experiment','') or '').split('/')[-1]}"
        by_exp.setdefault(key, []).append(r)
    for exp, rs in sorted(by_exp.items()):
        L.append(f"### {exp}")
        for r in sorted(rs, key=lambda x: str(x.get("run", ""))):
            mt = _metrics_flat(r.get("metrics_train"))
            c = _cfg(r)
            parent = f" (de {r.get('parent_run')})" if r.get("parent_run") else ""
            snap = " +snapshot" if r.get("_has_snapshot") else ""
            cfg_bits = []
            for k, lab in (("epochs", "ep"), ("lr", "lr"), ("patch_size", "p")):
                if c.get(k) not in (None, ""):
                    cfg_bits.append(f"{lab}={c.get(k)}")
            cfg_s = (", " + ", ".join(cfg_bits)) if cfg_bits else ""
            L.append(f"- **{r.get('run','')}**{parent}{snap}: f1={mt.get('leucaena_f1','?')}, "
                     f"iou={mt.get('iou','?')}{cfg_s} - {r.get('changes','') or 'sem nota'}")
        L.append("")

    # 3) matriz treino-AOI x predicao-AOI
    train_aois = sorted({r.get("aoi_train", "") for r in rows if r.get("aoi_train")})
    pred_aois = sorted({p.get("aoi", "") for r in rows for p in (r.get("predictions") or [])
                        if p.get("aoi")})
    L += ["## Matriz treino x predicao (generalizacao cross-area)", ""]
    if train_aois and pred_aois:
        header = "| treino \\ predicao | " + " | ".join(pred_aois) + " |"
        L.append(header)
        L.append("|" + "---|" * (len(pred_aois) + 1))
        for ta in train_aois:
            cells = []
            for pa in pred_aois:
                best = None
                for r in rows:
                    if r.get("aoi_train") != ta:
                        continue
                    for p in (r.get("predictions") or []):
                        if p.get("aoi") == pa:
                            v = _pred_metric(p)
                            if v is not None and (best is None or v > best):
                                best = v
                cells.append("" if best is None else f"{best:.3f}")
            L.append(f"| {ta} | " + " | ".join(cells) + " |")
    else:
        L.append("_(sem predicoes registradas ainda)_")
    L.append("")

    out.write_text("\n".join(L), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="PHD_RUN_ROOT")
    a = ap.parse_args()
    root = Path(a.root)
    rep = root / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    rows = collect(root)
    write_csv(rows, rep / "summary.csv")
    write_md(rows, rep / "summary.md")
    print(f"[OK] {len(rows)} run(s) -> {rep/'summary.csv'} , {rep/'summary.md'}")


if __name__ == "__main__":
    main()
