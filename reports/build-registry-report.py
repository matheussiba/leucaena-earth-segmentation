"""
build-registry-report.py
=========================
Varre todos os experiments/<rede>/<nome>/<run>/run.yaml do PHD_RUN_ROOT e gera:
  reports/summary.csv  - uma linha por run (rede, experimento, AOI treino, metricas,
                         changes, timestamps)
  reports/summary.md   - tabela legivel + evolucao run-001..run-00N por experimento
                         + MATRIZ treino-AOI x predicao-AOI (generalizacao cross-area)

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
        rows.append(d)
    return rows


def _metrics_flat(m: dict | None) -> dict:
    return {k: v for k, v in (m or {}).items()} if isinstance(m, dict) else {}


def write_csv(rows: list[dict], out: Path):
    cols = ["network", "experiment", "run", "profile", "aoi_train", "dataset_id",
            "status", "parent_run", "changes",
            "train_leucaena_f1", "train_iou", "train_best_val",
            "n_predictions", "predict_aois"]
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            mt = _metrics_flat(r.get("metrics_train"))
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
          "| rede | experimento | run | AOI treino | profile | f1(treino) | status | changes |",
          "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: (x.get("network", ""), x.get("experiment", ""),
                                         str(x.get("run", "")))):
        mt = _metrics_flat(r.get("metrics_train"))
        L.append(f"| {r.get('network','')} | {r.get('experiment','')} | {r.get('run','')} "
                 f"| {r.get('aoi_train','')} | {r.get('profile','')} "
                 f"| {mt.get('leucaena_f1','')} | {r.get('status','')} "
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
            parent = f" (de {r.get('parent_run')})" if r.get("parent_run") else ""
            L.append(f"- **{r.get('run','')}**{parent}: f1={mt.get('leucaena_f1','?')}, "
                     f"iou={mt.get('iou','?')} - {r.get('changes','') or 'sem nota'}")
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
