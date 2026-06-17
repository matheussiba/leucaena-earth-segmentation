"""
Preenche automaticamente reports/RESULTADOS.md com os resultados do exp_4.

Roda dentro do container Docker após o treino terminar.

Uso:
    python reports/fill-resultados.py
    python reports/fill-resultados.py -e 4 --split test
    python reports/fill-resultados.py --no-eval   # usa log já existente
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPORTS_DIR = Path(__file__).parent
FIGURAS_DIR = REPORTS_DIR / "figuras"
TEMPLATE_PATH = REPORTS_DIR / "RESULTADOS.md"
ROOT = REPORTS_DIR.parent

sys.path.insert(0, str(ROOT))


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-e", "--experiment", type=int, default=4, help="Experiment id (default: 4)")
    p.add_argument("--split", default="test", choices=("train", "val", "test"))
    p.add_argument("--patches-dir", default=None, help="Override patches dir (default: conf/paths.PATH_PATCHES_DIR)")
    p.add_argument("--no-eval", action="store_true", help="Skip running eval-patches.py; use existing log file")
    p.add_argument("--no-plot", action="store_true", help="Skip generating training curves plot")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def parse_manifest(manifest_path: str) -> dict:
    counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    tiles: set[str] = set()
    total = 0
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.get("split", "")
            if split in counts:
                counts[split] += 1
            total += 1
            tile = row.get("tile_name", "")
            if tile:
                tiles.add(tile)
    return {"total": total, "counts": counts, "tiles": sorted(tiles)}


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------

def run_eval(exp: int, split: str, patches_dir: str) -> str:
    cmd = [sys.executable, str(ROOT / "eval-patches.py"), "-e", str(exp), "--split", split,
           "--patches-dir", patches_dir]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT))
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("eval-patches.py failed — check the output above.")
    return result.stdout


def _num(s: str) -> str:
    return s.replace(",", "")


def parse_eval_output(text: str) -> dict:
    m: dict = {}
    for cls in ("background", "leucaena"):
        pat = rf"{cls}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d,]+)"
        match = re.search(pat, text)
        if match:
            m[cls] = {
                "precision": float(match.group(1)),
                "recall":    float(match.group(2)),
                "f1":        float(match.group(3)),
                "iou":       float(match.group(4)),
                "support":   int(_num(match.group(5))),
            }
    for key, pat in [
        ("macro_f1", r"macro F1\s+([\d.]+)"),
        ("accuracy", r"accuracy\s+([\d.]+)"),
    ]:
        match = re.search(pat, text)
        if match:
            m[key] = float(match.group(1))
    for key, pat in [
        ("tp", r"TP \(leucaena correct\)\s*=\s*([\d,]+)"),
        ("fp", r"FP \(background -> leucaena\)\s*=\s*([\d,]+)"),
        ("fn", r"FN \(leucaena -> background\)\s*=\s*([\d,]+)"),
        ("tn", r"TN \(background correct\)\s*=\s*([\d,]+)"),
    ]:
        match = re.search(pat, text)
        if match:
            m[key] = int(_num(match.group(1)))
    return m


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def run_plot_training(exp: int) -> None:
    cmd = [sys.executable, "-m", "utils.plot_training", "-e", str(exp)]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(ROOT), check=False)
    src = ROOT / "experiments" / f"exp_{exp}" / "logs" / "training_curves.png"
    if src.exists():
        FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
        dst = FIGURAS_DIR / f"exp{exp}_training_curves.png"
        shutil.copy(src, dst)
        print(f"  Copied -> {dst}")
    else:
        print(f"  Warning: training_curves.png not found at {src}")


def run_viz_patches(patches_dir: str) -> None:
    """Generate exactly 2 patch panels: dense leucaena and hard (sparse) leucaena."""
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    for select, out_name in [
        ("most-leucaena", "patch_densa.png"),
        ("hard",          "patch_dificil.png"),
    ]:
        tmp_dir = FIGURAS_DIR / f"_tmp_{select}"
        cmd = [
            sys.executable, str(ROOT / "viz-patches.py"),
            "--patches-dir", patches_dir,
            "--select", select,
            "--top-k", "1",
            "--out-dir", str(tmp_dir),
        ]
        print(f"  viz-patches ({select}): {' '.join(cmd[-4:])}")
        result = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        # viz-patches.py saves <tmp_dir>/<patch_id>.png — move first file found
        pngs = list(tmp_dir.glob("*.png")) if tmp_dir.exists() else []
        if pngs:
            dst = FIGURAS_DIR / out_name
            shutil.move(str(pngs[0]), str(dst))
            shutil.rmtree(tmp_dir, ignore_errors=True)
            print(f"    -> {dst}")
        else:
            print(f"    Warning: no PNG produced for select={select!r}. Check viz-patches.py output.")
            if result.stderr:
                print(result.stderr[:400])


# ---------------------------------------------------------------------------
# Template filling
# ---------------------------------------------------------------------------

def _f(val: float, d: int = 4) -> str:
    return f"{val:.{d}f}"


def _fi(val: int) -> str:
    # Format integer with dots as thousands separator (PT-BR)
    return f"{val:,}".replace(",", ".")


def _replace_row(lines: list[str], pattern: str, replacement: str) -> None:
    """In-place: replace the first line matching pattern."""
    for i, line in enumerate(lines):
        if re.search(pattern, line):
            lines[i] = replacement
            return


def fill_template(template: str, manifest: dict, metrics: dict) -> str:
    counts = manifest["counts"]
    tiles  = manifest["tiles"]

    # --- Section 2: tiles, total patches, splits ---
    # Tiles
    tile_sample = ", ".join(tiles[:6]) + ("..." if len(tiles) > 6 else "")
    template = re.sub(
        r"\[PREENCHER — ex\.: 17 tiles SF-23-Y-A-IV-2-\\\*\]",
        f"{len(tiles)} tile(s): {tile_sample}",
        template,
    )
    # Total patches
    template = re.sub(
        r"\*\*\[PREENCHER\]\*\* \(ex\.: 9\.341\)",
        f"**{_fi(manifest['total'])}**",
        template,
    )
    # Train / val / test counts
    tr, va, te = counts["train"], counts["val"], counts["test"]
    template = re.sub(
        r"\*\*\[PREENCHER\]\*\* / \[PREENCHER\] / \[PREENCHER\]",
        f"**{_fi(tr)}** / {_fi(va)} / {_fi(te)}",
        template,
    )

    if not metrics:
        return template

    bg = metrics.get("background", {})
    lc = metrics.get("leucaena",   {})
    mf1 = metrics.get("macro_f1", 0.0)
    tp  = metrics.get("tp", 0)
    fp  = metrics.get("fp", 0)
    fn  = metrics.get("fn", 0)
    tn  = metrics.get("tn", 0)

    lines = template.split("\n")

    # --- Table 4.1: per-class metrics ---
    _replace_row(lines,
        r"\| background \|",
        f"| background | {_f(bg.get('precision',0))} | {_f(bg.get('recall',0))} | "
        f"{_f(bg.get('f1',0))} | {_f(bg.get('iou',0))} | {_fi(bg.get('support',0))} |",
    )
    _replace_row(lines,
        r"\| \*\*leucaena\*\* \|",
        f"| **leucaena** | {_f(lc.get('precision',0))} | {_f(lc.get('recall',0))} | "
        f"{_f(lc.get('f1',0))} | {_f(lc.get('iou',0))} | {_fi(lc.get('support',0))} |",
    )
    _replace_row(lines,
        r"\| \*macro F1\* \|",
        f"| *macro F1* | | | {_f(mf1)} | | |",
    )

    # --- Table 4.2: confusion matrix ---
    _replace_row(lines,
        r"\| \*\*Real: background\*\* \|",
        f"| **Real: background** | TN = {_fi(tn)} | FP = {_fi(fp)} |",
    )
    _replace_row(lines,
        r"\| \*\*Real: leucaena\*\* \|",
        f"| **Real: leucaena** | FN = {_fi(fn)} | TP = {_fi(tp)} |",
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()
    exp = args.experiment

    from conf import paths  # imported here so ROOT is in sys.path first
    patches_dir = args.patches_dir or str(paths.PATH_PATCHES_DIR)
    manifest_path = os.path.join(patches_dir, "manifest.csv")

    # 1. Manifest
    print(f"\n[1/3] Manifest: {manifest_path}")
    manifest = parse_manifest(manifest_path)
    counts = manifest["counts"]
    print(f"  {manifest['total']:,} patches total | {len(manifest['tiles'])} tile(s)")
    print(f"  train={counts['train']:,}  val={counts['val']:,}  test={counts['test']:,}")

    # 2. Eval metrics
    print(f"\n[2/3] Métricas (exp_{exp}, split={args.split})")
    eval_log = ROOT / "experiments" / f"exp_{exp}" / "logs" / f"eval_patches_{args.split}.txt"
    if args.no_eval and eval_log.exists():
        print(f"  Usando log existente: {eval_log}")
        raw = eval_log.read_text(encoding="utf-8")
    else:
        raw = run_eval(exp, args.split, patches_dir)
    metrics = parse_eval_output(raw)
    lc = metrics.get("leucaena", {})
    print(f"  leucaena → P={lc.get('precision','?')}  R={lc.get('recall','?')}  F1={lc.get('f1','?')}")

    # 3. Training curves
    if not args.no_plot:
        print(f"\n[3/4] Curvas de treino")
        run_plot_training(exp)
    else:
        print("\n[3/4] Curvas de treino — pulado (--no-plot)")

    # 4. Patch panels: 1 dense + 1 hard (figures 5.1 and 5.2)
    print(f"\n[4/4] Painéis de patch (densa + difícil)")
    run_viz_patches(patches_dir)
    print("  Figs 5.3 e 5.4 (predição boa/difícil) precisam do modelo treinado:")
    print(f"    python inspect_validation_errors.py -e {exp} --split val --top-k 1 --rank-by f1")
    print(f"    python inspect_validation_errors.py -e {exp} --split val --top-k 1 --rank-by fp")

    # 6. Fill template
    print(f"\nPreenchendo {TEMPLATE_PATH} ...")
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    filled = fill_template(template, manifest, metrics)
    TEMPLATE_PATH.write_text(filled, encoding="utf-8")

    remaining = filled.count("[PREENCHER")
    print(f"  Pronto! {remaining} campo(s) restam com [PREENCHER] (orientador, data, próximos passos do professor).")
    print(f"\nQuando quiser o PDF, fale 'converte pra PDF'.")


if __name__ == "__main__":
    main()
