"""
Plot training curves from ``experiments/exp_<N>/logs/metrics.csv``.

Examples
--------
All epochs::

    python -m utils.plot_training -e 1

Up to epoch 10 only (useful while training is still running)::

    python -m utils.plot_training -e 1 --upto-epoch 10

Print numeric table only::

    python -m utils.plot_training -e 1 --table
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt

from conf import paths
from utils.training_log import load_metrics

METRICS_NAME = 'metrics.csv'
PLOT_NAME = 'training_curves.png'


def _default_metrics_path(experiment: int) -> str:
    return os.path.join(paths.PATH_EXPERIMENTS, f'exp_{experiment}', 'logs', METRICS_NAME)


def _default_plot_path(experiment: int) -> str:
    return os.path.join(paths.PATH_EXPERIMENTS, f'exp_{experiment}', 'logs', PLOT_NAME)


def _filter_rows(rows: list[dict], upto_epoch: int | None) -> list[dict]:
    if upto_epoch is None:
        return rows
    return [r for r in rows if r['epoch'] <= upto_epoch]


def print_table(rows: list[dict]) -> None:
    header = (
        f"{'ep':>4}  {'train_L':>9}  {'train_F1':>9}  "
        f"{'val_L':>9}  {'val_F1':>9}  {'lr':>10}  {'saved':>5}"
    )
    print(header)
    print('-' * len(header))
    for r in rows:
        print(
            f"{r['epoch']:4d}  "
            f"{r['train_loss']:9.4f}  {r['train_f1']:9.4f}  "
            f"{r['val_loss']:9.4f}  {r['val_f1']:9.4f}  "
            f"{r['learning_rate']:10.2e}  "
            f"{'yes' if r['checkpoint_saved'] else 'no':>5}"
        )


def plot_metrics(
    rows: list[dict],
    output_path: str,
    *,
    show: bool = False,
    title_suffix: str = '',
) -> str:
    epochs = [r['epoch'] for r in rows]
    train_loss = [r['train_loss'] for r in rows]
    val_loss = [r['val_loss'] for r in rows]
    train_f1 = [r['train_f1'] for r in rows]
    val_f1 = [r['val_f1'] for r in rows]
    lr = [r['learning_rate'] for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    title = 'Training curves'
    if title_suffix:
        title = f'{title} ({title_suffix})'
    fig.suptitle(title)

    ax = axes[0]
    ax.plot(epochs, train_loss, 'o-', label='train', color='#2563eb', markersize=4)
    ax.plot(epochs, val_loss, 'o-', label='val', color='#dc2626', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Loss')

    ax = axes[1]
    ax.plot(epochs, train_f1, 'o-', label='train F1', color='#2563eb', markersize=4)
    ax.plot(epochs, val_f1, 'o-', label='val F1', color='#dc2626', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1')
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('F1 (leucaena)')

    ax = axes[2]
    ax.plot(epochs, lr, 'o-', color='#059669', markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning rate')
    ax.grid(True, alpha=0.3)
    ax.set_title('LR')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)
    return output_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='Plot metrics.csv from a training run.')
    p.add_argument('-e', '--experiment', type=int, default=1, help='Experiment id (default: 1)')
    p.add_argument(
        '--metrics',
        type=str,
        default=None,
        help=f'Path to metrics.csv (default: experiments/exp_<e>/logs/{METRICS_NAME})',
    )
    p.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help=f'Output PNG (default: experiments/exp_<e>/logs/{PLOT_NAME})',
    )
    p.add_argument(
        '--upto-epoch',
        type=int,
        default=None,
        help='Plot only epochs 1..N (refresh while training is in progress)',
    )
    p.add_argument('--table', action='store_true', help='Print epoch table and skip plot')
    p.add_argument('--show', action='store_true', help='Open interactive window (if display available)')
    args = p.parse_args(argv)

    metrics_path = args.metrics or _default_metrics_path(args.experiment)
    output_path = args.output or _default_plot_path(args.experiment)

    rows = load_metrics(metrics_path)
    rows = _filter_rows(rows, args.upto_epoch)
    if not rows:
        print(f'No epochs to plot (upto-epoch={args.upto_epoch}).', file=sys.stderr)
        return 1

    print_table(rows)

    if args.table:
        return 0

    suffix = f'epochs 1–{rows[-1]["epoch"]}'
    out = plot_metrics(rows, output_path, show=args.show, title_suffix=suffix)
    print(f'\nSaved plot -> {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
