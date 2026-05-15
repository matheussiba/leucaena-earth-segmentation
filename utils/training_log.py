"""
Structured training metrics for plots and post-hoc analysis.

Writes ``metrics.csv`` (one row per epoch, appended live) and ``training_config.json``.
"""
from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone
from typing import Any

METRICS_FIELDS = (
    'epoch',
    'train_loss',
    'train_f1',
    'val_loss',
    'val_f1',
    'learning_rate',
    'best_val_loss',
    'epochs_no_improve',
    'checkpoint_saved',
    'elapsed_sec',
)


class MetricsLogger:
    """Append per-epoch metrics to CSV so plots can be refreshed during training."""

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_path = os.path.join(log_dir, 'metrics.csv')
        self.config_path = os.path.join(log_dir, 'training_config.json')
        self._t0 = time.perf_counter()
        self._file = open(self.metrics_path, 'w', newline='', encoding='utf-8')
        self._writer = csv.DictWriter(self._file, fieldnames=METRICS_FIELDS)
        self._writer.writeheader()
        self._file.flush()

    def write_config(self, config: dict[str, Any]) -> None:
        payload = {
            'started_at_utc': datetime.now(timezone.utc).isoformat(),
            **config,
        }
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)

    def log_epoch(
        self,
        *,
        epoch: int,
        train_loss: float,
        train_f1: float,
        val_loss: float,
        val_f1: float,
        learning_rate: float,
        best_val_loss: float | None,
        epochs_no_improve: int,
        checkpoint_saved: bool,
    ) -> None:
        row = {
            'epoch': epoch,
            'train_loss': f'{train_loss:.6f}',
            'train_f1': f'{train_f1:.6f}',
            'val_loss': f'{val_loss:.6f}',
            'val_f1': f'{val_f1:.6f}',
            'learning_rate': f'{learning_rate:.8e}',
            'best_val_loss': '' if best_val_loss is None else f'{best_val_loss:.6f}',
            'epochs_no_improve': epochs_no_improve,
            'checkpoint_saved': int(checkpoint_saved),
            'elapsed_sec': f'{time.perf_counter() - self._t0:.1f}',
        }
        self._writer.writerow(row)
        self._file.flush()
        print(
            f'[metrics] epoch={epoch} '
            f'train_loss={train_loss:.4f} train_f1={train_f1:.4f} '
            f'val_loss={val_loss:.4f} val_f1={val_f1:.4f} '
            f'lr={learning_rate:.2e} saved={checkpoint_saved}'
        )

    def close(self) -> None:
        self._file.close()


def load_metrics(metrics_path: str) -> list[dict[str, Any]]:
    """Load ``metrics.csv`` written by :class:`MetricsLogger`."""
    if not os.path.isfile(metrics_path):
        raise FileNotFoundError(f'Metrics file not found: {metrics_path}')
    with open(metrics_path, newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f'No rows in {metrics_path}')
    out = []
    for r in rows:
        out.append({
            'epoch': int(r['epoch']),
            'train_loss': float(r['train_loss']),
            'train_f1': float(r['train_f1']),
            'val_loss': float(r['val_loss']),
            'val_f1': float(r['val_f1']),
            'learning_rate': float(r['learning_rate']),
            'best_val_loss': float(r['best_val_loss']) if r['best_val_loss'] else None,
            'epochs_no_improve': int(r['epochs_no_improve']),
            'checkpoint_saved': bool(int(r['checkpoint_saved'])),
            'elapsed_sec': float(r['elapsed_sec']),
        })
    return out
