from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProgressRow:
    run: str
    status: str
    update: int | None
    total_updates: int | None
    rollout_return_recent: float | None
    rollout_reward_mean: float | None
    skipped_minibatches: int | None
    repaired_parameter_tensors: int | None
    log_path: Path | None

    @property
    def percent(self) -> float | None:
        if self.update is None or not self.total_updates:
            return None
        return min(100.0, 100.0 * self.update / self.total_updates)


def read_experiment_config(base_dir: str | Path) -> dict[str, str]:
    path = Path(base_dir) / "experiment_config.txt"
    if not path.exists():
        return {}
    config: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in raw_line:
            continue
        key, value = raw_line.split("=", 1)
        config[key.strip()] = value.strip()
    return config


def expected_run_names(config: dict[str, str]) -> list[str]:
    seeds = config.get("seeds", "").split()
    pairs = config.get("agent_encoders", "").split()
    if not seeds or not pairs:
        return []
    names: list[str] = []
    for seed in seeds:
        for pair in pairs:
            if ":" not in pair:
                continue
            agent, encoder = pair.split(":", 1)
            names.append(f"gpu_elo_{agent}_{encoder}_seed{seed}")
    return names


def _last_csv_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    last: dict[str, str] | None = None
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            last = row
    return last


def _to_int(value: str | None) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _to_float(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _latest_log(log_dir: Path, run_name: str) -> Path | None:
    candidates = [log_dir / f"{run_name}.log"]
    candidates.extend(sorted(log_dir.glob(f"{run_name}_retry_*.log")))
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None
    return max(existing, key=lambda path: path.stat().st_mtime)


def collect_progress(base_dir: str | Path, total_updates: int | None = None) -> list[ProgressRow]:
    base = Path(base_dir)
    train_dir = base / "train"
    log_dir = base / "logs"
    config = read_experiment_config(base)
    if total_updates is None:
        total_updates = _to_int(config.get("total_updates"))

    expected = expected_run_names(config)
    run_dirs = {path.name: path for path in train_dir.iterdir() if path.is_dir()} if train_dir.exists() else {}
    expected_set = set(expected)
    names = expected + sorted(name for name in run_dirs if name not in expected_set)

    rows: list[ProgressRow] = []
    for name in names:
        run_path = run_dirs.get(name, train_dir / name)
        metrics = _last_csv_row(run_path / "metrics.csv")
        update = _to_int(metrics.get("update")) if metrics is not None else None
        final_checkpoint = (
            run_path / "checkpoints" / f"update_{total_updates:04d}.pt" if total_updates is not None else None
        )
        if not run_path.exists():
            status = "missing"
        elif (run_path / "final_metrics.json").exists() and (
            final_checkpoint is None or final_checkpoint.exists()
        ):
            status = "complete"
        else:
            status = "partial"

        rows.append(
            ProgressRow(
                run=name,
                status=status,
                update=update,
                total_updates=total_updates,
                rollout_return_recent=_to_float(metrics.get("rollout_return_recent")) if metrics is not None else None,
                rollout_reward_mean=_to_float(metrics.get("rollout_reward_mean")) if metrics is not None else None,
                skipped_minibatches=_to_int(metrics.get("skipped_minibatches")) if metrics is not None else None,
                repaired_parameter_tensors=(
                    _to_int(metrics.get("repaired_parameter_tensors")) if metrics is not None else None
                ),
                log_path=_latest_log(log_dir, name),
            )
        )
    return rows


def _format_optional(value: object, precision: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{precision}f}"
    return str(value)


def format_progress_bar(percent: float | None, width: int = 24) -> str:
    width = max(1, width)
    if percent is None:
        return f"[{'-' * width}]"
    bounded = min(100.0, max(0.0, percent))
    filled = round(width * bounded / 100.0)
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def format_progress(rows: list[ProgressRow], bar_width: int = 24) -> str:
    headers = ["run", "status", "update", "progress", "pct", "return", "reward", "skipped", "repaired", "log"]
    body: list[list[str]] = []
    for row in rows:
        update = "-" if row.update is None else str(row.update)
        if row.total_updates is not None and row.update is not None:
            update = f"{row.update}/{row.total_updates}"
        body.append(
            [
                row.run,
                row.status,
                update,
                format_progress_bar(row.percent, bar_width),
                _format_optional(row.percent, precision=1),
                _format_optional(row.rollout_return_recent),
                _format_optional(row.rollout_reward_mean),
                _format_optional(row.skipped_minibatches, precision=0),
                _format_optional(row.repaired_parameter_tensors, precision=0),
                "-" if row.log_path is None else str(row.log_path),
            ]
        )
    widths = [len(header) for header in headers]
    for cells in body:
        widths = [max(width, len(cell)) for width, cell in zip(widths, cells)]
    lines = ["  ".join(header.ljust(width) for header, width in zip(headers, widths))]
    lines.append("  ".join("-" * width for width in widths))
    lines.extend("  ".join(cell.ljust(width) for cell, width in zip(cells, widths)) for cells in body)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize legacy GPU Elo training progress.")
    parser.add_argument("base_dir", help="Experiment base directory containing train/ and logs/")
    parser.add_argument("--total-updates", type=int, default=None)
    parser.add_argument("--bar-width", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_progress(args.base_dir, total_updates=args.total_updates)
    if not rows:
        print(f"No training runs found under {args.base_dir}")
        return
    print(format_progress(rows, bar_width=args.bar_width))


if __name__ == "__main__":
    main()
