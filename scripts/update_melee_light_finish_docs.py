from __future__ import annotations

import argparse
import csv
from datetime import date
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _fmt_float(value: str, digits: int = 1) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return value


def _top_agent_update_rows(path: Path, limit: int = 12) -> list[dict[str, str]]:
    rows = _read_csv(path)
    return sorted(rows, key=lambda row: float(row.get("elo_mean", "0")), reverse=True)[:limit]


def _top_competitor_rows(path: Path, limit: int = 12) -> list[dict[str, str]]:
    rows = _read_csv(path)
    return sorted(rows, key=lambda row: float(row.get("elo", "0")), reverse=True)[:limit]


def _experiment_config(run_dir: Path) -> dict[str, str]:
    config: dict[str, str] = {}
    path = run_dir / "experiment_config.txt"
    if not path.exists():
        return config
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        config[key.strip()] = value.strip()
    return config


def _build_experiment_section(run_dir: Path) -> str:
    elo_dir = run_dir / "elo"
    config = _experiment_config(run_dir)
    top_competitors = _top_competitor_rows(elo_dir / "elo.csv")
    top_agent_updates = _top_agent_update_rows(elo_dir / "agent_update_summary.csv")
    pair_results = _read_csv(elo_dir / "pair_results.csv")
    competitors = _read_csv(elo_dir / "competitors.csv")

    lines = [
        f"<!-- melee-light-finish:{run_dir.name} -->",
        f"## {date.today().isoformat()} Melee Light Self-Play GPU Elo Completion",
        "",
        f"Run directory: `{run_dir}`",
        "",
        "This run used external P2 control only. The Elo tournament rated",
        "`agent x checkpoint_update x character` competitors and preserved policy",
        "context across warmup plus scored games in each side-order series.",
        "",
        "Configuration:",
        "",
        f"- agents: `{config.get('agent_encoders', 'unknown')}`",
        f"- seeds: `{config.get('seeds', 'unknown')}`",
        f"- total updates: `{config.get('total_updates', 'unknown')}`",
        f"- checkpoint interval: `{config.get('save_interval', 'unknown')}`",
        f"- characters: `{config.get('characters', 'unknown')}`",
        f"- warmup games per side: `{config.get('warmup_games', 'unknown')}`",
        f"- scored games per side: `{config.get('scored_games', 'unknown')}`",
        f"- selected pairings: `{len(pair_results)}`",
        f"- competitors: `{len(competitors)}`",
        "",
        "Top agent/update means:",
        "",
        "| rank | agent | encoder | update | competitors | Elo mean | Elo std |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(top_agent_updates, start=1):
        lines.append(
            "| "
            f"{rank} | `{row.get('agent', '')}` | `{row.get('encoder', '')}` | "
            f"{row.get('update', '')} | {row.get('competitors', '')} | "
            f"{_fmt_float(row.get('elo_mean', ''))} | {_fmt_float(row.get('elo_std', ''))} |"
        )

    lines.extend(
        [
            "",
            "Top individual competitors:",
            "",
            "| rank | Elo | agent | encoder | seed | update | character | scored games | score rate |",
            "|---:|---:|---|---|---:|---:|---|---:|---:|",
        ]
    )
    for rank, row in enumerate(top_competitors, start=1):
        lines.append(
            "| "
            f"{rank} | {_fmt_float(row.get('elo', ''))} | `{row.get('agent', '')}` | "
            f"`{row.get('encoder', '')}` | {row.get('seed', '')} | {row.get('update', '')} | "
            f"`{row.get('character_name', '')}` | {row.get('scored_games', '')} | "
            f"{_fmt_float(row.get('score_rate', ''), 3)} |"
        )

    lines.extend(["", f"Full Elo report: `{elo_dir / 'report.md'}`", ""])
    return "\n".join(lines)


def _build_readme_section(run_dir: Path) -> str:
    elo_dir = run_dir / "elo"
    top_agent_updates = _top_agent_update_rows(elo_dir / "agent_update_summary.csv", limit=5)
    lines = [
        f"<!-- melee-light-finish-readme:{run_dir.name} -->",
        "### Latest Melee Light GPU Elo Completion",
        "",
        f"Latest completed run: `{run_dir}`.",
        "",
        "| agent | encoder | update | Elo mean |",
        "|---|---|---:|---:|",
    ]
    for row in top_agent_updates:
        lines.append(
            f"| `{row.get('agent', '')}` | `{row.get('encoder', '')}` | "
            f"{row.get('update', '')} | {_fmt_float(row.get('elo_mean', ''))} |"
        )
    lines.extend(["", f"Full report: `{elo_dir / 'report.md'}`", ""])
    return "\n".join(lines)


def _append_once(path: Path, marker: str, section: str) -> bool:
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in text:
        return False
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text + "\n" + section, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Append completed Melee Light Elo results to markdown docs.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir
    if not (run_dir / "elo" / "report.md").exists():
        raise SystemExit(f"missing Elo report: {run_dir / 'elo' / 'report.md'}")

    root = Path(__file__).resolve().parents[1]
    exp_section = _build_experiment_section(run_dir)
    readme_section = _build_readme_section(run_dir)
    _append_once(root / "EXPERIMENTS.md", f"<!-- melee-light-finish:{run_dir.name} -->", exp_section)
    _append_once(root / "README.md", f"<!-- melee-light-finish-readme:{run_dir.name} -->", readme_section)
    print(f"updated docs for {run_dir}")


if __name__ == "__main__":
    main()
