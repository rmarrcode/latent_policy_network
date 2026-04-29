from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_COLUMNS = [
    "agent",
    "encoder",
    "seed",
    "eval_return_mean",
    "eval_reward_age_0_3",
    "eval_reward_age_4_15",
    "eval_reward_age_16_plus",
    "eval_win_rate",
    "params",
    "run_dir",
]


def summarize(path: str | Path, top_k: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(path)
    present = [col for col in DEFAULT_COLUMNS if col in df.columns]
    ranked = df[present].sort_values("eval_return_mean", ascending=False).head(top_k)
    grouped = (
        df.groupby(["agent", "encoder"])[
            [
                "eval_return_mean",
                "eval_reward_age_0_3",
                "eval_reward_age_4_15",
                "eval_reward_age_16_plus",
                "eval_win_rate",
            ]
        ]
        .mean()
        .sort_values("eval_return_mean", ascending=False)
    )
    return ranked, grouped


def to_markdown_table(df: pd.DataFrame, index: bool = False) -> str:
    if index:
        df = df.reset_index()
    rows = [[str(col) for col in df.columns]]
    for _, row in df.iterrows():
        rows.append([str(value) for value in row.tolist()])
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows[1:]]
    return "\n".join([header, sep, *body])


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize latent-policy sweep CSVs.")
    parser.add_argument("summary", type=str)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--markdown", type=str, default=None)
    args = parser.parse_args()

    ranked, grouped = summarize(args.summary, top_k=args.top_k)
    print("Ranked runs:")
    print(ranked.to_string(index=False))
    print("\nGrouped means:")
    print(grouped.round(3).to_string())

    if args.markdown:
        out = Path(args.markdown)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            "# Sweep Summary\n\n"
            "## Ranked Runs\n\n"
            + to_markdown_table(ranked, index=False)
            + "\n\n## Grouped Means\n\n"
            + to_markdown_table(grouped.round(3), index=True)
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
