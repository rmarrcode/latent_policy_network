from __future__ import annotations

import csv

from latent_policy.training_progress import collect_progress, expected_run_names, format_progress


def test_expected_run_names_from_experiment_config():
    names = expected_run_names(
        {
            "seeds": "1 2",
            "agent_encoders": "static_mlp:mean full_hyper:gru",
        }
    )

    assert names == [
        "gpu_elo_static_mlp_mean_seed1",
        "gpu_elo_full_hyper_gru_seed1",
        "gpu_elo_static_mlp_mean_seed2",
        "gpu_elo_full_hyper_gru_seed2",
    ]


def test_collect_progress_reports_complete_partial_and_missing(tmp_path):
    base = tmp_path / "run"
    train = base / "train"
    logs = base / "logs"
    complete = train / "gpu_elo_static_mlp_mean_seed1"
    partial = train / "gpu_elo_full_hyper_gru_seed1"
    (complete / "checkpoints").mkdir(parents=True)
    partial.mkdir(parents=True)
    logs.mkdir(parents=True)
    (base / "experiment_config.txt").write_text(
        "\n".join(
            [
                "total_updates=120",
                "seeds=1",
                "agent_encoders=static_mlp:mean full_hyper:gru film:gru",
            ]
        ),
        encoding="utf-8",
    )
    (complete / "final_metrics.json").write_text("{}", encoding="utf-8")
    (complete / "checkpoints" / "update_0120.pt").write_text("checkpoint", encoding="utf-8")
    _write_metrics(complete / "metrics.csv", update=120, ret=-0.25, reward=-0.01, skipped=0, repaired=0)
    _write_metrics(partial / "metrics.csv", update=99, ret=-0.5, reward=-0.02, skipped=2, repaired=1)
    (logs / "gpu_elo_full_hyper_gru_seed1_retry_2_20260505_120000.log").write_text("retry", encoding="utf-8")

    rows = collect_progress(base)
    by_run = {row.run: row for row in rows}

    assert by_run["gpu_elo_static_mlp_mean_seed1"].status == "complete"
    assert by_run["gpu_elo_static_mlp_mean_seed1"].percent == 100.0
    assert by_run["gpu_elo_full_hyper_gru_seed1"].status == "partial"
    assert by_run["gpu_elo_full_hyper_gru_seed1"].update == 99
    assert by_run["gpu_elo_full_hyper_gru_seed1"].skipped_minibatches == 2
    assert by_run["gpu_elo_film_gru_seed1"].status == "missing"

    output = format_progress(rows, bar_width=10)
    assert "gpu_elo_full_hyper_gru_seed1" in output
    assert "partial" in output
    assert "[##########]" in output
    assert "[########--]" in output
    assert "82.5" in output


def _write_metrics(path, update: int, ret: float, reward: float, skipped: int, repaired: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "update",
                "rollout_return_recent",
                "rollout_reward_mean",
                "skipped_minibatches",
                "repaired_parameter_tensors",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "update": update,
                "rollout_return_recent": ret,
                "rollout_reward_mean": reward,
                "skipped_minibatches": skipped,
                "repaired_parameter_tensors": repaired,
            }
        )
