from __future__ import annotations

import argparse
import csv
import glob
import itertools
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from latent_policy.evaluation import append_context
from latent_policy.melee_light_env import MeleeLightKnockbackEnv, OBS_DIM
from latent_policy.ppo import config_from_dict, load_checkpoint
from latent_policy.public_envs import MELEE_LIGHT_CHARACTER_NAMES
from latent_policy.utils import select_device


CHARACTER_IDS = {name: idx for idx, name in MELEE_LIGHT_CHARACTER_NAMES.items()}
CHARACTER_IDS.update({str(idx): idx for idx in MELEE_LIGHT_CHARACTER_NAMES})


@dataclass(frozen=True)
class CheckpointInfo:
    path: str
    run_name: str
    agent: str
    encoder: str
    seed: int
    update: int


@dataclass(frozen=True)
class Competitor:
    id: str
    checkpoint_path: str
    run_name: str
    agent: str
    encoder: str
    seed: int
    update: int
    character: int
    character_name: str


@dataclass
class PairResult:
    pair_index: int
    competitor_a: str
    competitor_b: str
    a_agent: str
    a_encoder: str
    a_seed: int
    a_update: int
    a_character: str
    b_agent: str
    b_encoder: str
    b_seed: int
    b_update: int
    b_character: str
    warmup_games_per_side: int
    scored_games_per_side: int
    scored_games_total: int
    a_wins: int
    b_wins: int
    draws: int
    a_score: float
    avg_scored_length: float
    p1_a_wins: int
    p1_b_wins: int
    p2_a_wins: int
    p2_b_wins: int
    elo_a_before: float
    elo_b_before: float
    elo_a_after: float
    elo_b_after: float


def parse_character_ids(value: str) -> list[int]:
    out: list[int] = []
    for item in value.split(","):
        key = item.strip().lower()
        if not key:
            continue
        if key not in CHARACTER_IDS:
            valid = ", ".join(sorted(CHARACTER_IDS))
            raise ValueError(f"unknown character {item!r}; valid values: {valid}")
        out.append(int(CHARACTER_IDS[key]))
    if not out:
        raise ValueError("at least one character is required")
    return out


def _raw_obs_for_side(raw_obs: np.ndarray, side: int) -> np.ndarray:
    raw = np.nan_to_num(np.asarray(raw_obs, dtype=np.float32).reshape(-1), nan=0.0, posinf=1e6, neginf=-1e6)
    if side == 0:
        return raw.copy()
    if raw.size < OBS_DIM:
        raise ValueError(f"expected at least {OBS_DIM} Melee Light obs values, got {raw.size}")
    swapped = np.empty_like(raw)
    swapped[:14] = raw[14:28]
    swapped[14:28] = raw[:14]
    swapped[28] = -raw[28]
    swapped[29] = -raw[29]
    if raw.size > OBS_DIM:
        swapped[OBS_DIM:] = raw[OBS_DIM:]
    return swapped


def _policy_obs(
    raw_obs: np.ndarray,
    side: int,
    last_reward: float,
    step_count: int,
    episode_length: int,
    obs_dim: int,
) -> np.ndarray:
    side_obs = _raw_obs_for_side(raw_obs, side)
    if obs_dim == side_obs.size:
        return side_obs.astype(np.float32)
    if obs_dim == side_obs.size + 2:
        extra = np.asarray([last_reward, step_count / max(1, episode_length)], dtype=np.float32)
        return np.concatenate([side_obs, extra]).astype(np.float32)
    raise ValueError(f"checkpoint expects obs_dim={obs_dim}, but Melee Light side obs has {side_obs.size} values")


def _expected_score(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _elo_update(rating_a: float, rating_b: float, score_a: float, k_factor: float) -> tuple[float, float]:
    expected_a = _expected_score(rating_a, rating_b)
    delta = k_factor * (score_a - expected_a)
    return rating_a + delta, rating_b - delta


def _checkpoint_info(path: Path) -> CheckpointInfo:
    ckpt = torch.load(path, map_location="cpu")
    cfg = config_from_dict(ckpt["config"])
    run_name = path.parent.parent.name if path.parent.name == "checkpoints" else path.parent.name
    return CheckpointInfo(
        path=str(path),
        run_name=run_name,
        agent=str(cfg.policy.agent),
        encoder=str(cfg.policy.encoder),
        seed=int(cfg.seed),
        update=int(ckpt.get("update", 0)),
    )


def _load_checkpoint_infos(paths: list[Path]) -> list[CheckpointInfo]:
    deduped = sorted({str(path.resolve()): path for path in paths}.values())
    infos = [_checkpoint_info(path) for path in deduped]
    return sorted(infos, key=lambda item: (item.agent, item.encoder, item.seed, item.update, item.run_name, item.path))


def _expand_competitors(checkpoints: list[CheckpointInfo], characters: list[int]) -> list[Competitor]:
    competitors: list[Competitor] = []
    for ckpt in checkpoints:
        for character in characters:
            character_name = MELEE_LIGHT_CHARACTER_NAMES.get(character, str(character))
            competitor_id = (
                f"{ckpt.agent}:{ckpt.encoder}:seed{ckpt.seed}:u{ckpt.update}:"
                f"{character_name}:{Path(ckpt.path).parent.parent.name if Path(ckpt.path).parent.name == 'checkpoints' else Path(ckpt.path).parent.name}"
            )
            competitors.append(
                Competitor(
                    id=competitor_id,
                    checkpoint_path=ckpt.path,
                    run_name=ckpt.run_name,
                    agent=ckpt.agent,
                    encoder=ckpt.encoder,
                    seed=ckpt.seed,
                    update=ckpt.update,
                    character=character,
                    character_name=character_name,
                )
            )
    return competitors


def _select_pairings(
    competitors: list[Competitor],
    rng: random.Random,
    max_pairings: int | None,
    min_pairings_per_competitor: int,
) -> list[tuple[Competitor, Competitor]]:
    all_pairs = list(itertools.combinations(competitors, 2))
    rng.shuffle(all_pairs)
    if max_pairings is None or max_pairings >= len(all_pairs):
        return all_pairs

    selected: list[tuple[Competitor, Competitor]] = []
    selected_keys: set[tuple[str, str]] = set()
    coverage = {competitor.id: 0 for competitor in competitors}

    def key_for(pair: tuple[Competitor, Competitor]) -> tuple[str, str]:
        return tuple(sorted((pair[0].id, pair[1].id)))

    def add_pair(pair: tuple[Competitor, Competitor]) -> bool:
        key = key_for(pair)
        if key in selected_keys:
            return False
        selected.append(pair)
        selected_keys.add(key)
        coverage[pair[0].id] += 1
        coverage[pair[1].id] += 1
        return True

    while len(selected) < max_pairings and min(coverage.values(), default=0) < min_pairings_per_competitor:
        remaining = [pair for pair in all_pairs if key_for(pair) not in selected_keys]
        if not remaining:
            break
        remaining.sort(key=lambda pair: (coverage[pair[0].id] + coverage[pair[1].id], rng.random()))
        add_pair(remaining[0])

    for pair in all_pairs:
        if len(selected) >= max_pairings:
            break
        add_pair(pair)
    return selected


class LoadedPolicy:
    def __init__(self, checkpoint_path: str, device: torch.device):
        self.policy, self.cfg, self.policy_cfg, self.ckpt = load_checkpoint(checkpoint_path, device)
        self.policy.eval()
        self.device = device


class RuntimePolicy:
    def __init__(self, loaded: LoadedPolicy):
        self.loaded = loaded
        self.context = torch.zeros(
            (1, loaded.policy_cfg.context_len, loaded.policy_cfg.obs_dim),
            dtype=torch.float32,
            device=loaded.device,
        )

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool) -> int:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.loaded.device).view(1, -1)
        action = self.loaded.policy.act(obs_t, self.context, deterministic=deterministic)
        return int(action.item())

    def observe(self, obs: np.ndarray) -> None:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.loaded.device).view(1, -1)
        self.context = append_context(self.context, obs_t, done=None)


class PolicyCache:
    def __init__(self, device: torch.device):
        self.device = device
        self._cache: dict[str, LoadedPolicy] = {}

    def get(self, checkpoint_path: str) -> LoadedPolicy:
        checkpoint_path = str(Path(checkpoint_path).resolve())
        if checkpoint_path not in self._cache:
            self._cache[checkpoint_path] = LoadedPolicy(checkpoint_path, self.device)
        return self._cache[checkpoint_path]


def _run_game(
    env: MeleeLightKnockbackEnv,
    p1: RuntimePolicy,
    p2: RuntimePolicy,
    p1_character: int,
    p2_character: int,
    episode_length: int,
    deterministic: bool,
) -> tuple[int, int]:
    raw_obs, _ = env.reset(options={"agent_character": p1_character, "opponent_character": p2_character})
    p1_obs = _policy_obs(raw_obs, 0, 0.0, 0, episode_length, p1.loaded.policy_cfg.obs_dim)
    p2_obs = _policy_obs(raw_obs, 1, 0.0, 0, episode_length, p2.loaded.policy_cfg.obs_dim)
    step_count = 0

    while True:
        action_1 = p1.act(p1_obs, deterministic=deterministic)
        action_2 = p2.act(p2_obs, deterministic=deterministic)
        raw_next, reward, terminated, truncated, info = env.step(action_1, opponent_action=action_2)
        step_count += 1
        done = bool(terminated or truncated)
        p1_next = _policy_obs(raw_next, 0, float(reward), step_count, episode_length, p1.loaded.policy_cfg.obs_dim)
        p2_next = _policy_obs(raw_next, 1, float(-reward), step_count, episode_length, p2.loaded.policy_cfg.obs_dim)
        p1.observe(p1_next)
        p2.observe(p2_next)
        p1_obs = p1_next
        p2_obs = p2_next

        if done:
            winner = int(info.get("winner", 0 if reward > 0 else 1))
            if reward == 0.0 and "winner" not in info:
                winner = -1
            return winner, step_count

        if step_count >= episode_length:
            return 1, step_count


def _run_side_series(
    env: MeleeLightKnockbackEnv,
    cache: PolicyCache,
    p1_competitor: Competitor,
    p2_competitor: Competitor,
    warmup_games: int,
    scored_games: int,
    episode_length: int,
    deterministic: bool,
) -> dict[str, Any]:
    p1 = RuntimePolicy(cache.get(p1_competitor.checkpoint_path))
    p2 = RuntimePolicy(cache.get(p2_competitor.checkpoint_path))
    p1_wins = 0
    p2_wins = 0
    draws = 0
    scored_lengths: list[int] = []
    total_games = warmup_games + scored_games

    for game_idx in range(total_games):
        winner, length = _run_game(
            env,
            p1,
            p2,
            p1_competitor.character,
            p2_competitor.character,
            episode_length,
            deterministic,
        )
        if game_idx < warmup_games:
            continue
        scored_lengths.append(length)
        if winner == 0:
            p1_wins += 1
        elif winner == 1:
            p2_wins += 1
        else:
            draws += 1

    return {
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "draws": draws,
        "lengths": scored_lengths,
    }


def _run_pairing(
    pair_index: int,
    env: MeleeLightKnockbackEnv,
    cache: PolicyCache,
    competitor_a: Competitor,
    competitor_b: Competitor,
    ratings: dict[str, float],
    k_factor: float,
    warmup_games: int,
    scored_games: int,
    episode_length: int,
    deterministic: bool,
) -> PairResult:
    elo_a_before = ratings[competitor_a.id]
    elo_b_before = ratings[competitor_b.id]

    a_p1 = _run_side_series(
        env,
        cache,
        competitor_a,
        competitor_b,
        warmup_games,
        scored_games,
        episode_length,
        deterministic,
    )
    b_p1 = _run_side_series(
        env,
        cache,
        competitor_b,
        competitor_a,
        warmup_games,
        scored_games,
        episode_length,
        deterministic,
    )

    a_wins = int(a_p1["p1_wins"] + b_p1["p2_wins"])
    b_wins = int(a_p1["p2_wins"] + b_p1["p1_wins"])
    draws = int(a_p1["draws"] + b_p1["draws"])
    scored_games_total = max(1, 2 * scored_games)
    a_score = (a_wins + 0.5 * draws) / scored_games_total
    elo_a_after, elo_b_after = _elo_update(elo_a_before, elo_b_before, a_score, k_factor)
    ratings[competitor_a.id] = elo_a_after
    ratings[competitor_b.id] = elo_b_after
    lengths = list(a_p1["lengths"]) + list(b_p1["lengths"])

    return PairResult(
        pair_index=pair_index,
        competitor_a=competitor_a.id,
        competitor_b=competitor_b.id,
        a_agent=competitor_a.agent,
        a_encoder=competitor_a.encoder,
        a_seed=competitor_a.seed,
        a_update=competitor_a.update,
        a_character=competitor_a.character_name,
        b_agent=competitor_b.agent,
        b_encoder=competitor_b.encoder,
        b_seed=competitor_b.seed,
        b_update=competitor_b.update,
        b_character=competitor_b.character_name,
        warmup_games_per_side=warmup_games,
        scored_games_per_side=scored_games,
        scored_games_total=scored_games_total,
        a_wins=a_wins,
        b_wins=b_wins,
        draws=draws,
        a_score=a_score,
        avg_scored_length=float(np.mean(lengths)) if lengths else float("nan"),
        p1_a_wins=int(a_p1["p1_wins"]),
        p1_b_wins=int(a_p1["p2_wins"]),
        p2_a_wins=int(b_p1["p2_wins"]),
        p2_b_wins=int(b_p1["p1_wins"]),
        elo_a_before=elo_a_before,
        elo_b_before=elo_b_before,
        elo_a_after=elo_a_after,
        elo_b_after=elo_b_after,
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(groups: dict[tuple[Any, ...], list[float]], field_names: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, values in sorted(groups.items()):
        arr = np.asarray(values, dtype=np.float64)
        row = {field: key[idx] for idx, field in enumerate(field_names)}
        row.update(
            {
                "competitors": int(arr.size),
                "elo_mean": float(arr.mean()),
                "elo_std": float(arr.std()) if arr.size > 1 else 0.0,
                "elo_min": float(arr.min()),
                "elo_max": float(arr.max()),
            }
        )
        rows.append(row)
    return rows


def _write_report(
    path: Path,
    args: argparse.Namespace,
    competitors: list[Competitor],
    pair_results: list[PairResult],
    elo_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
) -> None:
    top = sorted(elo_rows, key=lambda row: row["elo"], reverse=True)[:20]
    lines = [
        "# Melee Light Elo Tournament",
        "",
        f"- checkpoints: `{len({c.checkpoint_path for c in competitors})}`",
        f"- competitors: `{len(competitors)}`",
        f"- pairings: `{len(pair_results)}`",
        f"- warmup games per side: `{args.warmup_games}`",
        f"- scored games per side: `{args.scored_games}`",
        f"- deterministic actions: `{not args.sample}`",
        "",
        "Each Elo update is applied after both side orders finish. Context is reset",
        "at the start of a side-order series and is preserved across warmup and",
        "scored games inside that series, so latent policies can adapt to the",
        "current opponent before scored games count.",
        "",
        "## Top Competitors",
        "",
        "| rank | elo | agent | encoder | seed | update | character | run |",
        "|---:|---:|---|---|---:|---:|---|---|",
    ]
    for rank, row in enumerate(top, start=1):
        lines.append(
            f"| {rank} | {row['elo']:.1f} | `{row['agent']}` | `{row['encoder']}` | "
            f"{row['seed']} | {row['update']} | `{row['character_name']}` | `{row['run_name']}` |"
        )
    lines.extend(
        [
            "",
            "## Agent / Update / Character Summary",
            "",
            "| agent | encoder | update | character | competitors | elo mean | elo std |",
            "|---|---|---:|---|---:|---:|---:|",
        ]
    )
    for row in sorted(summary_rows, key=lambda item: item["elo_mean"], reverse=True)[:40]:
        lines.append(
            f"| `{row['agent']}` | `{row['encoder']}` | {row['update']} | `{row['character_name']}` | "
            f"{row['competitors']} | {row['elo_mean']:.1f} | {row['elo_std']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _make_eval_env(args: argparse.Namespace, characters: list[int]) -> MeleeLightKnockbackEnv:
    return MeleeLightKnockbackEnv(
        frame_skip=args.frame_skip,
        max_episode_frames=args.episode_length * max(1, args.frame_skip),
        agent_character=characters[0],
        opponent_character=characters[0],
        opponent_control="external",
        opponent_level=0,
        stage=args.stage,
        close_spawn=True,
        spawn_spacing=args.spawn_spacing,
        spawn_y=args.spawn_y,
        headless=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Melee Light policy-vs-policy Elo evaluation.")
    parser.add_argument("--checkpoint", action="append", default=[], help="Checkpoint path. Can be repeated.")
    parser.add_argument("--checkpoint-glob", action="append", default=[], help="Glob for checkpoint paths.")
    parser.add_argument("--characters", default="fox,falco,falcon", help="Comma-separated character names or ids.")
    parser.add_argument("--agents", default=None, help="Optional comma-separated agent filter.")
    parser.add_argument("--updates", default=None, help="Optional comma-separated checkpoint update filter.")
    parser.add_argument("--seeds", default=None, help="Optional comma-separated seed filter.")
    parser.add_argument("--warmup-games", type=int, default=16)
    parser.add_argument("--scored-games", type=int, default=64)
    parser.add_argument("--max-pairings", type=int, default=None)
    parser.add_argument("--min-pairings-per-competitor", type=int, default=4)
    parser.add_argument("--elo-k", type=float, default=32.0)
    parser.add_argument("--initial-elo", type=float, default=1500.0)
    parser.add_argument("--episode-length", type=int, default=90)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--stage", type=int, default=0)
    parser.add_argument("--spawn-spacing", type=float, default=48.0)
    parser.add_argument("--spawn-y", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--sample", action="store_true", help="Sample actions instead of argmax.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-dir", default="runs/melee_light_elo")
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def _parse_int_filter(value: str | None) -> set[int] | None:
    if value is None:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def _parse_str_filter(value: str | None) -> set[str] | None:
    if value is None:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, args.torch_threads))
    rng = random.Random(args.seed)
    device = select_device(args.device)

    checkpoint_paths = [Path(path) for path in args.checkpoint]
    for pattern in args.checkpoint_glob:
        checkpoint_paths.extend(Path(path) for path in glob.glob(pattern))
    if not checkpoint_paths:
        raise SystemExit("no checkpoints matched")

    infos = _load_checkpoint_infos(checkpoint_paths)
    agent_filter = _parse_str_filter(args.agents)
    update_filter = _parse_int_filter(args.updates)
    seed_filter = _parse_int_filter(args.seeds)
    if agent_filter is not None:
        infos = [info for info in infos if info.agent in agent_filter]
    if update_filter is not None:
        infos = [info for info in infos if info.update in update_filter]
    if seed_filter is not None:
        infos = [info for info in infos if info.seed in seed_filter]
    if len(infos) < 2:
        raise SystemExit("need at least two checkpoints after filters")

    characters = parse_character_ids(args.characters)
    competitors = _expand_competitors(infos, characters)
    pairings = _select_pairings(competitors, rng, args.max_pairings, args.min_pairings_per_competitor)
    if not pairings:
        raise SystemExit("no pairings selected")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_csv(output_dir / "competitors.csv", [asdict(competitor) for competitor in competitors])

    cache = PolicyCache(device)
    ratings = {competitor.id: float(args.initial_elo) for competitor in competitors}
    pair_results: list[PairResult] = []

    env = _make_eval_env(args, characters)
    try:
        for index, (competitor_a, competitor_b) in enumerate(pairings, start=1):
            last_error: Exception | None = None
            for attempt in range(2):
                try:
                    result = _run_pairing(
                        index,
                        env,
                        cache,
                        competitor_a,
                        competitor_b,
                        ratings,
                        args.elo_k,
                        args.warmup_games,
                        args.scored_games,
                        args.episode_length,
                        deterministic=not args.sample,
                    )
                    break
                except Exception as exc:
                    last_error = exc
                    env.close()
                    env = _make_eval_env(args, characters)
            else:
                raise RuntimeError(f"pairing {index} failed after browser restart") from last_error
            pair_results.append(result)
            if args.progress_every > 0 and (index == 1 or index % args.progress_every == 0 or index == len(pairings)):
                print(
                    f"[{index}/{len(pairings)}] {competitor_a.id} vs {competitor_b.id}: "
                    f"score_a={result.a_score:.3f} elo=({result.elo_a_after:.1f},{result.elo_b_after:.1f})",
                    flush=True,
                )
            _write_csv(output_dir / "pair_results.csv", [asdict(row) for row in pair_results])
    finally:
        env.close()

    games_by_competitor = {competitor.id: 0 for competitor in competitors}
    wins_by_competitor = {competitor.id: 0.0 for competitor in competitors}
    for row in pair_results:
        games_by_competitor[row.competitor_a] += row.scored_games_total
        games_by_competitor[row.competitor_b] += row.scored_games_total
        wins_by_competitor[row.competitor_a] += row.a_wins + 0.5 * row.draws
        wins_by_competitor[row.competitor_b] += row.b_wins + 0.5 * row.draws

    elo_rows: list[dict[str, Any]] = []
    by_id = {competitor.id: competitor for competitor in competitors}
    for competitor_id, rating in sorted(ratings.items(), key=lambda item: item[1], reverse=True):
        competitor = by_id[competitor_id]
        games = games_by_competitor[competitor_id]
        elo_rows.append(
            {
                **asdict(competitor),
                "elo": rating,
                "scored_games": games,
                "score_rate": wins_by_competitor[competitor_id] / max(1, games),
            }
        )
    _write_csv(output_dir / "elo.csv", elo_rows)

    grouped: dict[tuple[Any, ...], list[float]] = {}
    grouped_agent_update: dict[tuple[Any, ...], list[float]] = {}
    for row in elo_rows:
        grouped.setdefault((row["agent"], row["encoder"], row["update"], row["character_name"]), []).append(row["elo"])
        grouped_agent_update.setdefault((row["agent"], row["encoder"], row["update"]), []).append(row["elo"])
    summary_rows = _summarize(grouped, ["agent", "encoder", "update", "character_name"])
    summary_agent_update_rows = _summarize(grouped_agent_update, ["agent", "encoder", "update"])
    _write_csv(output_dir / "agent_update_character_summary.csv", summary_rows)
    _write_csv(output_dir / "agent_update_summary.csv", summary_agent_update_rows)
    _write_report(output_dir / "report.md", args, competitors, pair_results, elo_rows, summary_rows)

    print(f"wrote Elo outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
