import numpy as np

from latent_policy.melee_light_elo import (
    Competitor,
    _elo_update,
    _expected_score,
    _policy_obs,
    _raw_obs_for_side,
    _select_pairings,
    parse_character_ids,
)


def _competitor(name: str) -> Competitor:
    return Competitor(
        id=name,
        checkpoint_path=f"/tmp/{name}.pt",
        run_name=name,
        agent=name,
        encoder="gru",
        seed=1,
        update=10,
        character=2,
        character_name="fox",
    )


def test_raw_obs_for_side_swaps_players_and_inverts_delta():
    raw = np.arange(30, dtype=np.float32)
    swapped = _raw_obs_for_side(raw, side=1)

    np.testing.assert_array_equal(swapped[:14], raw[14:28])
    np.testing.assert_array_equal(swapped[14:28], raw[:14])
    assert swapped[28] == -raw[28]
    assert swapped[29] == -raw[29]


def test_policy_obs_appends_wrapper_features_for_checkpoint_obs_dim():
    raw = np.arange(30, dtype=np.float32)
    obs = _policy_obs(raw, side=0, last_reward=1.0, step_count=9, episode_length=90, obs_dim=32)

    assert obs.shape == (32,)
    assert obs[-2] == 1.0
    assert np.isclose(obs[-1], 0.1)


def test_elo_update_rewards_match_winner():
    rating_a, rating_b = _elo_update(1500.0, 1500.0, score_a=0.75, k_factor=32.0)

    assert _expected_score(1500.0, 1500.0) == 0.5
    assert rating_a == 1508.0
    assert rating_b == 1492.0


def test_select_pairings_respects_max_and_covers_competitors():
    import random

    competitors = [_competitor(f"c{i}") for i in range(6)]
    pairings = _select_pairings(
        competitors,
        rng=random.Random(1),
        max_pairings=6,
        min_pairings_per_competitor=1,
    )

    assert len(pairings) == 6
    covered = {item.id for pair in pairings for item in pair}
    assert covered == {item.id for item in competitors}


def test_parse_character_ids_accepts_names_and_ids():
    assert parse_character_ids("fox,falco,4") == [2, 3, 4]
