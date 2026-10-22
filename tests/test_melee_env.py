from types import SimpleNamespace

import numpy as np
import pytest

from latent_policy.melee_env import (
    ACTION_NAMES, OBS_DIM, MeleeConfig, MeleeDecompEnv, MeleeFirstHitEnv,
    first_hit_result, make_melee_env, observation,
    resolve_iso_path,
)


def player(percent=0, stock=4, position=0):
    return SimpleNamespace(percent=percent, stock=stock, position=SimpleNamespace(x=position, y=0),
                           shield_strength=60, facing=True, on_ground=True, action=SimpleNamespace(value=14),
                           action_frame=1, speed_air_x_self=0, speed_ground_x_self=0, speed_y_self=0,
                           speed_x_attack=0, speed_y_attack=0, hitlag_left=0, invulnerable=False,
                           character=SimpleNamespace(value=1))


def state(first=0, second=0, frame=0, stock=4):
    return SimpleNamespace(players={1: player(first, stock, -20), 2: player(second, position=20)},
                           frame=frame, menu_state="game")


@pytest.mark.parametrize("first,second,reward,reason", [
    (0, 3, 1.0, "first_hit"), (3, 0, -1.0, "first_hit"),
    (2, 3, 0.0, "simultaneous_hit"), (0, 0, 0.0, "ongoing"),
])
def test_first_hit_adjudication(first, second, reward, reason):
    done, actual, actual_reason = first_hit_result(state(), state(first, second))
    assert done == bool(first or second)
    assert actual == reward
    assert actual_reason == reason


def test_no_hit_stock_loss_is_draw_and_shield_damage_is_not_hit():
    assert first_hit_result(state(), state(stock=3)) == (True, 0.0, "stock_loss_without_hit")
    shielded = state()
    shielded.players[2].shield_strength = 20
    assert first_hit_result(state(), shielded) == (False, 0.0, "ongoing")


def test_observations_use_each_players_perspective():
    first = observation(state(), 0, 4, 600)
    second = observation(state(), 1, 4, 600)
    assert first.shape == second.shape == (OBS_DIM,)
    np.testing.assert_array_equal(first[:16], second[16:32])
    np.testing.assert_array_equal(first[16:32], second[:16])
    assert first[32] == -second[32]
    assert np.isclose(first[-1], 4 / 600)


def frame_env(frames, max_frames=600):
    env = MeleeFirstHitEnv.__new__(MeleeFirstHitEnv)
    env.cfg = MeleeConfig(max_episode_frames=max_frames)
    env.state = state()
    env.elapsed = 0
    env.finished = False
    env.controllers = [object(), object()]
    env._send_action = lambda controller, action: None
    iterator = iter(frames)
    env._next = lambda: next(iterator)
    env._in_game = lambda current: current.menu_state == "game"
    return env


def test_action_repeat_stops_on_first_hit_before_later_counterhit():
    env = frame_env([state(0, 3, 1), state(3, 3, 2)])
    _, rewards, done, info = env.step([0, 0])
    assert done
    assert rewards.tolist() == [1, -1]
    assert info == {"reason": "first_hit", "frames": 1}


def test_timeout_is_draw_without_exceeding_frame_limit():
    env = frame_env([state(frame=1), state(frame=2)], max_frames=2)
    _, rewards, done, info = env.step([0, 0])
    assert done
    assert rewards.tolist() == [0, 0]
    assert info == {"reason": "timeout", "frames": 2}
    with pytest.raises(RuntimeError, match="reset"):
        env.step([0, 0])


def test_missing_frames_fail_instead_of_guessing_first_hit():
    env = frame_env([state(0, 3, frame=2)])
    with pytest.raises(RuntimeError, match="skipped a frame"):
        env.step([0, 0])


def test_action_mapping_releases_previous_inputs_and_targets_c_stick():
    env = MeleeFirstHitEnv.__new__(MeleeFirstHitEnv)
    env.melee = SimpleNamespace(Button=SimpleNamespace(BUTTON_C="c", BUTTON_MAIN="main", BUTTON_A="a"))
    calls = []
    controller = SimpleNamespace(release_all=lambda: calls.append("release"),
                                 tilt_analog=lambda *args: calls.append(args),
                                 press_button=lambda button: calls.append(button))
    env._send_action(controller, ACTION_NAMES.index("attack_left"))
    assert calls == ["release", ("main", 0.0, 0.5), "a"]
    calls.clear()
    env._send_action(controller, ACTION_NAMES.index("c_up"))
    assert calls == ["release", ("c", 0.5, 1.0)]


def test_missing_iso_fails_with_setup_instruction(tmp_path):
    with pytest.raises(FileNotFoundError, match="MELEE_ISO"):
        MeleeFirstHitEnv(MeleeConfig(iso_path=str(tmp_path / "missing.iso")))


def test_wrong_iso_revision_is_rejected(tmp_path):
    iso = tmp_path / "wrong.iso"
    iso.write_bytes(b"GALE01\x00\x00")
    with pytest.raises(ValueError, match="revision 2"):
        MeleeFirstHitEnv(MeleeConfig(iso_path=str(iso)))


def test_iso_discovery_requires_complete_supported_image(tmp_path, monkeypatch):
    downloads = tmp_path / "Downloads"
    downloads.mkdir()
    monkeypatch.setattr("latent_policy.melee_env.Path.home", lambda: tmp_path)
    monkeypatch.delenv("MELEE_ISO", raising=False)
    image = downloads / "Melee.iso"
    image.write_bytes(b"GALE01\x00\x02")
    assert resolve_iso_path() == type(image)("")
    with image.open("r+b") as handle:
        handle.truncate(1_459_978_240)
    assert resolve_iso_path() == image
    assert resolve_iso_path("/explicit/game.iso") == type(image)("/explicit/game.iso")


def test_slippi_reset_relaunches_previous_console():
    env = MeleeFirstHitEnv.__new__(MeleeFirstHitEnv)
    env.cfg = MeleeConfig()
    env.console = object()
    events = []
    env.close = lambda: events.append("close")
    env._start = lambda: events.append("start")
    env._next = lambda: state()
    env._in_game = lambda current: True
    env.character = SimpleNamespace(value=1)
    env.reset()
    assert events == ["close", "start"]
    assert env.elapsed == 0
    assert not env.finished


def test_decomp_missing_data_fails_without_requesting_dolphin(tmp_path):
    pytest.importorskip("melee_sim")
    with pytest.raises(FileNotFoundError, match="extracted game data"):
        make_melee_env(MeleeConfig(backend="decomp", data_dir=str(tmp_path / "missing")))


def test_native_adapter_maps_inputs_and_stops_at_first_hit(tmp_path, monkeypatch):
    simulator = pytest.importorskip("melee_sim")

    class FakeBatch:
        def __init__(self, **kwargs):
            self.current_frame = np.zeros(1, dtype=simulator.gamestate_dtype())
            self.controller_action_view = np.zeros((1, 1), dtype=simulator.controller_input_dtype())
            self.closed = False

        def configure_match(self, **kwargs):
            self.match_config = kwargs

        def reset_all(self):
            self.current_frame[...] = np.zeros(1, dtype=simulator.gamestate_dtype())
            self.current_frame["frame_id"] = -2
            self.current_frame["slots"]["stocks"] = 4
            self.current_frame["slots"]["char_id"] = 1

        def reset_cursor(self):
            pass

        def step(self):
            self.current_frame["frame_id"] += 1
            if self.current_frame["frame_id"][0] == 1:
                self.current_frame["slots"]["percent"][0, 1] = 3

        def close(self):
            self.closed = True

    monkeypatch.setattr(simulator, "EnvBatch", FakeBatch)
    env = MeleeDecompEnv(MeleeConfig(backend="decomp", data_dir=str(tmp_path)))
    assert env.reset().shape == (2, OBS_DIM)
    assert env.elapsed == 0
    _, rewards, done, info = env.step([ACTION_NAMES.index("attack_left"), ACTION_NAMES.index("c_up")])
    assert rewards.tolist() == [1, -1]
    assert done and info["frames"] == 1
    inputs = env.batch.controller_action_view["players"]
    assert inputs["buttons"]["A"][0, 0, 0] == 1
    assert inputs["main_stick_x"][0, 0, 0] == 0
    assert inputs["c_stick_y"][0, 0, 1] == 1
    env.reset()
    assert not env.finished
    env.close()
    assert env.batch.closed
