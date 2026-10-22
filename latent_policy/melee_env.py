from __future__ import annotations

import importlib
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


ACTION_NAMES = (
    "neutral", "left", "right", "up", "down", "jump", "shield", "attack",
    "attack_left", "attack_right", "attack_up", "attack_down", "special",
    "special_left", "special_right", "special_up", "special_down", "grab",
    "c_left", "c_right", "c_up", "c_down",
)
OBS_DIM = 35


@dataclass
class MeleeConfig:
    backend: str = "slippi"
    data_dir: str = ""
    dolphin_path: str = ".cache/slippi/squashfs-root/AppRun"
    iso_path: str = ""
    character: str = "FOX"
    stage: str = "FINAL_DESTINATION"
    frame_skip: int = 4
    max_episode_frames: int = 600
    timeout_seconds: float = 30.0
    headless: bool = True


def resolve_iso_path(value: str = "") -> Path:
    value = value or os.environ.get("MELEE_ISO", "")
    if value:
        return Path(value).expanduser()
    candidates = []
    downloads = Path.home() / "Downloads"
    for pattern in ("*.iso", "*.gcm"):
        for candidate in downloads.glob(pattern):
            with candidate.open("rb") as handle:
                header = handle.read(8)
            if header == b"GALE01\x00\x02" and candidate.stat().st_size == 1_459_978_240:
                candidates.append(candidate)
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError("Multiple Melee images found in Downloads; choose one with --iso")
    return Path("")


def first_hit_result(previous: Any, current: Any) -> tuple[bool, float, str]:
    damaged = [current.players[port].percent > previous.players[port].percent for port in (1, 2)]
    if any(damaged):
        if all(damaged):
            return True, 0.0, "simultaneous_hit"
        return True, 1.0 if damaged[1] else -1.0, "first_hit"
    if any(current.players[port].stock < previous.players[port].stock for port in (1, 2)):
        return True, 0.0, "stock_loss_without_hit"
    return False, 0.0, "ongoing"


def observation(state: Any, side: int, elapsed: int, max_frames: int) -> np.ndarray:
    features = []
    for port in ((1, 2) if side == 0 else (2, 1)):
        player = state.players[port]
        features.extend([
            player.position.x / 100.0, player.position.y / 100.0, player.percent / 100.0,
            player.shield_strength / 60.0, float(player.facing), float(player.on_ground),
            player.action.value / 400.0, player.action_frame / 60.0,
            player.speed_air_x_self / 5.0, player.speed_ground_x_self / 5.0,
            player.speed_y_self / 5.0, player.speed_x_attack / 5.0,
            player.speed_y_attack / 5.0, player.hitlag_left / 30.0,
            float(player.invulnerable), player.character.value / 32.0,
        ])
    own, other = (state.players[port] for port in ((1, 2) if side == 0 else (2, 1)))
    features.extend([(other.position.x - own.position.x) / 100.0,
                     (other.position.y - own.position.y) / 100.0, elapsed / max_frames])
    return np.nan_to_num(np.asarray(features, dtype=np.float32))


class MeleeFirstHitEnv:
    obs_dim = OBS_DIM
    action_space_n = len(ACTION_NAMES)

    def __init__(self, cfg: MeleeConfig):
        if cfg.frame_skip <= 0 or cfg.max_episode_frames <= 0 or cfg.timeout_seconds <= 0:
            raise ValueError("frame_skip, max_episode_frames, and timeout_seconds must be positive")
        self.cfg = cfg
        self.console = None
        self.controllers = []
        self.state = None
        self.elapsed = 0
        self.finished = True
        iso = resolve_iso_path(cfg.iso_path)
        if not iso.is_file():
            raise FileNotFoundError("Set MELEE_ISO or iso_path to your Melee NTSC 1.02 ISO.")
        with iso.open("rb") as handle:
            header = handle.read(8)
        if len(header) != 8 or header[:6] != b"GALE01" or header[7] != 2:
            raise ValueError("Expected an uncompressed Melee NTSC 1.02 ISO (GALE01, disc revision 2)")
        self.iso_path = str(iso.resolve())
        self.cfg.iso_path = self.iso_path
        self.melee = importlib.import_module("melee")
        self.character = self.melee.Character[cfg.character.upper()]
        self.stage = self.melee.Stage[cfg.stage.upper()]
        self.menu_helper = self.melee.MenuHelper()

    def _start(self) -> None:
        from melee.console import DolphinBuild, get_dolphin_version

        path = str(Path(self.cfg.dolphin_path).expanduser().resolve())
        version = get_dolphin_version(path)
        if self.cfg.headless and not version.mainline and version.build != DolphinBuild.EXI_AI:
            raise ValueError("Headless mode requires the ExiAI or headless mainline Slippi build.")
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as connection:
            connection.bind(("127.0.0.1", 0))
            port = connection.getsockname()[1]
        fast = version.build == DolphinBuild.EXI_AI
        self.console = self.melee.Console(
            path=path, slippi_port=port, blocking_input=True, polling_mode=True,
            polling_timeout=self.cfg.timeout_seconds, online_delay=0, fullscreen=False,
            gfx_backend="Null" if self.cfg.headless else "", disable_audio=True,
            emulation_speed=0, save_replays=False, use_exi_inputs=fast, enable_ffw=fast,
        )
        try:
            self.controllers = [self.melee.Controller(self.console, port=player_port,
                                type=self.melee.ControllerType.STANDARD) for player_port in (1, 2)]
            self.console.run(iso_path=self.iso_path,
                             platform="headless" if self.cfg.headless and version.mainline else None)
            if not self.console.connect() or not all(controller.connect() for controller in self.controllers):
                raise ConnectionError("Could not connect both libmelee controllers to Slippi Dolphin")
        except BaseException:
            self.close()
            raise

    def _next(self):
        state = self.console.step()
        if state is None:
            raise TimeoutError("No Slippi frame received; check Dolphin, ISO, and Gecko-code setup")
        return state

    def _in_game(self, state) -> bool:
        return state.menu_state in (self.melee.Menu.IN_GAME, self.melee.Menu.SUDDEN_DEATH)

    def _obs(self) -> np.ndarray:
        return np.stack([observation(self.state, side, self.elapsed, self.cfg.max_episode_frames) for side in (0, 1)])

    def reset(self) -> np.ndarray:
        if self.console is not None:
            self.close()
        self._start()
        self.state = self._next()
        deadline = time.monotonic() + self.cfg.timeout_seconds
        menu_frames = 0
        while True:
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out selecting characters/stage and starting a fresh Melee bout")
            if self._in_game(self.state):
                if self.state.frame >= 0 and all(port in self.state.players for port in (1, 2)):
                    if any(self.state.players[port].percent != 0 for port in (1, 2)):
                        raise RuntimeError("Fresh bout did not start at zero damage")
                    if any(self.state.players[port].character != self.character for port in (1, 2)):
                        raise RuntimeError("Dolphin selected a different character than configured")
                    self.elapsed = 0
                    self.finished = False
                    return self._obs()
                for controller in self.controllers:
                    controller.release_all()
            else:
                for index, controller in enumerate(self.controllers):
                    self.menu_helper.menu_helper_simple(
                        self.state, controller, character_selected=self.character,
                        stage_selected=self.stage, costume=index, cpu_level=0,
                        autostart=index == 0 and menu_frames > 120, swag=False,
                    )
                menu_frames += 1
            self.state = self._next()

    def _send_action(self, controller, action: int) -> None:
        name = ACTION_NAMES[action]
        controller.release_all()
        directions = {"left": (0.0, 0.5), "right": (1.0, 0.5), "up": (0.5, 1.0), "down": (0.5, 0.0)}
        direction = name.rsplit("_", 1)[-1]
        if direction in directions:
            stick = self.melee.Button.BUTTON_C if name.startswith("c_") else self.melee.Button.BUTTON_MAIN
            controller.tilt_analog(stick, *directions[direction])
        button = None
        if name.startswith("attack"):
            button = "BUTTON_A"
        elif name.startswith("special"):
            button = "BUTTON_B"
        elif name in {"jump", "shield", "grab"}:
            button = {"jump": "BUTTON_Y", "shield": "BUTTON_L", "grab": "BUTTON_Z"}[name]
        if button is not None:
            controller.press_button(getattr(self.melee.Button, button))

    def step(self, actions) -> tuple[np.ndarray, np.ndarray, bool, dict[str, Any]]:
        if self.finished:
            raise RuntimeError("Call reset before stepping a finished bout")
        if len(actions) != 2 or any(int(action) != action or not 0 <= action < self.action_space_n for action in actions):
            raise ValueError("Provide one valid discrete action for each player")
        for controller, action in zip(self.controllers, actions):
            self._send_action(controller, int(action))
        reward, reason = 0.0, "ongoing"
        for _ in range(min(self.cfg.frame_skip, self.cfg.max_episode_frames - self.elapsed)):
            previous = self.state
            self.state = self._next()
            if not self._in_game(self.state) or any(port not in self.state.players for port in (1, 2)):
                raise RuntimeError("Game ended unexpectedly before first-hit adjudication")
            if self.state.frame != previous.frame + 1:
                raise RuntimeError("Slippi skipped a frame; first-hit order cannot be determined reliably")
            self.elapsed += 1
            self.finished, reward, reason = first_hit_result(previous, self.state)
            if self.finished:
                break
        if not self.finished and self.elapsed >= self.cfg.max_episode_frames:
            self.finished, reason = True, "timeout"
        return self._obs(), np.asarray([reward, -reward], dtype=np.float32), self.finished, {
            "reason": reason, "frames": self.elapsed,
        }

    def close(self) -> None:
        for controller in self.controllers:
            controller.disconnect()
        self.controllers = []
        if self.console is not None:
            self.console.stop()
            self.console = None
        self.state = None
        self.finished = True


class NativeController:
    def __init__(self, simulator):
        self.simulator = simulator
        self.release_all()

    def release_all(self) -> None:
        self.value = self.simulator.neutral_controller((1, 1))

    def press_button(self, button) -> None:
        getattr(self.value.buttons, button.name.removeprefix("BUTTON_"))[...] = True

    def tilt_analog(self, button, horizontal: float, vertical: float) -> None:
        stick = self.value.c_stick if button.name == "BUTTON_C" else self.value.main_stick
        stick.x[...] = horizontal
        stick.y[...] = vertical


def native_game_state(frame):
    players = {}
    for index in range(2):
        source = frame["slots"][index]
        players[index + 1] = SimpleNamespace(
            position=SimpleNamespace(x=float(source["pos_x"]), y=float(source["pos_y"])),
            percent=float(source["percent"]), stock=int(source["stocks"]),
            shield_strength=float(source["shield_hp"]), facing=bool(source["facing"]),
            on_ground=bool(source["on_ground"]), action=SimpleNamespace(value=int(source["action_id"])),
            action_frame=int(source["action_frame"]), speed_air_x_self=float(source["speed_air_x_self"]),
            speed_ground_x_self=float(source["speed_ground_x_self"]), speed_y_self=float(source["speed_y_self"]),
            speed_x_attack=float(source["speed_x_attack"]), speed_y_attack=float(source["speed_y_attack"]),
            hitlag_left=int(source["hitlag"]), invulnerable=bool(source["invulnerable"]),
            character=SimpleNamespace(value=int(source["char_id"])),
        )
    return SimpleNamespace(frame=int(frame["frame_id"]), players=players)


class MeleeDecompEnv(MeleeFirstHitEnv):
    def __init__(self, cfg: MeleeConfig):
        if cfg.frame_skip <= 0 or cfg.max_episode_frames <= 0:
            raise ValueError("frame_skip and max_episode_frames must be positive")
        self.cfg = cfg
        self.elapsed = 0
        self.finished = True
        self.simulator = importlib.import_module("melee_sim")
        self.melee = importlib.import_module("melee")
        root = Path(cfg.data_dir or os.environ.get("MSL_DATA_DIR", "data")).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(
                "The decomp simulator needs extracted game data. Set MSL_DATA_DIR or --data-dir; "
                "extract with: python -m melee_sim.extract_data --iso /path/to/Melee.iso --out-dir data"
            )
        self.character = self.simulator.Character[cfg.character.upper()]
        self.stage = self.simulator.Stage[cfg.stage.upper()]
        self.batch = self.simulator.EnvBatch(batch_size=1, length=1, data_dir=str(root.resolve()))
        self.controllers = [NativeController(self.simulator) for _ in range(2)]
        self.rng = np.random.default_rng(1)
        self.state = None

    def seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def _next(self):
        self.batch.reset_cursor()
        for index, controller in enumerate(self.controllers):
            self.simulator.write_controller(self.batch.controller_action_view, controller.value, player=index)
        self.batch.step()
        return native_game_state(self.batch.current_frame[0])

    def _in_game(self, state) -> bool:
        return True

    def reset(self) -> np.ndarray:
        self.batch.configure_match(
            stage=self.stage,
            players=(self.simulator.PlayerConfig(self.character, costume=0),
                     self.simulator.PlayerConfig(self.character, costume=1)),
            seed=int(self.rng.integers(0, 2**32)),
        )
        self.batch.reset_all()
        for controller in self.controllers:
            controller.release_all()
        self.state = native_game_state(self.batch.current_frame[0])
        for _ in range(180):
            if self.state.frame >= 0:
                break
            self.state = self._next()
        if self.state.frame < 0 or any(player.percent != 0 for player in self.state.players.values()):
            raise RuntimeError("The native simulator did not start a fresh zero-damage bout")
        self.elapsed = 0
        self.finished = False
        return self._obs()

    def close(self) -> None:
        self.batch.close()
        self.finished = True


def make_melee_env(cfg: MeleeConfig):
    if cfg.backend == "decomp":
        return MeleeDecompEnv(cfg)
    if cfg.backend == "slippi":
        return MeleeFirstHitEnv(cfg)
    raise ValueError(f"Unknown Melee backend: {cfg.backend}")
