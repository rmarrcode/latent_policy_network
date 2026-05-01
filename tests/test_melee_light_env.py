from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from latent_policy.melee_light_env import _patch_main_js_source, _rewrite_runtime_html, load_melee_light_action_specs
from latent_policy.public_envs import GymSingleDiscreteVecEnv, PublicEnvConfig


def test_load_melee_light_action_specs():
    specs = load_melee_light_action_specs()
    assert len(specs) >= 8
    assert specs[0]["name"] == "idle"
    assert all("name" in spec for spec in specs)


def test_patch_main_js_source_is_idempotent():
    source = (
        "function gameTick(input) {\n"
        "    saveGameState(input,ports);\n\n"
        "  setTimeout(gameTick, 16, input);\n"
        "}\n"
    )
    patched = _patch_main_js_source(source)
    assert patched.count("window.__latentPolicyMeleeTick(input);") == 1
    assert _patch_main_js_source(patched) == patched


def test_rewrite_runtime_html_swaps_loader():
    html = """<html><body>
  <script>
    window.offlineMode = true;
    (function() {
      if('serviceWorker' in navigator) {
        navigator.serviceWorker.register('js/service-worker.js');
      }
    })();
    var scripts = [
      "./js/main.js",
      "./js/animations.js",
    ];
    var loadCount = 0;

    function handleScriptLoad() {
      loadCount++;
      if (loadCount >= scripts.length) {
        document.getElementById("loadScreen").remove();
        start();
      }
    }

    scripts.forEach(function(src) {
      var script = document.createElement("script");
      script.type = "text/javascript";
      script.onload = handleScriptLoad;
      document.body.appendChild(script);
      script.src = src;
    });
  </script>
</body></html>
"""
    rewritten = _rewrite_runtime_html(html)
    assert "./js/bridge.js" in rewritten
    assert "./js/main.js" not in rewritten
    assert _rewrite_runtime_html(rewritten) == rewritten


def test_runtime_bridge_timeout_counts_as_loss():
    bridge_path = Path(__file__).resolve().parents[1] / "latent_policy" / "melee_light_runtime" / "runtime_bridge.js"
    source = bridge_path.read_text(encoding="utf-8")
    assert "reward: -1.0" in source
    assert "terminated: true" in source
    assert "timeout: true" in source


def test_runtime_bridge_removes_loading_overlay():
    bridge_path = Path(__file__).resolve().parents[1] / "latent_policy" / "melee_light_runtime" / "runtime_bridge.js"
    source = bridge_path.read_text(encoding="utf-8")
    assert 'document.getElementById("loadScreen")' in source
    assert "loadScreen.remove()" in source
    assert "hidePageChrome()" in source
    assert "resetVfxQueue()" in source


def test_gym_single_make_env_routes_to_melee_light(monkeypatch):
    melee_module = importlib.import_module("latent_policy.melee_light_env")
    captured: dict[str, object] = {}

    class DummyMeleeLightKnockbackEnv:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(melee_module, "MeleeLightKnockbackEnv", DummyMeleeLightKnockbackEnv)
    vec_env = GymSingleDiscreteVecEnv.__new__(GymSingleDiscreteVecEnv)
    vec_env.cfg = PublicEnvConfig(
        kind="gym_single",
        name="melee_light_knockback",
        episode_length=21,
        env_kwargs={"frame_skip": 3, "opponent_level": 2},
    )

    env = vec_env._make_env()

    assert isinstance(env, DummyMeleeLightKnockbackEnv)
    assert captured["frame_skip"] == 3
    assert captured["opponent_level"] == 2
    assert captured["max_episode_frames"] == 63


def test_gym_single_terminal_reward_survives_auto_reset(monkeypatch):
    melee_module = importlib.import_module("latent_policy.melee_light_env")

    class Discrete:
        def __init__(self, n: int):
            self.n = n

    class DummyMeleeLightKnockbackEnv:
        action_space = Discrete(2)

        def __init__(self, **kwargs):
            pass

        def reset(self, seed=None):
            return np.zeros(3, dtype=np.float32)

        def step(self, action):
            return np.zeros(3, dtype=np.float32), 1.0, True, False, {}

        def close(self):
            pass

    monkeypatch.setattr(melee_module, "MeleeLightKnockbackEnv", DummyMeleeLightKnockbackEnv)
    vec_env = GymSingleDiscreteVecEnv(
        PublicEnvConfig(kind="gym_single", name="melee_light_knockback", num_envs=1, episode_length=5)
    )

    try:
        _, rewards, done, info = vec_env.step(np.array([0]))
    finally:
        vec_env.close()

    assert rewards.tolist() == [1.0]
    assert done.tolist() == [True]
    assert info["episode_return"].tolist() == [1.0]
