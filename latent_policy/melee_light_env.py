from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import threading
import urllib.request
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np

MELEE_LIGHT_REPO_URL = "https://github.com/Saber0Github/smashmelee.git"
NODE_VERSION = "v16.20.2"
NODE_DIST = f"node-{NODE_VERSION}-linux-x64"
NODE_TARBALL_URL = f"https://nodejs.org/dist/{NODE_VERSION}/{NODE_DIST}.tar.xz"
RUNTIME_PATCH_VERSION = "melee-light-self-play-v1"
OBS_DIM = 30
_RUNTIME_LOCK = threading.Lock()


def load_melee_light_action_specs() -> list[dict[str, Any]]:
    runtime_dir = Path(__file__).with_name("melee_light_runtime")
    return json.loads((runtime_dir / "action_map.json").read_text(encoding="utf-8"))


def _patch_main_js_source(source: str) -> str:
    hook = (
        "    if (typeof window !== \"undefined\" && typeof window.__latentPolicyMeleeTick === \"function\") {\n"
        "      window.__latentPolicyMeleeTick(input);\n"
        "    }\n"
    )
    if "window.__latentPolicyMeleeTick(input);" in source:
        return source
    anchor = "    saveGameState(input,ports);\n\n  setTimeout(gameTick, 16, input);\n"
    if anchor not in source:
        raise ValueError("could not locate Melee Light gameTick loop for patching")
    return source.replace(anchor, f"    saveGameState(input,ports);\n{hook}\n  setTimeout(gameTick, 16, input);\n", 1)


def _rewrite_runtime_html(source: str) -> str:
    original = """  <script>
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
"""
    replacement = """  <script>
    window.offlineMode = true;
  </script>
  <script type="text/javascript" src="./js/bridge.js"></script>
"""
    if "./js/bridge.js" in source:
        return source
    if original not in source:
        raise ValueError("could not locate Melee Light runtime loader block")
    return source.replace(original, replacement, 1)


def _sha256_dir(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(str(file_path.relative_to(path)).encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest()


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, destination.open("wb") as out:
        shutil.copyfileobj(response, out)


def _run(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({' '.join(cmd)}):\n{proc.stdout}")
    return proc.stdout


def _ensure_node(cache_dir: Path) -> Path:
    node_root = cache_dir / "node"
    node_bin = node_root / NODE_DIST / "bin" / "node"
    if node_bin.exists():
        return node_root / NODE_DIST

    tarball_path = cache_dir / f"{NODE_DIST}.tar.xz"
    if not tarball_path.exists():
        _download_file(NODE_TARBALL_URL, tarball_path)
    node_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tarball_path, "r:xz") as tar:
        tar.extractall(node_root)
    if not node_bin.exists():
        raise RuntimeError(f"failed to extract Node runtime to {node_root}")
    return node_root / NODE_DIST


def _ensure_upstream_source(cache_dir: Path, repo_url: str) -> Path:
    source_dir = cache_dir / "upstream"
    if source_dir.exists():
        return source_dir
    _run(["git", "clone", "--depth", "1", repo_url, str(source_dir)], cwd=cache_dir)
    return source_dir


def _prepare_patched_source(upstream_dir: Path, work_dir: Path) -> Path:
    source_dir = work_dir / "source"
    if source_dir.exists():
        shutil.rmtree(source_dir)
    shutil.copytree(upstream_dir, source_dir, ignore=shutil.ignore_patterns(".git"))
    package_json = source_dir / "package.json"
    if package_json.exists():
        package_json.unlink()

    runtime_templates = Path(__file__).with_name("melee_light_runtime")
    (source_dir / "src" / "main" / "loadscreen.js").write_text(
        (runtime_templates / "loadscreen-stub.js").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (source_dir / "src" / "main" / "multiplayer" / "streamclient.js").write_text(
        (runtime_templates / "streamclient-stub.js").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (source_dir / "src" / "main" / "multiplayer" / "spectatorclient.js").write_text(
        (runtime_templates / "spectatorclient-stub.js").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    main_js_path = source_dir / "src" / "main" / "main.js"
    main_js_path.write_text(_patch_main_js_source(main_js_path.read_text(encoding="utf-8")), encoding="utf-8")
    return source_dir


def _prepare_build_workspace(work_dir: Path) -> Path:
    runtime_templates = Path(__file__).with_name("melee_light_runtime")
    build_dir = work_dir / "builder"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    shutil.copytree(runtime_templates, build_dir)
    return build_dir


def _populate_build_sources(source_dir: Path, build_dir: Path) -> None:
    src_root = source_dir / "src"
    vendor_src_dir = build_dir / "vendor_src"
    vendor_src_dir.mkdir(parents=True, exist_ok=True)
    for child in src_root.iterdir():
        if child.is_dir():
            shutil.copytree(child, vendor_src_dir / child.name)
        elif child.is_file():
            shutil.copy2(child, vendor_src_dir / child.name)
    shutil.copy2(src_root / "main.js", build_dir / "melee-light-entry.js")


def _copy_runtime_assets(source_dir: Path, runtime_dir: Path) -> None:
    dist_dir = source_dir / "dist"
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)
    shutil.copytree(dist_dir, runtime_dir)

    include_svg = source_dir / "src" / "input" / "gamepad" / "includeGamepadSVG.js"
    include_svg_out = runtime_dir / "src" / "input" / "gamepad"
    include_svg_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(include_svg, include_svg_out / "includeGamepadSVG.js")

    html_path = runtime_dir / "meleelight.html"
    html_path.write_text(_rewrite_runtime_html(html_path.read_text(encoding="utf-8")), encoding="utf-8")


def ensure_melee_light_runtime(
    cache_dir: str | Path | None = None,
    repo_url: str = MELEE_LIGHT_REPO_URL,
) -> Path:
    base_dir = Path(cache_dir) if cache_dir is not None else Path(__file__).resolve().parent.parent / ".cache" / "melee_light"
    base_dir.mkdir(parents=True, exist_ok=True)

    runtime_dir = base_dir / "runtime"
    stamp_path = base_dir / "runtime_stamp.json"
    template_hash = _sha256_dir(Path(__file__).with_name("melee_light_runtime"))
    desired_stamp = {
        "repo_url": repo_url,
        "node_version": NODE_VERSION,
        "patch_version": RUNTIME_PATCH_VERSION,
        "template_hash": template_hash,
    }

    with _RUNTIME_LOCK:
        if runtime_dir.exists() and stamp_path.exists():
            current_stamp = json.loads(stamp_path.read_text(encoding="utf-8"))
            if current_stamp == desired_stamp and (runtime_dir / "js" / "bridge.js").exists():
                return runtime_dir

        node_dir = _ensure_node(base_dir)
        upstream_dir = _ensure_upstream_source(base_dir, repo_url)
        work_dir = base_dir / "work"
        if work_dir.exists():
            shutil.rmtree(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        source_dir = _prepare_patched_source(upstream_dir, work_dir)
        build_dir = _prepare_build_workspace(work_dir)

        env = os.environ.copy()
        env["PATH"] = f"{node_dir / 'bin'}:{env.get('PATH', '')}"
        env["MELEE_LIGHT_SRC"] = str(source_dir)
        env["MELEE_LIGHT_RUNTIME_DIR"] = str(runtime_dir)

        npm_bin = node_dir / "bin" / "npm"
        npx_bin = node_dir / "bin" / "npx"
        _run([str(npm_bin), "install", "--no-package-lock"], cwd=build_dir, env=env)
        _populate_build_sources(source_dir, build_dir)

        _copy_runtime_assets(source_dir, runtime_dir)
        webpack_output = _run([str(npx_bin), "webpack", "--config", str(build_dir / "webpack.config.js")], cwd=build_dir, env=env)
        if "ERROR in" in webpack_output:
            raise RuntimeError(f"webpack build failed:\n{webpack_output}")
        stamp_path.write_text(json.dumps(desired_stamp, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return runtime_dir


class _SilentStaticHandler(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return None


class _StaticServer:
    def __init__(self, directory: Path):
        self.directory = directory
        handler = partial(_SilentStaticHandler, directory=str(directory))
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self) -> str:
        host, port = self.server.server_address
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


class MeleeLightKnockbackEnv:
    def __init__(
        self,
        frame_skip: int = 4,
        max_episode_frames: int = 240,
        agent_character: int = 2,
        opponent_character: int = 0,
        stage: int = 0,
        opponent_level: int = 4,
        opponent_control: str = "cpu",
        close_spawn: bool = True,
        spawn_spacing: float = 48.0,
        spawn_y: float = 0.0,
        headless: bool = True,
        chrome_binary: str | None = None,
        cache_dir: str | Path | None = None,
        repo_url: str = MELEE_LIGHT_REPO_URL,
        startup_timeout_s: float = 180.0,
        script_timeout_s: float = 60.0,
        render_mode: str | None = None,
    ) -> None:
        from gymnasium import spaces
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.support.ui import WebDriverWait

        self._config = {
            "frame_skip": int(frame_skip),
            "max_episode_frames": int(max_episode_frames),
            "agent_character": int(agent_character),
            "opponent_character": int(opponent_character),
            "stage": int(stage),
            "opponent_level": int(opponent_level),
            "opponent_control": str(opponent_control),
            "close_spawn": bool(close_spawn),
            "spawn_spacing": float(spawn_spacing),
            "spawn_y": float(spawn_y),
        }
        self.uses_external_opponent = str(opponent_control) != "cpu"
        self.render_mode = render_mode
        self._startup_timeout_s = float(startup_timeout_s)
        self._script_timeout_s = float(script_timeout_s)
        self._action_specs = load_melee_light_action_specs()
        self.action_space = spaces.Discrete(len(self._action_specs))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)
        self._runtime_dir = ensure_melee_light_runtime(cache_dir=cache_dir, repo_url=repo_url)
        self._server = _StaticServer(self._runtime_dir)

        options = Options()
        if headless:
            options.add_argument("--headless=new")
        options.add_argument("--autoplay-policy=no-user-gesture-required")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--mute-audio")
        options.add_argument("--no-sandbox")
        options.add_argument("--window-size=1400,900")
        binary = chrome_binary or self._default_chrome_binary()
        if binary is not None:
            options.binary_location = binary

        self._driver = None
        try:
            self._driver = webdriver.Chrome(options=options)
            self._driver.set_page_load_timeout(self._startup_timeout_s)
            self._driver.set_script_timeout(self._script_timeout_s)
            self._driver.get(f"{self._server.url}/meleelight.html")

            wait = WebDriverWait(self._driver, self._startup_timeout_s)
            wait.until(
                lambda driver: bool(
                    driver.execute_script(
                        "return Boolean(window.__meleeLightKnockbackEnv && window.__meleeLightKnockbackEnv.ready);"
                    )
                )
            )
        except Exception:
            self.close()
            raise

    def _default_chrome_binary(self) -> str | None:
        candidates = [
            "/usr/bin/google-chrome",
            "/usr/bin/chromium-browser",
            "/usr/bin/chromium",
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                return candidate
        return None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        _ = seed
        config = dict(self._config)
        if options:
            config.update({k: int(v) if isinstance(v, (bool, int, np.integer)) else v for k, v in options.items()})
        payload = self._driver.execute_script("return window.__meleeLightKnockbackEnv.reset(arguments[0]);", config)
        obs = np.nan_to_num(np.asarray(payload["observation"], dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
        info = dict(payload.get("info", {}))
        return obs, info

    def step(self, action: int, opponent_action: int | None = None):
        payload = self._driver.execute_async_script(
            """
            const action = arguments[0];
            const opponentAction = arguments[1];
            const done = arguments[arguments.length - 1];
            const env = window.__meleeLightKnockbackEnv;
            env.step(action, opponentAction, function(result) {
              done(result);
            });
            """,
            int(action),
            None if opponent_action is None else int(opponent_action),
        )
        obs = np.nan_to_num(np.asarray(payload["observation"], dtype=np.float32), nan=0.0, posinf=1e6, neginf=-1e6)
        reward = float(payload["reward"])
        if not np.isfinite(reward):
            reward = 0.0
        terminated = bool(payload["terminated"])
        truncated = bool(payload["truncated"])
        info = dict(payload.get("info", {}))
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        driver = getattr(self, "_driver", None)
        server = getattr(self, "_server", None)
        self._driver = None
        self._server = None
        if driver is not None:
            driver.quit()
        if server is not None:
            server.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            return None
