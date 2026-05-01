import actionSpecs from "./action_specs";
import {
  currentPlayers,
  findingPlayers,
  gameMode,
  addPlayer,
  changeGamemode,
  cpuDifficulty,
  mType,
  player,
  playerType,
  playing,
  ports,
  setCS,
  setFindingPlayers,
  setMtype,
  setCurrentPlayer,
  setPlayerType,
  setPlaying,
  setStageSelect,
  setStarting,
  setStartTimer,
  setVersusMode,
  startGame,
  starting,
  versusMode,
} from "main/main";
import { debugNetworkInputs, resetNetworkInputs, setNetworkInput } from "main/multiplayer/streamclient";
import { Box2D } from "main/util/Box2D";
import { Vec2D } from "main/util/Vec2D";
import { resetVfxQueue } from "main/vfx/vfxQueue";

const OBS_DIM = 30;
const ACTION_SPECS = Array.isArray(actionSpecs) ? actionSpecs : [];
const DEFAULT_CONFIG = {
  frame_skip: 4,
  max_episode_frames: 240,
  agent_character: 2,
  opponent_character: 0,
  stage: 0,
  opponent_level: 4,
  opponent_control: "cpu",
  close_spawn: true,
  spawn_spacing: 48,
  spawn_y: 0,
};

let bridgeConfig = { ...DEFAULT_CONFIG };
let logicFrameCount = 0;
let episodeFrameCount = 0;
let pendingOutcome = null;
let envReady = false;

function hideLoadScreen() {
  const loadScreen = document.getElementById("loadScreen");
  if (loadScreen) {
    loadScreen.remove();
  }
}

function hidePageChrome() {
  ["topButtonContainer", "keyboardPrompt", "players", "debug", "buttons"].forEach((id) => {
    const element = document.getElementById(id);
    if (element) {
      element.style.display = "none";
    }
  });
  document.body.style.margin = "0";
  document.body.style.overflow = "hidden";
}

function baseInput() {
  return {
    a: false,
    b: false,
    x: false,
    y: false,
    z: false,
    l: false,
    r: false,
    s: false,
    du: false,
    dl: false,
    dr: false,
    dd: false,
    lA: 0,
    rA: 0,
    lsX: 0,
    lsY: 0,
    csX: 0,
    csY: 0,
    rawX: 0,
    rawY: 0,
    rawcsX: 0,
    rawcsY: 0,
  };
}

function actionToInput(actionIndex) {
  const spec = ACTION_SPECS[actionIndex] || ACTION_SPECS[0] || { name: "idle" };
  const input = baseInput();
  Object.keys(spec).forEach((key) => {
    if (key === "name") {
      return;
    }
    input[key] = spec[key];
  });
  input.rawX = input.lsX;
  input.rawY = input.lsY;
  input.rawcsX = input.csX;
  input.rawcsY = input.csY;
  return input;
}

function encodePlayerState(p) {
  return [
    p.phys.pos.x,
    p.phys.pos.y,
    p.phys.cVel.x,
    p.phys.cVel.y,
    p.phys.kVel.x,
    p.phys.kVel.y,
    p.phys.face,
    p.phys.grounded ? 1 : 0,
    p.percent / 100,
    p.hit.knockback / 100,
    p.hit.hitstun / 60,
    p.hit.hitlag / 20,
    p.phys.shieldHP / 60,
    p.timer / 60,
  ];
}

function observation() {
  const p0 = player[0];
  const p1 = player[1];
  const obs = [
    ...encodePlayerState(p0),
    ...encodePlayerState(p1),
    p1.phys.pos.x - p0.phys.pos.x,
    p1.phys.pos.y - p0.phys.pos.y,
  ];
  return obs;
}

function ensurePlayers() {
  while (ports < 2) {
    addPlayer(ports, 99);
  }
  setCurrentPlayer(0, 0);
  setCurrentPlayer(1, 1);
  setMtype(0, 99);
  setMtype(1, 99);
  setPlayerType(0, 2);
  setPlayerType(1, 1);
}

function configureMatch() {
  ensurePlayers();
  setFindingPlayers(false);
  setVersusMode(1);
  setStageSelect(bridgeConfig.stage);
  cpuDifficulty[1] = bridgeConfig.opponent_level;
  setCS(0, bridgeConfig.agent_character);
  setCS(1, bridgeConfig.opponent_character);
  setPlayerType(1, bridgeConfig.opponent_control === "cpu" ? 1 : 2);
}

function placePlayer(index, x, y, face) {
  const p = player[index];
  const width = p.charAttributes.hurtboxOffset[0];
  const height = p.charAttributes.hurtboxOffset[1];
  p.actionState = "WAIT";
  p.timer = 1;
  p.inCSS = false;
  p.currentAction = "NONE";
  p.currentSubaction = "NONE";
  p.percent = 0;
  p.hit.knockback = 0;
  p.hit.hitlag = 0;
  p.hit.hitstun = 0;
  p.phys.pos = new Vec2D(x, y);
  p.phys.posPrev = new Vec2D(x, y);
  p.phys.cVel = new Vec2D(0, 0);
  p.phys.kVel = new Vec2D(0, 0);
  p.phys.kDec = new Vec2D(0, 0);
  p.phys.face = face;
  p.phys.facePrev = face;
  p.phys.grounded = true;
  p.phys.onSurface = [0, 0];
  p.phys.airborneTimer = 0;
  p.phys.doubleJumped = false;
  p.phys.jumpsUsed = 0;
  p.phys.hurtbox = new Box2D([x - width, y + height], [x + width, y]);
  p.phys.ECBp = [new Vec2D(x, y), new Vec2D(x + 3, y + 7), new Vec2D(x, y + 14), new Vec2D(x - 3, y + 7)];
  p.phys.ECB1 = [new Vec2D(x, y), new Vec2D(x + 3, y + 7), new Vec2D(x, y + 14), new Vec2D(x - 3, y + 7)];
}

function applySpawnLayout() {
  if (!bridgeConfig.close_spawn) {
    return;
  }
  const halfSpacing = Math.max(1, Number(bridgeConfig.spawn_spacing) || DEFAULT_CONFIG.spawn_spacing) / 2;
  const y = Number.isFinite(Number(bridgeConfig.spawn_y)) ? Number(bridgeConfig.spawn_y) : 0;
  placePlayer(0, -halfSpacing, y, 1);
  placePlayer(1, halfSpacing, y, -1);
}

function detectOutcome() {
  if (starting) {
    return null;
  }
  if (player[1].hit.hitlag > 0 && player[1].hit.knockback > 0) {
    return {
      reward: 1.0,
      terminated: true,
      truncated: false,
      info: {
        winner: 0,
        loser: 1,
        agent_knockback: player[0].hit.knockback,
        opponent_knockback: player[1].hit.knockback,
      },
    };
  }
  if (player[0].hit.hitlag > 0 && player[0].hit.knockback > 0) {
    return {
      reward: -1.0,
      terminated: true,
      truncated: false,
      info: {
        winner: 1,
        loser: 0,
        agent_knockback: player[0].hit.knockback,
        opponent_knockback: player[1].hit.knockback,
      },
    };
  }
  return null;
}

function freezeMatch() {
  setPlaying(false);
}

function beginMatch() {
  resetNetworkInputs();
  setNetworkInput(0, actionToInput(0));
  setNetworkInput(1, actionToInput(0));
  pendingOutcome = null;
  episodeFrameCount = 0;
  configureMatch();
  changeGamemode(2);
  startGame();
  resetVfxQueue();
  applySpawnLayout();
  setStartTimer(0);
  setStarting(false);
}

function mergeConfig(config) {
  bridgeConfig = {
    ...bridgeConfig,
    ...(config || {}),
  };
}

function stepResult(extra) {
  return {
    observation: observation(),
    reward: extra.reward,
    terminated: extra.terminated,
    truncated: extra.truncated,
    info: {
      frame_count: episodeFrameCount,
      ...extra.info,
    },
  };
}

function debugPlayerState(index) {
  const p = player[index];
  if (!p) {
    return null;
  }
  return {
    actionState: p.actionState,
    currentAction: p.currentAction,
    currentSubaction: p.currentSubaction,
    timer: p.timer,
    percent: p.percent,
    face: p.phys.face,
    grounded: Boolean(p.phys.grounded),
    pos: {
      x: p.phys.pos.x,
      y: p.phys.pos.y,
    },
    cVel: {
      x: p.phys.cVel.x,
      y: p.phys.cVel.y,
    },
    kVel: {
      x: p.phys.kVel.x,
      y: p.phys.kVel.y,
    },
    hit: {
      knockback: p.hit.knockback,
      hitlag: p.hit.hitlag,
      hitstun: p.hit.hitstun,
    },
  };
}

function debugState() {
  return {
    logicFrameCount,
    episodeFrameCount,
    gameMode,
    versusMode,
    starting,
    playing,
    findingPlayers,
    ports,
    currentPlayers: Array.from(currentPlayers),
    playerType: Array.from(playerType),
    mType: Array.from(mType),
    networkInputs: debugNetworkInputs(),
    players: [debugPlayerState(0), debugPlayerState(1), debugPlayerState(2), debugPlayerState(3)],
  };
}

function reset(config) {
  hideLoadScreen();
  hidePageChrome();
  mergeConfig(config);
  beginMatch();
  return {
    observation: observation(),
    info: {
      frame_count: episodeFrameCount,
    },
  };
}

function asyncStep(action, opponentAction, callback) {
  if (pendingOutcome !== null) {
    callback(stepResult(pendingOutcome));
    return;
  }
  setNetworkInput(0, actionToInput(action));
  if (bridgeConfig.opponent_control !== "cpu") {
    setNetworkInput(1, actionToInput(opponentAction == null ? 0 : opponentAction));
  }
  const targetFrame = logicFrameCount + bridgeConfig.frame_skip;

  function poll() {
    if (pendingOutcome !== null) {
      callback(stepResult(pendingOutcome));
      return;
    }
    if (episodeFrameCount >= bridgeConfig.max_episode_frames) {
      pendingOutcome = {
        reward: -1.0,
        terminated: true,
        truncated: false,
        info: {
          timeout: true,
          winner: 1,
          loser: 0,
          agent_knockback: player[0].hit.knockback,
          opponent_knockback: player[1].hit.knockback,
        },
      };
      freezeMatch();
      callback(stepResult(pendingOutcome));
      return;
    }
    if (logicFrameCount >= targetFrame) {
      callback(
        stepResult({
          reward: 0.0,
          terminated: false,
          truncated: false,
          info: {},
        }),
      );
      return;
    }
    setTimeout(poll, 1);
  }

  poll();
}

export function initMeleeLightKnockbackBridge() {
  hideLoadScreen();
  hidePageChrome();
  if (window.__latentPolicyMeleeTick === undefined) {
    window.__latentPolicyMeleeTick = function meleeLightEnvTick() {
      logicFrameCount += 1;
      episodeFrameCount += 1;
      if (pendingOutcome !== null) {
        return;
      }
      const outcome = detectOutcome();
      if (outcome !== null) {
        pendingOutcome = outcome;
        freezeMatch();
      }
    };
  }

  window.__meleeLightKnockbackEnv = {
    actionCount: ACTION_SPECS.length,
    actionSpecs: ACTION_SPECS,
    obsDim: OBS_DIM,
    ready: true,
    debugState,
    reset,
    step(action, opponentActionOrCallback, maybeCallback) {
      if (typeof opponentActionOrCallback === "function") {
        asyncStep(action, 0, opponentActionOrCallback);
        return;
      }
      asyncStep(action, opponentActionOrCallback, maybeCallback);
    },
  };
  envReady = true;
}

export function isBridgeReady() {
  return envReady;
}
