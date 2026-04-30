function emptyInput() {
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

const networkInputs = [emptyInput(), emptyInput(), emptyInput(), emptyInput()];

export const giveInputs = [false, false, false, false];
export const HOST_GAME_ID = null;
export const inServerMode = false;
export const meHost = false;

export function setNetworkInput(playerSlot, input) {
  networkInputs[playerSlot] = {
    ...emptyInput(),
    ...input,
  };
}

export function resetNetworkInputs() {
  for (let i = 0; i < networkInputs.length; i += 1) {
    networkInputs[i] = emptyInput();
  }
}

export function updateNetworkInputs() {}
export function connectToMPRoom() {}
export function connectToMPServer() {}
export function syncGameMode() {}
export function syncCharacter() {}
export function syncTagText() {}
export function syncStage() {}
export function syncStartGame() {}
export function logIntoServer() {}

export function retrieveNetworkInputs(playerSlot) {
  return {
    ...emptyInput(),
    ...networkInputs[playerSlot],
  };
}

export function debugNetworkInputs() {
  return networkInputs.map((input) => ({
    ...emptyInput(),
    ...input,
  }));
}
