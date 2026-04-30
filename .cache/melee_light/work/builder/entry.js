import "animations";
import "./melee-light-entry";
import { start } from "main/main";
import { initMeleeLightKnockbackBridge } from "./runtime_bridge";

function boot() {
  start();
  initMeleeLightKnockbackBridge();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
