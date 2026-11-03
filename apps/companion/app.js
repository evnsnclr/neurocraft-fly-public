"use strict";

const token = location.hash.slice(1);
history.replaceState(null, "", location.pathname);

const element = (id) => document.getElementById(id);
const headers = {
  Authorization: "Bearer " + token,
  "Content-Type": "application/json",
};
const labels = {
  idle: "Ready for setup",
  setting_up: "Preparing this computer…",
  ready: "Companion ready",
  starting: "Starting the playground…",
  playing: "Playground is live",
  stopped: "Session stopped",
  error: "Needs attention",
};

let current = { busy: false, phase: "idle" };
let closing = false;

function settings() {
  return {
    java_home: element("java").value,
    player_name: element("player").value,
    client_mods_dir: element("mods").value,
    client_launcher_dir: element("launcher").value,
    accept_eula: element("eula").checked,
    allow_downloads: element("downloads").checked,
  };
}

function showError(message) {
  element("error").textContent = message;
  element("error").hidden = !message;
}

async function api(path, body) {
  const response = await fetch("/api/" + path, {
    method: body === undefined ? "GET" : "POST",
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.error || "Companion request failed");
  }
  return result;
}

async function action(name, body = settings()) {
  try {
    showError("");
    await api(name, body);
    await refresh();
  } catch (error) {
    showError(error.message);
  }
}

async function refresh() {
  try {
    const state = await api("state");
    if (closing) return;
    current = state;
    if (current.phase === "error") {
      showError(current.log.at(-1) || "See progress below.");
      document.querySelector("details").open = true;
    }
    element("phase").textContent = labels[current.phase] || current.phase;
    element("log").textContent = current.log.join("\n");
    element("log").scrollTop = element("log").scrollHeight;
    for (const id of ["setup", "start", "install"]) {
      element(id).disabled = current.busy;
    }
    element("stop").disabled = !current.busy;
    for (const button of document.querySelectorAll("[data-condition]")) {
      button.disabled = current.phase !== "playing";
    }
    element("join").hidden = !current.join_address;
    element("address").textContent = current.join_address;
  } catch (error) {
    if (!closing) showError(error.message);
  }
}

async function copyAddress() {
  try {
    await navigator.clipboard.writeText(current.join_address);
  } catch {
    showError("Select and copy the join address shown above.");
  }
}

async function saveReport() {
  try {
    const report = await api("report");
    const link = document.createElement("a");
    link.href = URL.createObjectURL(
      new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }),
    );
    link.download = "neurocraft-support.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(link.href), 1000);
  } catch (error) {
    showError(error.message);
  }
}

async function closeCompanion() {
  try {
    showError("");
    await api("close", {});
    closing = true;
    clearInterval(timer);
    element("phase").textContent = "Stopping Companion. You can close this tab.";
  } catch (error) {
    showError(error.message);
  }
}

element("setup").onclick = () => action("setup");
element("start").onclick = () => action("start");
element("install").onclick = () => action("install-client");
element("stop").onclick = () => action("stop", {});
for (const button of document.querySelectorAll("[data-condition]")) {
  button.onclick = () => action("condition", { condition: button.dataset.condition });
}
element("copy").onclick = copyAddress;
element("report").onclick = saveReport;
element("close").onclick = closeCompanion;

if (!token) {
  showError("Open this window using the NeuroCraft Fly Companion shortcut.");
}
refresh();
const timer = setInterval(refresh, 1500);
