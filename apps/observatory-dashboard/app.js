"use strict";

const MAX_LOCAL_FILE_BYTES = 64 * 1024 * 1024;
const SERIES_COLORS = ["#6ee7dd", "#f2b84b", "#ff746c", "#a9e45b", "#bca8ff", "#76a8ff"];
const MOTOR_AXES = [
  "forward_thrust",
  "lateral_thrust",
  "vertical_thrust",
  "yaw_torque",
  "pitch_torque",
];

const elements = {
  file: document.querySelector("#trace-file"),
  traceState: document.querySelector("#trace-state"),
  traceDetail: document.querySelector("#trace-detail"),
  mode: document.querySelector("#mode-label"),
  tier: document.querySelector("#tier-label"),
  claim: document.querySelector("#claim-label"),
  tick: document.querySelector("#tick-label"),
  validation: document.querySelector("#validation-label"),
  faultCount: document.querySelector("#fault-count"),
  faultList: document.querySelector("#fault-list"),
  motorBars: document.querySelector("#motor-bars"),
  identityList: document.querySelector("#identity-list"),
  activityLegend: document.querySelector("#activity-legend"),
  canvas: document.querySelector("#activity-chart"),
  fileLabel: document.querySelector("#file-label"),
};

function setText(element, value) {
  element.textContent = String(value);
}

function shortHash(value) {
  return typeof value === "string" && value.length >= 16
    ? `${value.slice(0, 10)}…${value.slice(-6)}`
    : "—";
}

async function sha256Hex(text) {
  const bytes = new TextEncoder().encode(text);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, "0")).join("");
}

function hashMaterialFromCanonicalLine(line, entryHash) {
  const prefix = `{"entry_hash":"${entryHash}",`;
  if (!line.startsWith(prefix)) {
    throw new Error("trace line is not in the canonical envelope form");
  }
  return `{${line.slice(prefix.length)}`;
}

async function parseTrace(text) {
  const lines = text.split("\n").filter((line) => line.length > 0);
  if (lines.length === 0) {
    throw new Error("file is empty");
  }
  const parsed = lines.map((line) => JSON.parse(line));
  const isTrace = parsed.every(
    (entry) => entry && typeof entry === "object" && "entry_hash" in entry && "kind" in entry,
  );
  if (!isTrace) {
    const events = parsed.length === 1 && Array.isArray(parsed[0]) ? parsed[0] : parsed;
    return { manifest: null, records: events, chainState: "EVENT JSON · NOT HASH-VERIFIED" };
  }

  let previous = "0".repeat(64);
  let footerSeen = false;
  let manifest = null;
  const records = [];
  for (let index = 0; index < parsed.length; index += 1) {
    const entry = parsed[index];
    if (footerSeen) {
      throw new Error("content appears after trace footer");
    }
    if (entry.previous_hash !== previous) {
      throw new Error(`chain link mismatch at line ${index + 1}`);
    }
    const material = hashMaterialFromCanonicalLine(lines[index], entry.entry_hash);
    if ((await sha256Hex(material)) !== entry.entry_hash) {
      throw new Error(`entry hash mismatch at line ${index + 1}`);
    }
    if (index === 0 && entry.kind !== "manifest") {
      throw new Error("first entry isn't a manifest");
    }
    if (entry.kind === "manifest") {
      if (manifest !== null) throw new Error("duplicate manifest");
      manifest = entry.payload;
    } else if (entry.kind === "record") {
      records.push(entry.payload);
    } else if (entry.kind === "footer") {
      if (entry.payload.record_count !== records.length) {
        throw new Error("footer record count mismatch");
      }
      footerSeen = true;
    } else {
      throw new Error(`unknown entry kind at line ${index + 1}`);
    }
    previous = entry.entry_hash;
  }
  if (!footerSeen) throw new Error("required trace footer is missing");
  return { manifest, records, chainState: `HASH CHAIN VERIFIED · ${records.length} RECORDS` };
}

function renderMotors(record) {
  elements.motorBars.replaceChildren();
  const values = record?.applied_action ?? Object.fromEntries(MOTOR_AXES.map((axis) => [axis, 0]));
  for (const axis of MOTOR_AXES) {
    const value = Number.isFinite(Number(values[axis])) ? Math.max(-1, Math.min(1, Number(values[axis]))) : 0;
    const row = document.createElement("div");
    row.className = "motor-row";
    const label = document.createElement("span");
    label.textContent = axis.replaceAll("_", " ");
    const track = document.createElement("span");
    track.className = "motor-track";
    const fill = document.createElement("span");
    fill.className = "motor-fill";
    fill.style.left = `${value < 0 ? 50 + value * 50 : 50}%`;
    fill.style.width = `${Math.abs(value) * 50}%`;
    track.append(fill);
    const number = document.createElement("span");
    number.textContent = value.toFixed(3);
    row.append(label, track, number);
    elements.motorBars.append(row);
  }
}

function renderIdentity(manifest, latest) {
  const artifacts = manifest?.artifacts ?? {};
  const identity = latest?.identity ?? {};
  const rows = [
    ["Run", manifest?.run_id ?? identity.run_id],
    ["Graph", artifacts.runtime_graph_hash ?? identity.graph_hash],
    ["Adapter", artifacts.adapter_hash ?? identity.adapter_hash],
    ["Dynamics", artifacts.dynamics_hash ?? identity.dynamics_hash],
    ["Decoder", artifacts.decoder_hash ?? identity.decoder_hash],
  ];
  elements.identityList.replaceChildren();
  for (const [label, value] of rows) {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    const detail = document.createElement("dd");
    term.textContent = label;
    detail.textContent = label === "Run" ? value ?? "—" : shortHash(value);
    detail.title = value ?? "";
    row.append(term, detail);
    elements.identityList.append(row);
  }
}

function renderFaults(records) {
  const faults = records.filter(
    (record) => record?.fault_code || record?.validation === "NEUTRAL_FAULT" || record?.validation === "REJECTED",
  );
  setText(elements.faultCount, `${faults.length} ${faults.length === 1 ? "FAULT" : "FAULTS"}`);
  elements.faultList.replaceChildren();
  if (faults.length === 0) {
    const empty = document.createElement("li");
    empty.className = "empty-row";
    empty.textContent = records.length ? "No faulted records in the loaded file." : "No records loaded.";
    elements.faultList.append(empty);
    return;
  }
  for (const record of faults.slice(-100).reverse()) {
    const row = document.createElement("li");
    const tick = document.createElement("span");
    const fault = document.createElement("span");
    tick.textContent = `tick ${record.game_tick ?? "—"}`;
    fault.className = "fault-code";
    fault.textContent = `${record.fault_code ?? record.validation} · ${record.neutral_reason ?? "neutral"}`;
    row.append(tick, fault);
    elements.faultList.append(row);
  }
}

function renderActivity(records) {
  const canvas = elements.canvas;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#121513";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = "#343a36";
  context.lineWidth = 1;
  for (let row = 1; row < 4; row += 1) {
    const y = (canvas.height * row) / 4;
    context.beginPath();
    context.moveTo(0, y);
    context.lineTo(canvas.width, y);
    context.stroke();
  }

  const visible = records.slice(-240);
  const keys = Array.from(
    new Set(visible.flatMap((record) => Object.keys(record?.motor_pool_values ?? {}))),
  ).slice(0, SERIES_COLORS.length);
  elements.activityLegend.replaceChildren();
  if (visible.length < 2 || keys.length === 0) {
    context.fillStyle = "#aaa89f";
    context.font = "18px SFMono-Regular, monospace";
    context.fillText("No recorded motor-pool activity", 28, 48);
    return;
  }

  const magnitudes = visible.flatMap((record) =>
    keys.map((key) => Math.abs(Number(record.motor_pool_values?.[key] ?? 0))),
  );
  const scale = Math.max(1, ...magnitudes);
  keys.forEach((key, keyIndex) => {
    const color = SERIES_COLORS[keyIndex];
    const legend = document.createElement("span");
    legend.style.setProperty("--series-color", color);
    legend.textContent = key;
    elements.activityLegend.append(legend);
    context.strokeStyle = color;
    context.lineWidth = 2;
    context.beginPath();
    visible.forEach((record, index) => {
      const x = (index / (visible.length - 1)) * canvas.width;
      const value = Number(record.motor_pool_values?.[key] ?? 0);
      const y = canvas.height / 2 - (value / scale) * (canvas.height * 0.42);
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    });
    context.stroke();
  });
}

function render(data, file) {
  const latest = data.records.at(-1) ?? null;
  setText(elements.traceState, data.chainState);
  setText(elements.traceDetail, `${file.name} · ${(file.size / 1024).toFixed(1)} KiB`);
  setText(elements.mode, data.manifest?.mode ?? latest?.mode ?? "UNDECLARED");
  setText(elements.tier, data.manifest?.data_tier ?? latest?.data_tier ?? "UNDECLARED");
  setText(elements.claim, data.manifest?.claim_scope ?? latest?.claim_scope ?? "UNDECLARED");
  setText(elements.tick, latest ? `tick ${latest.game_tick}` : "tick —");
  setText(elements.validation, latest?.validation ?? "NO RECORD");
  setText(elements.fileLabel, `LOCAL FILE · ${file.name}`);
  renderMotors(latest);
  renderIdentity(data.manifest, latest);
  renderFaults(data.records);
  renderActivity(data.records);
}

async function loadSelectedFile() {
  const [file] = elements.file.files;
  if (!file) return;
  if (file.size > MAX_LOCAL_FILE_BYTES) {
    throw new Error("local file exceeds the 64 MiB observatory limit");
  }
  setText(elements.traceState, "INSPECTING LOCAL FILE");
  const data = await parseTrace(await file.text());
  render(data, file);
}

elements.file.addEventListener("change", () => {
  loadSelectedFile().catch((error) => {
    setText(elements.traceState, "TRACE REJECTED");
    setText(elements.traceDetail, error instanceof Error ? error.message : "unknown parse failure");
    setText(elements.mode, "UNDECLARED");
    setText(elements.tier, "UNDECLARED");
    setText(elements.claim, "UNDECLARED");
    renderMotors(null);
    renderIdentity(null, null);
    renderFaults([]);
    renderActivity([]);
  });
});

renderMotors(null);
renderActivity([]);
