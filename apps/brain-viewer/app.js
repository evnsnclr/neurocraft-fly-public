// NeuroCraft brain viewer — presentation aid only, never evidence.
// Renders the committed steering-fixture topology; glow = live dimensionless
// model activity from the local telemetry stream. Same-origin requests only.
"use strict";

const LAYOUT_URL = "layout-flight-steering-v1.json";
const TELEMETRY_URL = "/telemetry";
const SMOOTHING = 0.35; // exponential approach toward the latest sample
const PULSE_TICKS = 14;

const canvas = document.getElementById("brain");
const ctx = canvas.getContext("2d");
const identityLine = document.getElementById("identity-line");
const liveBadge = document.getElementById("live-badge");
const motorBars = document.getElementById("motor-bars");
const tickLine = document.getElementById("tick-line");
const fixtureSpan = document.getElementById("fixture-id");

let layout = null;
let displayed = null; // smoothed activity per node
let latest = null; // last telemetry sample
let pulses = []; // {x, y, age} sensory pulse rings
let lastEncoded = null;

function resize() {
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * scale));
  canvas.height = Math.max(1, Math.floor(rect.height * scale));
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
}
window.addEventListener("resize", resize);

function px(node, rect) {
  return { x: node.x * rect.width, y: node.y * rect.height };
}

function edgeColor(sign, alpha) {
  if (sign > 0) return `rgba(255, 184, 77, ${alpha})`;
  if (sign < 0) return `rgba(77, 198, 255, ${alpha})`;
  return `rgba(110, 120, 135, ${alpha * 0.6})`;
}

function activityColor(value, alpha) {
  return value >= 0
    ? `rgba(255, 214, 130, ${alpha})`
    : `rgba(120, 210, 255, ${alpha})`;
}

function draw() {
  requestAnimationFrame(draw);
  if (!layout) return;
  const rect = canvas.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);

  if (latest && latest.activity && displayed) {
    for (let i = 0; i < displayed.length; i += 1) {
      displayed[i] += (latest.activity[i] - displayed[i]) * SMOOTHING;
    }
  }

  // Edges: faint structural web, brightened by presynaptic activity.
  ctx.lineWidth = 0.6;
  for (const edge of layout.edges) {
    const a = px(layout.nodes[edge.pre], rect);
    const b = px(layout.nodes[edge.post], rect);
    const pre = displayed ? Math.min(1, Math.abs(displayed[edge.pre])) : 0;
    const alpha = 0.05 + pre * 0.5;
    ctx.strokeStyle = edgeColor(edge.sign, alpha);
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    const mx = (a.x + b.x) / 2;
    ctx.bezierCurveTo(mx, a.y, mx, b.y, b.x, b.y);
    ctx.stroke();
  }

  // Nodes: core dot plus activity halo.
  for (const node of layout.nodes) {
    const p = px(node, rect);
    const value = displayed ? displayed[node.index] : 0;
    const magnitude = Math.min(1, Math.abs(value));
    if (magnitude > 0.02) {
      const halo = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, 6 + magnitude * 16);
      halo.addColorStop(0, activityColor(value, 0.55 * magnitude + 0.15));
      halo.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = halo;
      ctx.beginPath();
      ctx.arc(p.x, p.y, 6 + magnitude * 16, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = magnitude > 0.02 ? activityColor(value, 0.95) : "rgba(150,160,175,0.7)";
    ctx.beginPath();
    ctx.arc(p.x, p.y, node.group === "sensory" ? 4.2 : 2.4, 0, Math.PI * 2);
    ctx.fill();
  }

  // Sensory pulse rings.
  pulses = pulses.filter((pulse) => pulse.age < PULSE_TICKS);
  for (const pulse of pulses) {
    pulse.age += 0.5;
    const t = pulse.age / PULSE_TICKS;
    ctx.strokeStyle = `rgba(255, 240, 200, ${0.6 * (1 - t)})`;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    ctx.arc(pulse.x * rect.width, pulse.y * rect.height, 6 + t * 34, 0, Math.PI * 2);
    ctx.stroke();
  }
}

function renderMotors(sample) {
  const entries = Object.entries(sample.motor_frame || {});
  motorBars.innerHTML = "";
  for (const [name, value] of entries) {
    const row = document.createElement("div");
    row.className = "motor-row";
    const label = document.createElement("div");
    label.className = "name";
    label.textContent = `${name} ${value.toFixed(3)}`;
    const track = document.createElement("div");
    track.className = "motor-track";
    const zero = document.createElement("div");
    zero.className = "zero";
    const fill = document.createElement("div");
    fill.className = value < 0 ? "motor-fill neg" : "motor-fill";
    const half = Math.min(1, Math.abs(value)) * 50;
    fill.style.left = value < 0 ? `${50 - half}%` : "50%";
    fill.style.width = `${half}%`;
    track.appendChild(zero);
    track.appendChild(fill);
    row.appendChild(label);
    row.appendChild(track);
    motorBars.appendChild(row);
  }
  tickLine.textContent = `tick ${sample.game_tick} · cmd ${sample.command_seq}`;
}

function onSample(sample) {
  if (!layout) return;
  if (!Array.isArray(sample.activity) || sample.activity.length !== layout.nodes.length) {
    identityLine.textContent =
      `telemetry does not match the ${layout.fixture_id} layout — glow disabled`;
    renderMotors(sample);
    return;
  }
  if (!displayed) displayed = new Array(layout.nodes.length).fill(0);
  latest = sample;
  identityLine.textContent =
    `${layout.fixture_id} · profile ${sample.controller_profile || "?"} · ` +
    `run ${sample.run_id || "?"}`;
  // Pulse on meaningful encoded-input change.
  if (sample.encoded) {
    for (const [key, value] of Object.entries(sample.encoded)) {
      const previous = lastEncoded ? lastEncoded[key] : undefined;
      if (previous !== undefined && Math.abs(value - previous) > 0.04) {
        const bodyId = Number(key.split(":").pop());
        const node = layout.nodes.find((n) => n.body_id === bodyId);
        if (node) pulses.push({ x: node.x, y: node.y, age: 0 });
      }
    }
    lastEncoded = sample.encoded;
  }
  renderMotors(sample);
}

async function start() {
  try {
    const response = await fetch(LAYOUT_URL);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    layout = await response.json();
  } catch (error) {
    identityLine.textContent = `could not load the circuit layout (${error.message})`;
    return;
  }
  fixtureSpan.textContent = layout.fixture_id;
  resize();

  const source = new EventSource(TELEMETRY_URL);
  source.onopen = () => {
    liveBadge.textContent = "LIVE";
    liveBadge.classList.add("on");
  };
  source.onerror = () => {
    liveBadge.textContent = "WAITING";
    liveBadge.classList.remove("on");
  };
  source.onmessage = (event) => {
    try {
      onSample(JSON.parse(event.data));
    } catch (error) {
      // Malformed lines are skipped; the stream is presentation-only.
    }
  };
  requestAnimationFrame(draw);
}

start();
