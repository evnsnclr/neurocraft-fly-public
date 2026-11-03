"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const script = fs.readFileSync(
  path.join(__dirname, "../../apps/companion/app.js"),
  "utf8",
);
const state = { phase: "playing", busy: true, log: [], join_address: "127.0.0.1:1234" };
const response = (result, ok = true) => ({ ok, json: async () => result });
const settle = () => new Promise(setImmediate);

function companion(fetch) {
  const elements = new Map();
  const cleared = [];
  const interval = {};
  const context = {
    location: { hash: "#private-test-token", pathname: "/" },
    history: { replaceState() {} },
    document: {
      getElementById(id) {
        if (!elements.has(id)) elements.set(id, {});
        return elements.get(id);
      },
      querySelectorAll() { return []; },
      querySelector() { return {}; },
    },
    fetch,
    setInterval(callback) { interval.callback = callback; return interval; },
    clearInterval(value) { cleared.push(value); },
  };
  vm.runInNewContext(script, context);
  return { elements, cleared, interval };
}

test("a rejected close keeps status polling active and shows the server error", async () => {
  let rejectClose = true;
  const calls = [];
  const app = companion(async (url, options) => {
    calls.push([url, options]);
    if (url === "/api/close") {
      assert.equal(options.method, "POST");
      assert.equal(options.body, "{}");
      assert.equal(options.headers.Authorization, "Bearer private-test-token");
      return response({ error: "Cannot stop the worker yet" }, !rejectClose);
    }
    return response(state);
  });
  await settle();
  await app.elements.get("close").onclick();
  assert.equal(app.elements.get("error").textContent, "Cannot stop the worker yet");
  assert.equal(app.elements.get("error").hidden, false);
  assert.equal(app.elements.get("phase").textContent, "Playground is live");
  assert.deepEqual(app.cleared, []);
  await app.interval.callback();
  assert.equal(calls.at(-1)[0], "/api/state");

  rejectClose = false;
  const requestsBeforeRetry = calls.length;
  await app.elements.get("close").onclick();
  assert.equal(calls.length, requestsBeforeRetry + 1, "close must not refresh an exiting server");
  assert.equal(app.elements.get("error").hidden, true);
  assert.equal(app.elements.get("phase").textContent, "Stopping Companion. You can close this tab.");
  assert.deepEqual(app.cleared, [app.interval]);
});

test("a network failure keeps the close button usable and does not claim shutdown", async () => {
  const app = companion(async (url) => {
    if (url === "/api/close") throw new Error("Connection interrupted");
    return response(state);
  });
  await settle();
  await app.elements.get("close").onclick();
  assert.equal(app.elements.get("error").textContent, "Connection interrupted");
  assert.equal(app.elements.get("phase").textContent, "Playground is live");
  assert.deepEqual(app.cleared, []);
  assert.equal(typeof app.elements.get("close").onclick, "function");
});

test("a status request already in flight cannot overwrite accepted shutdown", async () => {
  let finishState;
  let polling = false;
  const app = companion(async (url) => {
    if (url === "/api/close") return response({ status: "stopping" });
    if (polling) {
      polling = false;
      return new Promise((resolve) => { finishState = resolve; });
    }
    return response(state);
  });
  await settle();
  polling = true;
  const refresh = app.interval.callback();
  await app.elements.get("close").onclick();
  finishState(response(state));
  await refresh;
  assert.equal(app.elements.get("phase").textContent, "Stopping Companion. You can close this tab.");
  assert.equal(app.elements.get("error").hidden, true);
});
