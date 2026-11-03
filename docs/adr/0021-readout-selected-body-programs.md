# ADR 0021: Readout-selected body programs — the `program_action` lane

_Brain-to-body interface architecture, 2026-09-03_

---

| Field | Value |
|---|---|
| **Status** | **Implemented on `main` in three reviewed lanes; not yet run live.** Lane P (wire and service plumbing, `d3223fb`), lane R (DNa01 steering binding, selection policy, program profile, `14b9a5f`; D-077), lane M (mod-side READOUT authority, receipts, HUD, `c57b41a`). The signed turn bias is carried on the wire but consumed by no program until lane C' threads it into CRUISE. The launcher refuses `make live-demo-program` until the mod's admission of `program_action` frames is on the classpath it launches, which it now is. No session under `full_mcns_loom_program_v1` has been observed |
| **Date** | 2026-09-03 |
| **Authority** | `packages/protocol/src/neurocraft_protocol/program_messages.py` and `dispatch.py`; `SessionValidator` receipt binding by kind; `services/neural-engine/.../server.py` (`Controller.step -> Command`, the `_FrameLane` latch); `packages/controller-runtime/.../steering_readout.py`, `program_policy.py`, `FullBrainController.steering_readout_means`; `scripts/run_connectome_service.py` profile `full_mcns_loom_program_v1` and `--program-decisions-out`; `mod/.../control/{CommandFrame,ProgramActionFrame}`, `CommandValidator.FrameLane`, `ControlProtocol.decodeProgramAction` / `encodeProgramReceipt`; `body/MotorAuthority.READOUT`, `BodyProgramRuntime.advanceReadout`; the guarded arbitration arm and the `faulted` release in `FlyEntity.tick()`; the READOUT lines in `NeuroCraftHud` |
| **Precedent** | ADR 0019 (the architecture this ADR realises: hand-chosen labeled readouts select scripted body programs over event-gated inputs); ADR 0020 (the body-program frame, and its §7 decision that stage 4's wire change is a new sibling frame inside protocol v2); ADR 0018 (the one-tick budget and `target_tick` retirement every command frame obeys); ADR 0015 (handedness: positive yaw turns toward the body's right; the sign convention below inherits it); ADR 0006 / 0011 / 0012 / 0013 (the fail-neutral family this ADR must not weaken); D-077 (binding DNa01 is a taxonomy change) |
| **Supersedes** | One sentence of ADR 0020 §3.3 — "a program may own a tick only if the admitted command carries `neutral = true`; the connectome always wins" — **for `program_action` frames only**. A program frame is a non-neutral admitted command whose whole content is a program selection: under READOUT authority it owns the tick by design, and under an operator hold the operator's program owns the tick and the frame is receipted `PROGRAM_REFUSED / operator` naming what actually ran. ADR 0020 §3.3's rule exists because a displaced *motor vector* has no honest receipt; a program frame requests no motors and has its own. For raw `action` frames the rule stands unchanged |
| **Evidence** | Python tests 1145 → 1358; mod unit tests 117 → 151; server GameTests 42 → 72; five cross-language goldens; a ten-condition offline probe battery on the full artifact (`scripts/probe_program_policy.py`); executed negative controls NC1–NC13 recorded in the three lanes' CHANGELOG bullets; five adversarial review passes (lane P twice, lane M twice, lane R once) with every finding applied, including one CRITICAL (a rejected program frame reaching the body) found and fixed before landing. **No live session.** No receipt (ADR 0019) |

## The claim this ADR is allowed to make

**The connectome's labeled readouts select and modulate scripted body programs through a
declared, hand-chosen mapping, while the connectome keeps running.** That is the architecture
ADR 0019 chose and the one EON Systems' demo uses. Every constant in the mapping is an
engineering constant with its rationale written beside it; the mapping's identity
(`policy_hash`) rides on every frame, so no sentence about "what the readout selected" can be
separated from the exact mapping that selected it.

**What it does not claim.** The fly is not deciding anything. The mapping is not anatomy. A
`READOUT`-selected `ESCAPE_JUMP` means exactly this: the rectified TTMn readout crossed a
declared threshold under the loom-v1 encoder, and a declared state machine then requested a
scripted 24-tick jump. Nothing about the readouts' live behaviour has been observed; the probe
numbers below are one deterministic run of synthetic sensor frames.

## Context

### What the live session showed

The first surviving session (2026-09-02, `STATUS.md`) ran the connectome's raw five-axis motor
lane: forward tied to vertical under the pinned projection, yaw at most 0.036, a standing wall
in the frontal ray holding a standing escape. The fly climbed into the ceiling and pushed into a
corner for 83 s. Lane A's body programs (ADR 0020) could not help: they are PAUSE-gated and
mutually exclusive with the neural lane, so while a program ran no readout could select
anything. Stage 4b's job was to let the connectome run *and* choose.

### Where the selection policy lives — service side

Two placements were evaluated (`docs/design/stage4b-program-action-design.md` §1). The policy
lives **service-side**, as a pure function over `(ReadoutValues, PolicyState)` wrapped around
`FullBrainController`, because (a) it is testable without Minecraft against the same synthetic
views and held-condition probe the controller tests already use; (b) a decision recorded with
its inputs and `policy_hash` is a recomputation, not a reading; (c) the DNa01 binding chains
into `policy_hash`, not `decoder_hash`, so no calibration cross-binding rotates; (d) the mod
displays exactly what it received — no client-side inference of the mapping. The one thing a
mod-side policy would buy — a body that keeps cruising when no frame arrives — is exactly the
autonomy the entity is pinned never to have.

## Decision

### 1. Wire: two new sibling frames inside `neurocraft.protocol.v2`

`program_action` (service → mod) carries the full causal envelope of an `action` (identity,
`command_seq`, observation binding, `target_tick`, `valid_through_tick`, a **non-empty** neural
interval, `state_digest`) plus `program` (an enum of the ten frozen `BodyProgramId` names),
`modulation` (four [0, 1] attenuators), `parameters.turn_bias` in [−1, 1], `readout
{ttmn_jump, dna01_turn, wing_common}` — the exact scalars the policy consumed — and
`policy_hash`. No `motors`, no `neutral`. `program_receipt` (mod → service) reports
`PROGRAM_APPLIED | PROGRAM_REFUSED | REJECTED` with the requested and the *executed* program,
modulation and parameters, the emitted motors, a closed `reason` enum (`unregistered |
precondition | operator`) and a fault code. Rules: APPLIED echoes the request; REFUSED for
`unregistered | precondition` executed HOVER under the requested modulation; REFUSED for
`operator` names whatever program actually ran under its own modulation; REJECTED executed
nothing. Five goldens pin both languages structurally, key order included. The pilot-v3
implementation-hash pin, which the `messages.py` edit would have tripped, was demoted to
history (recorded against `87a4243`, no longer verified against the working tree — ADR 0019;
`RISKS.md` R-27).

### 2. One driving frame kind per session, latched on both ends

The service latches the kind of the first non-neutral frame its controller emits and refuses a
controller that later emits the other kind (`MALFORMED`, fail-closed). The mod's
`CommandValidator` keeps a session-scoped `FrameLane` and refuses a program frame in a MOTOR
session or a non-neutral action in a PROGRAM session with `PROTOCOL_ERROR`. Neutral `action`s
— what a paused service emits in either lane — never decide or violate the lane, so Lane A's
PAUSE path is byte-identical. A RESET does not clear the lane: it does not change which
controller is on the other end.

### 3. `MotorAuthority.READOUT` — a third authority, never `SCRIPTED` reused

`SCRIPTED` means the operator holds the body while the service is paused; `READOUT` means a
program selected each tick by an admitted `program_action` frame while the service is RUNNING.
`advanceReadout` engages only from `NEURAL`; the mod applies no dwell or hysteresis of its own
(that is the policy's job — two anti-flapping layers would make behaviour a function of
neither); an unregistered or precondition-refused program falls back to HOVER with a
`PROGRAM_REFUSED` receipt and is re-asked only when the *requested* id changes; the operator's
PAUSE preempts READOUT (`reason=operator`) and RESUME hands it back on the next admitted frame.

**Fail-neutral rules, each pinned by a GameTest through the real validator path.** An empty
window is a neutral tick with authority retained and a stall counted — a body that kept
cruising with no admitted command would be motion under no command. A paused service releases
READOUT (`reason=paused`). A latched `ControllerFault` releases it (`reason=faulted`), so the
HUD can never show a green READOUT program under a red control line. **A rejected program frame
never reaches the READOUT arm**: the arbitration filter is guarded by admission — the review
found the unguarded form, inherited from the design's own pseudo-code, letting a
`SEQUENCE_ERROR` frame engage READOUT and move the fly — and the REJECTED receipt goes out
through the normal path. The differential fixture recorded from `main` before Lane A still
reproduces: with no admitted command the tick is byte-identical to the pre-program entity.

### 4. Policy v1 — declared, hand-chosen, hashed

Default `CRUISE`. `ESCAPE_JUMP` when the rectified TTMn jump readout is at least `ESCAPE_ON`
while armed and out of refractory; held `ESCAPE_HOLD_TICKS = 24` (equal by construction to the
program's fixed duration), then `ESCAPE_REFRACTORY_TICKS = 40`; re-armed below `ESCAPE_OFF`.
`ESCAPE_ON` and the urgency span were **derived by rules declared before the probe ran**:
`ESCAPE_ON = 0.5 × (peak jump on the standard 20-tick front approach ramp under
full_mcns_loom_v1)` = 0.011389…, `ESCAPE_OFF = ESCAPE_ON / 2`, `ESCAPE_URGENCY_SPAN = peak −
ESCAPE_ON` (numerically equal to `ESCAPE_ON`) — not the design draft's 0.05, which the loom
encoder's approach transient (peak 0.022778) would never have reached. `CRUISE` speed is the
wing common term mapped into [0.5, 1.0]; its `turn_bias` is `dna01_turn =
(R − L) / (|L| + |R| + 1e-4)` with a 0.10 deadband, positive toward the body's right (D-077).
The policy identity (`b4df62fe…85e8`) hashes every constant, every formula string, the
per-state output table, the driving-readout labels and the readout bindings. It is a
**declaration hash**: the unit tests, not the hash, guard that the code matches the
declaration — the sign-flip control changed behaviour and left the hash untouched, which is why
that control exists.

### 5. What the probe showed, reported straight

Ten conditions, 30 ticks each from reset, calibrated loom-v1 controller. Every held condition
(open bright, open dark, blocked front / left / right, held collision) stays in CRUISE with the
jump term at least 6× below threshold and `dna01_turn` exactly 0.0. The front approach ramp
selects ESCAPE_JUMP from tick 8 with urgency reaching exactly 1.0 at the ramp peak (by the span
rule). Left-ray and right-ray approach ramps give `dna01_turn` +0.88 and −0.84 — away from the
approaching side — and the sign-flip control inverts both. The right-ray ramp produces 2.8× the
TTMn jump of the left and selects ESCAPE twelve ticks earlier: the review traced ray slot →
channel → LC4 side and found no transposition, and the effect follows the population rather
than run order, so it is a property of this artifact's wiring as annotated (LC4 71 L / 55 R
under per-capita drive plus downstream wiring, which the probe does not separate) — not a claim
about fly biology, and nothing here should be read as bilateral symmetry. A contact onset
saturates cruise speed and selects no escape. One deterministic run; no dispersion estimate;
no player-loom trial.

### 6. Honesty surfaces

HUD: `BODY PROGRAM: ESCAPE_JUMP · READOUT ← TTMn 0.42` — the arrow names the driving readout
for CRUISE (WING) and ESCAPE_JUMP (TTMn) only; a fallback renders `· READOUT · FALLBACK` with no
arrow — and `READOUT MAPPING: HAND-CHOSEN (ADR 0019) · NOT THE FLY DECIDING · DNa01 -0.01 WING
+0.61`, the disclaimer inside the string (D-074), both pinned through the decoded synchronized
channels. Status: `authority=READOUT program=… source=READOUT lane=PROGRAM …
last_readout=(…) turn_bias=… policy_hash=…`. Log: `BODY PROGRAM A -> B tick=… source=READOUT
reason=selected readout=… policy=…`. Banner: "The connectome is running and its labeled readouts
(TTMn jump, DNa01 L−R, wing common) select and modulate scripted body programs through a
declared hand-chosen mapping. The mapping is engineering. The programs are scripted. The
connectome is not deciding anything." **Never:** "decided / chose / wanted", "the connectome
steers", "the escape reflex fired" without "the TTMn readout crossed a declared threshold",
"the fly reacts to you" until a controlled trial is observed, or any sentence pairing a READOUT
program with "connectome not driving" or an OPERATOR program with "selected by the readout".

### 7. What a trace can and cannot prove

No full-brain trace of any kind can be committed today: the receipt-trace recorder is
fixture-typed and `run_connectome_service.py` refuses `--trace-out` for every full profile. The
interim surface is the diagnostic decision writer (`--program-decisions-out`, schema
`neurocraft.program_decision.v1`), labelled everywhere as diagnostic, not content-addressed,
not evidence. When a full-brain recorder lands (design §5.2, lane T) the record binds readouts,
`policy_hash`, policy state and the receipt so that "the readout selected X at tick T" becomes a
recomputation. Until then it is a log line, and every sentence built on it says so.

## Executed negative controls

Recorded verbatim in the three CHANGELOG bullets: NC1 zero TTMn → ESCAPE never selected; NC2
synthetic readouts → every condition's selection counts change; NC3 all-open → CRUISE only;
NC4 DNa01 sign flip → unit test fails and both lateral biases invert; NC5 lane latch removed →
the mixed-frame GameTest fails; NC8 `program` key dropped → goldens fail in both languages;
NC10 HUD disclaimer removed → the string pin fails; NC11 admission guard removed → a rejected
frame reaches the body and the terminal receipt becomes unencodable; NC12 `faulted` release
removed → READOUT survives a latched fault; NC13 every program labelled with a driver → the
FALLBACK pin fails. NC9 (a policy constant changed without re-deriving `policy_hash`) is
covered by the lane-R identity test.

## Consequences

- `make live-demo-program` is the demo lane. The launcher refuses it unless the mod it launches
  admits program frames (it parses `ControlProtocol.java` for the arm).
- ADR 0020's operator lane is unchanged for its own use and preempts READOUT.
- The signed turn bias is carried but unused by any program until lane C' threads
  `ProgramParameters` into `BodyProgram.step`; the ADR 0015 class of sign risk stays open until
  then (D-077).
- `replay_implementation_sha256` and the `neurocraft.gametest_code_bundle.v1` `code_hash`
  rotate; no calibration regenerates; `RunIdentity` and `observation_schema_hash` are
  unchanged.
- The policy's 24-tick hold and the program's fixed 24-tick duration are equal by construction;
  a frame retired mid-jump freezes the program clock while the policy's hold keeps counting, so
  the policy may hand back to CRUISE before the jump completes — the mod is honest either way
  (receipts and ledger show it), and the alignment is the policy's to pin when it matters.
- Open, recorded in `STATUS.md`: whether a live approach reaches `ESCAPE_ON` through the
  entity-aware rays; whether the operator PAUSE ack arrives in time to engage a program live
  (under investigation); the lateral asymmetry.
