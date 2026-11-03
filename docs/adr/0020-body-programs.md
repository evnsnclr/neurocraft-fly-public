# ADR 0020: Body programs on the fly entity

_Server-authoritative motion architecture, 2026-09-02_

---

| Field | Value |
|---|---|
| **Status** | **Implemented on `feature/body-programs`; not merged; live observation of stage 2 pending.** Lane A of stage 3 — the frame, the authority transaction, the operator surface, the presentation channels, and HOVER — is implemented and GameTest-pinned on the branch. Nine of the ten programs are declared and not built. Nothing here has been observed in a live session, and `PLAN.md` stage 2 gates the merge |
| **Date** | 2026-09-02 |
| **Authority** | `dev.neurocraft.fly.body`; `StagedProgramAuthority`; the arbitration branch in `FlyEntity.tick()`; `FlyControllerRuntime.requestProgramAuthority` / `releaseProgramAuthority` / `processPendingProgramAuthority` and the third early return in `recordActuation`; the `program` subtree in `FlyCommands`; two synchronized presentation fields on `FlyEntity`; the `BODY PROGRAM:` / `MODULATION:` lines in `NeuroCraftHud`; `scripts/verify_paused_window.py` |
| **Precedent** | ADR 0014 (the symmetric hover envelope these programs live inside); ADR 0015 (body-frame handedness); ADR 0016 / D-074 (motor-driven client animation, the declared-presentation-constant discipline, and the put-the-caveat-where-the-image-goes lesson); ADR 0017 (entity-aware rays); ADR 0006/0011/0012/0013/0018 (the fail-neutral retirement family this ADR must not weaken); ADR 0019 (the pivot that makes body programs the architecture) |
| **Supersedes** | Nothing |
| **Evidence** | 42 server GameTests (10 new, up from 32), 117 mod unit tests (10 new, up from 107), 7 new Python tests, one committed differential fixture recorded from `main`, and **four** executed negative controls recorded below. No live session. No receipt: ADR 0019 retired receipts for ordinary work |

## The claim this ADR is allowed to make

**Body programs are scripted presentation and engineering.** Every constant in them is
declared, argued for in prose, and derived from the Minecraft body's own physics or
chosen outright. None is derived from the connectome, from fly anatomy, or from any
measurement of fly behaviour. A running body program is **not** a connectome behavior,
not evidence about the connectome, and not anatomy. In stage 3 an operator command
selects one; nothing selects one from a neural readout yet.

While a program runs the neural service is paused, and a paused service does not step the
connectome at all. **Today the evidence for that is the mod's own record, not the trace**:
the transition log, `/neurocraftfly status`, and the HUD line, plus the structural fact
that scripted authority is reachable only through an accepted `PAUSE` and that a
non-neutral admitted command preempts the program instantly.

The trace-side check exists and is unit-pinned — `scripts/verify_paused_window.py`
recomputes the service's `service_neutral_v1` digest with `reason = "paused"` — but it is
**currently unusable, and that is a fact about the service, not about the checker.** No
committed trace can contain a paused window: the service refuses `PAUSE` while `RECORDING`
and refuses `RECORD_START` unless `RUNNING`
(`services/neural-engine/src/neurocraft_service/server.py:302-310`, `:381-383`, pinned by
`tests/service/test_service.py:1077-1088`), and even if that were relaxed a
service-originated neutral action carries no evidence to bind
(`packages/controller-runtime/src/neurocraft_controller/trace.py:532-537`), the recorder
refuses to flush an evidence-less pending (`:696-697`), and its two-slot pending window
raises on the third such action (`:551-552`; `max_pending` is 1 or 2 at `:406-409`),
faulting `TRACE_FAILURE` (`server.py:614-623`). So the checker is written, tested against
synthetic actions, and waiting for a service that can record a paused window. Saying
otherwise would be exactly the kind of claim this project exists not to make. See §3.

## Context

### What stage 3 has to produce

`PLAN.md` stage 3: ten named programs — hover, cruise, bank, land, walk, groom, feed with
the proboscis, escape-jump, flee, approach — "each is a small state machine over the
existing bounded five-axis actuation plus client animation, selected by an index the
entity holds. **Nothing selects them from the connectome yet — an operator command does**,
and the default is hover." Its honest claim is "the fly has ten hand-written body programs
and they look right".

Two jobs, different sizes. The small one is writing deterministic motion routines. The
large one is deciding **who owns the motor vector**, in a codebase whose entire safety and
evidence story rests on the answer being "exactly one thing, and it is the validated
neural command". Lane A is the second job, and it lands alone.

### There was no scripted motor source anywhere before this

Worth stating plainly, because the fly already hovers, cruises and banks *visually*.

- `FlyEntity.tick()` had exactly one path to motion: sample sensors, ask
  `FlyControllerRuntime.control`, and if the result is accepted hand
  `decision.appliedAction()` to `PrimitiveActuator.apply`.
- `ValidationResult.accepted` sets `appliedAction = command.rawAction().clamped()`, so the
  applied motor vector was the neural command, clamped, and nothing else.
- What existed for hover/cruise/bank was the *actuator* (ADR 0014) and the *animation*
  (ADR 0016). Neither produces a command.

### Three properties this ADR may not break

1. **Fail-closed admission.** `CommandValidator` enforces freshness, sequence continuity,
   identity binding, an observation-receipt check, and a deadline hold. A rejected frame
   applies neutral motors *structurally*: `ValidationResult`'s constructor throws if a
   rejected result carries non-neutral motors.
2. **Receipt truthfulness.** `ControlProtocol.encodeActuationReceipt` answers one question
   — for command N, what did the world do? — and derives `validation` as
   `APPLIED`/`CLAMPED` from the difference between requested and applied motors. A receipt
   whose `applied_motors` did not come from `requested_motors` by clamping is a lie the
   schema cannot express.
3. **No hidden autonomy.** `projectOwnedEntityHasNoMobOrNavigationInheritance` pins that
   `FlyEntity` extends `Entity` directly and inherits no `Mob` goals or navigation. A body
   program that decided *when* to run would be exactly that, wearing a different name.

### Three load-bearing citations, verified in the tree before this lane was written

Lane A adds a receipt-free tick path through `FlyControllerRuntime.recordActuation` — the
exact region ADR 0018's Defect B lived in. It is safe only if all three of these hold. All
three were read and confirmed at `25011b5` before any code was written:

| Claim | Where | Holds? |
|---|---|---|
| The service retires validated-window slots on `target_tick` from **observation** arrivals, explicitly so an un-receipted command costs no permanent slot | `packages/protocol/src/neurocraft_protocol/sequence.py:88` calls `retire_expired_actions`; `:136` is `while self._pending_actions and self._pending_actions[0].target_tick < current_tick` | **Holds.** The docstring (`:90-133`) states the ADR 0018 rationale in those words |
| `CommandValidator` advances its liveness anchor **inside the validator on acceptance**, not in the actuator — so discarding an accepted command cannot latch `DEADLINE_MISSED` | `mod/src/main/java/dev/neurocraft/fly/control/CommandValidator.java:172-174` (`pending.removeFirst(); lastAppliedValidThroughTick = next.validThroughTick(); return ValidationResult.accepted(next);`), read by `actuationHoldThroughTick()` at `:327` | **Holds** exactly as quoted |
| A PAUSED service sends **neutral actions**, not silence, so the hold anchor keeps advancing and the session does not fail closed during a pause | `services/neural-engine/src/neurocraft_service/server.py:543-553` routes `PAUSED`/`REPLAYING`/`FAULTED` to `_neutral_action(..., reason=self.control_state.value.lower())`; `_neutral_action` at `:201-231` sets `neutral=True`, equal `neural_step_start`/`neural_step_end`, and the `service_neutral_v1` digest at `:207-218` | **Holds.** Note the branch covers three control states, not only PAUSED; §3 uses that fact rather than ignoring it |

## Decision

### 1. A body program is a pure function; a runtime owns its state; the entity arbitrates

New package `dev.neurocraft.fly.body`, server-side only, in `mod/src/main`:

| Type | Shape |
|---|---|
| `BodyProgramId` | enum of exactly ten values, `HOVER` first. Only registered ids can run |
| `Modulation` | immutable record of four `double`s clamped to `[0, 1]`, `NEUTRAL = (1, 1, 1, 0)` |
| `BodyProgramState` | immutable record `{BodyProgramId id, long enteredGameTick, int ticksInProgram, int phaseIndex}` |
| `BodyProgramContext` | immutable record: this tick's `PhysicalSensorSnapshot` plus `Optional<SurfaceBelow>`, `Optional<BodyTargetVector>`, `Optional<ReachableFood>` |
| `BodyProgramStep` | immutable record `{PrimitiveAction motors, BodyProgramState next, float presentationPhase, Optional<BodyProgramId> exitTo}` |
| `BodyProgram` | `interface { BodyProgramId id(); boolean canEnter(BodyProgramContext); BodyProgramStep step(BodyProgramState, BodyProgramContext, Modulation); }` |
| `BodyProgramChannel` | the packed synchronized presentation word; see §6 |
| `MotorAuthority` | enum `NEURAL`, `SCRIPTED` |
| `SelectionSource` | enum `DEFAULT`, `OPERATOR`, `READOUT` |
| `BodyProgramRuntime` | per-entity holder of state, modulation, authority, counters, and a bounded transition ledger. The only mutable thing in the package |

Every program's motors go out through `PrimitiveAction.clamped(...)`, so a buggy program is
bounded by exactly the rails the neural lane is, and then by `PrimitiveActuator`'s speed
cap, yaw rate, and pitch clamp.

**Determinism is structural.** `step` is a function of `(state, context, modulation)`.
`ticksInProgram` derives from the authoritative game tick, never a wall clock. There is no
RNG in the package, and `BodyPackageBoundaryTest` pins the absence of `java.util.Random`,
`Math.random`, `System.nanoTime` and `System.currentTimeMillis` by reading the source text.

`FlyEntity.tick()` gained one arbitration branch and nothing else. `advance` returns
`Optional.empty()` unless scripted authority is held, so with no operator command the tick
path is what it was.

**That is pinned as a differential against `main`, and the pin's reach is bounded.**
`neuralAuthorityIsByteIdenticalToThePreProgramPath` replays a scenario recorded by running
an unmodified `main` tree (418d902) in a scratch worktree, letting `FlyEntity.tick()` run
for sixteen ticks with nothing driven, and capturing thirteen raw bit patterns per tick.
The fixture is committed at
`mod/src/gametest/resources/neurocraft/fly-tick-baseline-main.txt` and the branch must
reproduce it exactly. **It covers only the neutral arm of `tick()`**, because the GameTest
lane runs with no bridge configured and `control` therefore always rejects; the accepted
arm cannot be driven there at all. Its inertness rests on two weaker things, named here so
nobody mistakes the differential for total coverage: the expression that reaches the
actuator is textually unchanged from main, and `advance` is asserted, on every one of those
sixteen ticks, to decline under all four `(admitted, neutral)` combinations. The accepted
arm's real exercise is the opt-in external-bridge lane.

`DORMANCY_GAP_TICKS`'s premise is untouched, deliberately: the entity keeps calling
`FlyControllerRuntime.control` every tick under scripted authority — it discards the
*result*, not the *call* — so no third dormancy window is opened. That is why the
arbitration lives after `control()` rather than around it.

### 2. Selection comes from outside the program, and in stage 3 that outside is the operator

| Source | Stage | Mechanism |
|---|---|---|
| `DEFAULT` | now | `HOVER`, entered at spawn, on any exit, on any refusal, on release, on preemption |
| `OPERATOR` | now | `/neurocraftfly program <id> [speed] [turn] [amplitude] [urgency]`, `/neurocraftfly program release`, `/neurocraftfly program status` |
| `READOUT` | stage 4 | §7. Nothing emits this value yet |

A program **never** selects itself. `canEnter` is a *precondition*: it can only refuse a
selection someone else made, and a refusal falls back to HOVER with `reason=refused`.
Exits may name a successor (`BodyProgramStep.exitTo`) — the one place a program influences
what runs next, bounded to its declared successor and always logged.

**The operator lane is permanent.** Once readouts select programs it becomes the declared
negative control: the only way to run a program at its written form, so a readout's effect
is measurable as a departure from that baseline. Its HUD line is red where a readout's is
green, so it can never be mistaken for the wired lane on camera.

### 3. Scripted authority is gated on an accepted PAUSE, and interlocked on `neutral`

**This is the decision the whole lane turns on, and the argument is the receipt.**

If a body program could displace the motor vector while the connectome was still issuing
commands, `encodeActuationReceipt` would have to describe a tick in which requested and
applied motors differ for a reason that is not clamping. The schema has exactly two words
for that difference, `APPLIED` and `CLAMPED`, and both would be false. There is no honest
receipt for a co-driven tick. So co-driving is refused and the two lanes are made
**mutually exclusive by construction**:

1. **Entry is a two-phase transaction on the existing control channel.**
   `/neurocraftfly program <id>` issues `ControlOperation.PAUSE` and engages scripted
   authority **only on the accepted acknowledgement, at its effective tick**. Rejected,
   invalid, or missed → authority stays `NEURAL`, the operator gets a failure message, and
   the body stays on the neural lane. **A missed acknowledgement is not the same as
   "nothing happened", and the operator message says so**: the request did reach the
   service, so the service may have applied the PAUSE and simply not been heard in time.
   The fly is then on the neural lane against a paused service — it will not move, and
   further program requests fail `INVALID_STATE` until `/neurocraftfly resume`. That
   recovery is named in the failure message.
   `StagedProgramAuthority` is `StagedReset` applied to a second
   transaction, and `processPendingProgramAuthority` runs at the same tick boundary as
   `processPendingReset`. The three transactions are mutually exclusive: each refuses to
   stage while another is pending.
2. **Release is the mirror**: `RESUME`, authority drops to `NEURAL` on the accepted
   acknowledgement, and the program is dropped the same tick.
3. **The neutral interlock.** A program may own a tick **only if the command admitted on
   that tick carries `neutral = true` and an empty neural interval**
   (`neuralStepEnd() == neuralStepStart()`). The flag alone is not a sound proxy for "the
   service is paused" — a controller can emit a neutral action while `RUNNING`, and
   `DeterministicNeutralController` does — but a paused service also leaves its neural
   cursor where it was, and `ActionFrame` permits an empty interval only for a neutral
   action. Requiring both is strictly tighter, and it is the same pair of signatures
   `verify_paused_window.py` demands of a trace. A non-neutral admitted command drops the program
   that instant, the neural command is applied as normal for that tick, and the transition
   is logged `reason=preempted`. **The connectome always wins.** If *no* command is
   admitted, no admitted command carries `neutral = true`, so the program does not own the
   tick either: authority is retained and the body holds neutral for one tick. That is the
   strict reading and it is the safe one.
4. **A world reset never drops a program silently.** `resetForFixture` and
   `readAdditionalSaveData` return the runtime to its spawn state, which would otherwise
   erase both the authority and the transition ledger — reachable by a chunk reload as well
   as by an operator command. `FlyEntity` therefore calls `releaseScripted(tick, "reset")`
   first when authority is held, so the loss is logged on the server logger even though the
   in-memory ledger is cleared. `/neurocraftfly reset` additionally **refuses** while
   scripted authority is held and tells the operator to release first, because a reset that
   dropped the program while the service stayed paused would leave the two lanes disagreeing
   about who owns the body.

The receipt is **suppressed** on a program-owned tick — `recordActuation` gained a third
early return. Safe because of ADR 0018 (citation table above).

**What the record actually says about a filmed scripted interlude today**, with no new
schema field:

- **The mod's transition log** — one deterministic, greppable line per transition on the
  server logger, naming the source and the reason (§8). This is the primary record.
- **`/neurocraftfly status`** — authority, program, source, entry tick, transition count,
  owned-tick count, exact modulation, and the state of the authority transaction.
- **The HUD line** — red `OPERATOR`, with the disclaimer inside the string.

**And what it cannot say yet.** `scripts/verify_paused_window.py` recomputes the
`service_neutral_v1` digest with `reason = "paused"` for every action in a trace and
reports, per tick, whether the connectome was stepping. It refuses to accept the `neutral`
flag as proof on its own, and it distinguishes `paused` from the other two neutral reasons
(`replaying`, `faulted`) because only `paused` backs a scripted-body-program claim; it also
requires `neural_step_start == neural_step_end` as a second, independent signature. All of
that is pinned by tests and by executed negative control 2b below. **But no trace it could
read can exist today**, for the service-side reasons set out at the top of this ADR:
`PAUSE` is refused while `RECORDING`, `RECORD_START` is refused unless `RUNNING`, and a
service-originated neutral action has no evidence to flush, so a recorder that saw one
would fault `TRACE_FAILURE` within three actions. The checker is ready and unusable, and
that is the honest state. Making it usable is a service-side change; it is not in this
ADR's scope and it is the first thing to fix if a filmed scripted interlude ever needs to
be defended from the trace rather than from the log.

**What the digest does not establish, stated because it matters:** a match proves the
*service* reported itself paused for that action. It is not evidence that a body program
ran, that any particular footage is what it claims to be, or that the session was healthy.
It is one negative fact, checked exactly.

**Cost, stated:** while paused the connectome is frozen, so on `RESUME` after a long
scripted interlude its state is stale relative to a body that has moved. That is a real
property of this design, it is not a defect, and it is not hidden — but a stage-4 readout
resuming into a moved world is a thing to watch.

### 4. Modulation: four scalars, all attenuating, all declared

| Scalar | Meaning | Read by |
|---|---|---|
| `speed` | fraction of **the program's own** declared maximum forward/vertical thrust | CRUISE, WALK, FLEE, APPROACH, LAND |
| `turnRate` | fraction of the program's declared maximum yaw torque | BANK, WALK, FLEE, APPROACH |
| `amplitude` | fraction of the program's declared animation or bob amplitude | HOVER, WALK, GROOM, FEED |
| `urgency` | compresses declared phase durations toward their declared floors | ESCAPE_JUMP, FLEE, GROOM, FEED |

1. **A modulation scalar can never widen a bound.** Each is a multiplier in `[0, 1]` on a
   maximum the program itself declares, and that maximum is already inside the actuator
   rails. The clamp in `Modulation`'s compact constructor is the whole of that guarantee,
   and executed negative control 3 shows it is load-bearing.
2. **`urgency` never takes a duration below its declared floor.**
3. **A program declares which scalars it reads**, and varying an unread scalar produces a
   bit-identical motor sequence. Pinned per program (`hoverIsDeterministic`).
4. **The neutral default is `speed = 1, turnRate = 1, amplitude = 1, urgency = 0`** — the
   program exactly as written. Chosen over a mid-range default so the operator lane can
   reproduce the written form and a stage-4 readout's effect is a measurable departure
   from it. A `0.5` default would mean the two lanes never produce the same motion and the
   negative control would lose its meaning.

### 5. The world inputs programs need, and the boundary they may not cross

Three programs need something the six sensor rays cannot give: the rays return six scalar
distances clamped to 6.0 — a distance per body-aligned direction, never a bearing.

| Need | Minimal honest source | Programs | Lane |
|---|---|---|---|
| Is there a landable surface below, how far? | `snapshot.rayDownBlocks()` plus one `getBlockState(pos).isFaceSturdy(level, pos, Direction.UP)` | LAND, WALK entry | C |
| Where is the person? | `level.getNearestPlayer(entity, radius)`, converted to a body-frame bearing with the ADR 0015 basis | APPROACH, FLEE | G |
| Is there food in reach? | `level.getEntitiesOfClass(ItemEntity.class, box, pred)` over a declared item set | FEED | H |

**The boundary, which is the most important sentence in this ADR after §3:**

> **`BodyProgramContext` is a body-program input. It is never merged into
> `PhysicalSensorSnapshot`, and it is structurally unable to reach the bridge.**

Two independent reasons, both load-bearing. `PhysicalSensorSnapshot` is the sensory vector
the Python encoder consumes, and its shape is hashed into `observation_schema_hash`, so
adding a field rotates that identity and every downstream binding. And handing the
connectome a *target vector* would be navigation policy hidden in the encoder — precisely
what the Faithful Organism Mode purity rule forbids. A proximity scalar is a sensor
reading; a bearing to the nearest player is a goal.

Enforcement is three-layered: `BodyProgramRuntime` holds no reference to the bridge;
`BodyPackageBoundaryTest` reads every source file in the package and asserts both an import
allowlist and the absence of `NeuralBridge`, `ControlProtocol`, `ActionFrame`,
`CommandValidator` and `FlyControllerRuntime`; and `theObservationSchemaIsUnchanged` pins
`PhysicalSensorSnapshot`'s record components and `encodeObservation`'s emitted key set
against literal lists.

When the world inputs land, `getNearestPlayer` and `getEntitiesOfClass` resolve ties by
list order, which is chunk-section order and is not a declared invariant. The runtime must
re-sort candidates by `(distanceSqr, entity UUID)` before taking the first.

### 6. Presentation: two new synchronized channels, ADR 0016 otherwise untouched

The entity publishes two more synchronized fields beside the five motor ones:

- `BODY_PROGRAM` — `EntityDataSerializers.INT`, a packed `BodyProgramChannel`.
- `BODY_PROGRAM_PHASE` — `EntityDataSerializers.FLOAT`, a normalized `[0, 1)` phase the
  program advances deterministically.

`BodyProgramChannel` packs the selection source (2 bits), the program id (4 bits), and the
four modulation scalars (4 bits each) into one positive `int`. **Packing is a judgment
call and it has a cost:** the published modulation is quantized to tenths, so the HUD's
`MODULATION:` line shows one decimal place while the authoritative `Modulation` the runtime
applies is exactly what the selector supplied and `/neurocraftfly status` prints it exactly.
The alternative was six synchronized fields. This follows the ADR 0016 / D-074 discipline
— a presentation transform is declared where it is applied — and the codec is round-trip
tested. It is called out here so a reviewer can overrule it.

The client reads both channels in `FlyRenderer.extractRenderState` as getters, exactly like
the six it already reads, so the render-only invariant is untouched. Lane A adds **no**
animation overlay: `FlyAnimation`, `FlyModel`, and every ADR 0016 constant and client
GameTest are unchanged. The per-program overlays, the index-aware `poseLegs`, and the
proboscis model part land with lanes B–H.

### 7. Stage 4's wire change, decided now so stage 4 need not relitigate it

**Recommendation: a new message type `program_action` inside `neurocraft.protocol.v2`. Not
a new field on `action`. Not a version bump.** Three findings, all verified in the tree:

**(a) There is no such thing as an additive field on this wire.** Python's `_exact_keys` is
called with `required=cls.FIELDS` and never with `optional`; Java's `requireExactKeys` is
an exact set identity. A new field is mandatory in both directions the instant it exists.

**(b) A version bump costs the project its own evidence base.** `TraceManifest.from_dict`
refuses any trace whose `protocol_version` differs from the current constant, and there is
no dual-version reader, so `v3` makes every committed trace unverifiable — including the
frozen `EXTERNAL-BRIDGE-*` lineages pinned in append-only receipts. The version string's
*length* is also pinned by a handshake-transcript test literal.

**(c) A new message type leaves the existing action path alone.** It costs a dataclass, a
`_MESSAGE_TYPES` entry, a `parse_message` branch, a `oneOf` member in the schema, a
`decodeInbound` case, an `ActionFrame` sibling, and a golden. It does not touch
`Action.FIELDS`, `golden/action.json`, the `Action(...)` construction sites, or the
`CrossLanguageGoldenTest` action assertions. Unknown message types already fail closed on
both sides.

`program_action` carries the full causal envelope — identity, `command_seq`,
`observation_seq`, `observation_tick`, `target_tick`, `valid_through_tick`,
`neural_step_start`, `neural_step_end`, `state_digest` — plus `program` and `modulation`,
and is admitted by the *same* `CommandValidator` under the *same* rules. Selection rides
inside the envelope so it is bound to the neural steps and the state digest that produced
it; an out-of-band selector channel is refused because a selection not bound to
`command_seq` cannot be replayed or checked against a claim made over footage.

Growing v2's frame set means the string `neurocraft.protocol.v2` in a committed receipt no
longer names a fixed frame set. That is a small honesty cost, recorded here, and it is
cheaper than the permanent trace loss a bump would cause.

**Unavoidable regardless of route:** any edit to `messages.py` rotates
`ArtifactIdentities.code_hash`, because `replay_implementation_sha256` recurses over
`packages/protocol/src`, and it trips the byte pin in
`experiments/preregistration/fixture-physics-pilot-v3.json`. Preregistrations are
append-only (ADR 0019), so stage 4 needs a new one or a restructured test in the same
change. **Stage 3 pays none of this — it does not touch `packages/` at all.**

### 8. Honesty surfaces

**HUD.** One new line directly under `CONTROL:`, in one of exactly these forms:

```
BODY PROGRAM: HOVER · DEFAULT
BODY PROGRAM: WALK · OPERATOR — SCRIPTED, CONNECTOME NOT DRIVING
BODY PROGRAM: ESCAPE_JUMP · READOUT
```

`DEFAULT` in the existing neutral grey, `READOUT` in the active green, **`OPERATOR` in the
paused red**, so a scripted fly is unmistakable in a frame grab. The `OPERATOR` line
carries its disclaimer **inside the string**, not beside it — the direct lesson of D-074 /
ADR 0016, where a PNG lifted out of its receipt travelled without the visual-gain caveat.
A screenshot of a scripted fly must say so on its face. A fifth line,
`MODULATION: SPD 0.6 TRN 1.0 AMP 1.0 URG 0.0`, appears only while modulation is
non-default, and the panel grows to fit it. It is formatted to one decimal place because
that is the resolution the channel actually publishes (§6); showing two would overstate the
precision of what is on screen. The `CONTROL:` line is drawn in the paused red whenever the
body is scripted as well as whenever the controller is paused, so a frame grab never pairs
a green controller line with a red scripted-body line and invites the reader to take the
green half as the whole story.

**Log.** One line per transition, on the server logger, deterministic and greppable, also
retained in a bounded in-memory ledger:

```
BODY PROGRAM <FROM> -> <TO> tick=<gameTick> source=<DEFAULT|OPERATOR|READOUT> reason=<selected|exit|refused|preempted|released>
```

**Status.** `/neurocraftfly status` and `/neurocraftfly program status` report
`authority=… program=… source=… since_tick=… transitions=… program_owned_ticks=…
modulation=(…) authority_txn=…`, with the modulation at full precision.

**Trace.** Stage 3 adds **no** trace schema field. A claim is checked against the
paused-neutral digest recipe plus the transition log.

## HOVER, the one program lane A ships

Declared engineering constants, all in `HoverProgram`:

| Constant | Value | Meaning |
|---|---|---|
| `HOVER_BOB_AMPLITUDE` | 0.12 | peak commanded vertical thrust of the bob |
| `HOVER_BOB_PERIOD_TICKS` | 40 | two seconds at 20 Hz |
| `HOVER_SETTLE_SPEED_BLOCKS_PER_TICK` | 0.02 | deadband below which the fly is treated as parked |
| `HOVER_SETTLE_THRUST` | 0.25 | maximum station-keeping thrust |
| `HOVER_SETTLE_FULL_SPEED_BLOCKS_PER_TICK` | 0.10 | drift speed at which station keeping is at full thrust |

HOVER is unconditional, unbounded, and never self-exits. It reads only `amplitude`. Its
station-keeping term opposes the body-frame horizontal drift through
`BodyFrameMath.toBody`, the exact inverse of the ADR 0015 `toWorld` basis, added in this
change with an inverse-at-every-yaw unit test rather than restated inside the program.

**The bob is actuation, not animation**, and the zero-mean pin found a real defect in the
declared formula. Neutral motors hold altitude *exactly*, so a zero-motor hover is a body
frozen in space with beating wings; the bob is the fix. But the vertical axis is a
first-order filter — thrust adds to velocity, the body moves, then `AIR_VERTICAL_DRAG`
scales the velocity — and driving it with a sine that starts at zero leaves a velocity
transient whose position integral is roughly `1/(1 - drag)` times its size. Measured on the
declared constants, a plain `sin(2*pi*t/40)` bob **climbs about 0.83 blocks over its first
period**: a hovering fly that ascends forever. `HoverProgram` therefore enters at the phase
whose steady-state velocity is zero, `atan2(sin w, cos w - AIR_VERTICAL_DRAG)` — a closed
form in two already-declared constants, not a fitted number, and one that moves correctly
if either constant changes. **Measured in-world** by temporarily tightening the pin and
reading the failure message: drift over one 40-tick period is **below `1e-12` blocks**
against the `0.01` bound, and the vertical velocity at the half-period is `2.17e-17`.
`FlyEntity.AIR_VERTICAL_DRAG` became public for this and for no other reason.

**The bob's size is a judgment call worth a second opinion.** The same measurement puts
the peak excursion at **0.3894 blocks** above the entry altitude — the fly rises about
0.39 blocks, comes back, and repeats every two seconds. That follows from
`HOVER_BOB_AMPLITUDE = 0.12` and the period, and it is far more motion than a real fly's
hover. It reads as alive rather than as a statue, which is the point, but if it looks
wrong on camera the fix is to lower `HOVER_BOB_AMPLITUDE`, not to touch the phase.

## Regression pins

Server GameTests join `FlyEntityGameTests`, the single registered `fabric-gametest`
entrypoint. **The executed count went from 32 to 42.**

**Global**, six — five GameTests and one mod unit test:

1. `noProgramRunsWithoutASelection` — spawn, tick 45 ticks with no operator command; assert
   authority `NEURAL`, source `DEFAULT`, program `HOVER`, zero owned ticks, zero
   transitions, the published channel equal to the declared default, and the fly did not
   move. *The no-hidden-autonomy pin.*
2. `neuralAuthorityIsByteIdenticalToThePreProgramPath` — a **differential against `main`**:
   sixteen ticks of `FlyEntity.tick()` reproduced bit-for-bit from a fixture recorded on an
   unmodified `main` tree, with `advance` asserted to decline all four
   `(admitted, neutral)` combinations on every one of them. Covers the neutral arm only;
   see §1 for what it does not reach. *The inertness pin, and the rollback story.*
3. `scriptedAuthorityRequiresAnAcceptedPause` — an unacknowledged transaction stays
   `WAITING`, refuses to apply, and leaves authority `NEURAL`; a rejected PAUSE resolves
   `REJECTED` and leaves authority `NEURAL`; only an accepted PAUSE at its effective tick
   engages the operator lane.
4. `aNonNeutralCommandPreemptsAnyProgram` — hold scripted authority, admit a non-neutral
   command; the program is dropped that tick, the tick is not counted as owned, authority
   returns to `NEURAL`, and the transition is logged `reason=preempted`.
5. `bodyProgramsCannotReachTheBridge` (`mod/src/test`) — source-reading allowlist of exactly
   which JDK types and which project types the package may import, plus a forbidden-token
   list that covers both the bridge types and the `FlyEntity` accessors that would turn the
   constants-only allowance into a path to the neural lane. Its sibling
   `onlyTheStagedPauseTransactionCanEngageScriptedAuthority` pins the call sites of
   `engageScripted` outside the package to exactly one production caller,
   `StagedProgramAuthority`, plus the explicitly named GameTest file.
6. `theObservationSchemaIsUnchanged` — `PhysicalSensorSnapshot`'s 24 record components and
   both key sets of `encodeObservation`'s frame, against literal lists.

**HOVER**, four plus the bob pin:

- `hoverEntersUnconditionallyAndUnregisteredProgramsAreRefused`
- `hoverNeverSelfExits`
- `hoverStaysInsideTheActuationEnvelope` — over five modulations including two outside
  `[0, 1]`, every axis inside the rails, `|vertical| <= HOVER_BOB_AMPLITUDE`,
  `hypot(forward, lateral) <= HOVER_SETTLE_THRUST`, zero torque, and after actuation the
  speed cap and yaw rate hold
- `hoverIsDeterministic` — bit-identical across identical runs, bit-identical when the
  three unread scalars vary, and *not* identical when `amplitude` varies
- `hoverBobIsZeroMeanOverOnePeriod` — in-world, `|dy| < 0.01` blocks over 40 driven ticks

**Mod unit tests**: `BodyProgramChannelTest` (codec round-trip, published resolution,
malformed-word degradation, out-of-range level decoding, the modulation clamp) and two new
`BodyFrameMathTest` cases for `toBody`.

**Python**: `tests/service/test_verify_paused_window.py`, seven cases over the digest recipe
and the window logic.

## Executed negative controls

Four controls. Each mutation was applied to the working tree, the named suite was run, the
file was restored from a pre-mutation copy and its SHA-256 verified byte-identical, and the
suite was re-run green. Verbatim results:

| # | Mutation | Expected | Observed |
|---|---|---|---|
| 1 | `BodyProgramRuntime.advance` self-selects `SCRIPTED`/`OPERATOR` before the authority check | `noProgramRunsWithoutASelection` fails | **3 required tests failed** — `no_program_runs_without_aselection` ("a program took motor authority without a selection on tick 45"), `neural_authority_is_byte_identical_to_the_pre_program_path` ("tick 0 position word 0 diverged from main's FlyEntity.tick()"), and `scripted_authority_requires_an_accepted_pause` ("a program ran before an accepted PAUSE on tick 0"). The named pin fired, and the `main` differential independently caught the same mutation |
| 2 | `StagedProgramAuthority.advance` returns `READY` with no acknowledgement | `scriptedAuthorityRequiresAnAcceptedPause` fails | **1 required test failed** — `scripted_authority_requires_an_accepted_pause` ("authority advanced before its acknowledgement on tick 0"). Exactly the named pin |
| 2b | `verify_paused_window.classify_action` believes the `neutral` flag instead of checking the digest | the digest check stops discriminating | **2 failed, 5 passed** — `test_a_forged_neutral_digest_is_refused_rather_than_believed` and `test_a_neutral_action_stamped_with_another_reason_is_not_a_paused_window`. The digest, not the flag, is doing the work |
| 3 | `Modulation`'s compact constructor returns the raw value instead of clamping to `[0, 1]` | HOVER's envelope pin fails | **1 required GameTest failed** — `hover_stays_inside_the_actuation_envelope` ("HOVER exceeded its declared bob amplitude at tick 0 under Modulation[speed=5.0, turnRate=5.0, amplitude=5.0, urgency=5.0]"); and separately **1 unit test failed** of 117 — `BodyProgramChannelTest > modulationClampsIntoTheUnitInterval` |

Restoration hashes (SHA-256, verified after each restore):
`BodyProgramRuntime.java` `1a83f8e54baa54f7afed1d2cc19f157e319b5cc6066275b4e375ae95c8dff21c`;
`StagedProgramAuthority.java` `f8cd8d314bc888f975dd5f0426accfe5e3b3560f89fac75e849d414bb5d40aa4`;
`Modulation.java` `54818dc1c8c59ebf849e9baa0c467a33a8a8efa3c700faa8c05f388492dd9703`;
`verify_paused_window.py` `b57892f4199b71ea57b5a219dc76ebee58019b980dee6876c0a67fad00c07275`.

**The differential fixture found a defect in itself, and the record should say so.** Its first
form compared all thirteen words bit-for-bit and passed once, then failed deterministically on
the next three runs at the position words. The cause was not the code under test: a GameTest
structure is placed at a different absolute position every run, and near |x| = 3e6 a double
keeps about 5e-10 of resolution, so the low bits of `getX() - origin.x` are a function of where
the batch landed. The fix is the split comparison described above. A pin that passes once and
then fails for a reason outside the code is worse than no pin, and this one was caught only
because the negative controls were re-executed rather than assumed.

**Control 2b was executed against synthetic actions built by the test, and it could not have
been executed any other way.** `verify_trace_file` — the path that reads a committed trace
off disk and verifies its hash-linked integrity first — has never been run against real
data, and under the current service it cannot be: no committed trace can contain a paused
window, for the reasons given at the top of this ADR and in §3. This is not "untested
pending stage 2"; it is unreachable until the service is changed. The four controls above
therefore prove the digest recipe and the window logic, and prove nothing about a real
trace.

## Consequences

- **Stage 3 changes no wire, no schema, no golden, no identity.** It does not touch
  `messages.py`, `packages/`, `CommandValidator`, `ValidationResult`, `ControlProtocol` or
  `PhysicalSensorSnapshot`. It rotates the `neurocraft.gametest_code_bundle.v1` `code_hash`,
  which hashes all of `mod/src` recursively — the same rotation ADR 0016 declared, for the
  same reason, and no test pins a literal.
- **Body programs are scripted engineering and are labelled as such everywhere** — in the
  enum's javadoc, in the operator command's own success message, in the transition log, on
  the HUD, and here. Not anatomy, not fly behaviour, not connectome output. A sentence that
  blends a program with the connectome is a defect.
- **The fly is about to look far better than the connectome can account for, and that is
  the standing hazard.** Under the pinned wing-differential projection the connectome sets
  `lateral_thrust = 0.0` and `pitch_torque = 0.0` and makes `forward_thrust` and
  `vertical_thrust` the identical number — about 1.5 usable degrees of freedom. Body
  programs use all five axes. **The first time this fly banks through a coordinated turn,
  pitches toward a target, or walks, it will be doing something the connectome has never
  commanded and cannot command on the pinned profile.** This is ADR 0016's visual-gain
  hazard one level up and it is worse, because it is real motion rather than a magnified
  drawing. The HUD line, the log, the interlock and the `OPERATOR` colour exist for it. The
  paused-window verifier is meant to as well, and cannot yet.
- **A long PAUSE is mechanically survivable and has not been observed.** The neutral-action
  path keeps the hold anchor advancing, so a multi-minute scripted interlude should not fail
  closed. *Should.* Nothing in this repository has watched it happen.
- **Stage 2 has not happened.** No live session has survived under the current stack. This
  lane is design and code stacked on an unobserved foundation, and the arbitration branch is
  one more thing between a future bug and its fix. **It does not merge before stage 2 is
  observed and written into `STATUS.md`.**
- **The strict interlock has a consequence worth watching live.** Because a program owns a
  tick only when a *neutral command with an empty neural interval is admitted*, a paused
  session that drops or rejects a frame leaves the body holding neutral for that tick. That
  is fail-safe and correct, but on a lossy connection it could read as a stutter, and only a
  live session will say whether it does.
- **The honesty surface this ADR wanted is not the one it has.** The design intended the
  trace to be the place a scripted interlude is checked. The service cannot record one, so
  the mod's log is doing that job instead — a weaker artifact, because it is not
  content-addressed, not hash-linked, and not bound to the neural steps. Anyone repeating
  the claim over footage should say "the mod's log and status line say the service was
  paused", not "the trace proves the connectome was not driving".
