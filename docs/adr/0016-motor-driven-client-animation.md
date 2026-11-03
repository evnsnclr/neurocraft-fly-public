# ADR 0016: Motor-driven client animation for the fly body

_Client presentation, drafted 2026-08-31_

---

| Field | Value |
|---|---|
| **Status** | Implemented; client-GameTest-pinned. Adversarial review pending |
| **Date** | 2026-08-31 |
| **Authority** | `mod/src/client/.../FlyAnimation.java` (the declared mapping and every presentation constant), `FlyPresentationSmoother.java` (the per-entity filter), `FlyModel.setupAnim` (the drawn body), and `FlyRenderer.extractRenderState` / `applyBodyPose` (the render-only pose); `FlyClientGameTest.presentationVisualGainJson` (the receipt's presentation-gain marker, D-074) |
| **Precedent** | ADR 0003 (bounded actuation; the renderer has never been in the control path); ADR 0008 (mode taxonomy — this is presentation, and presentation is not a mode); ADR 0015 (the body-frame handedness this ADR's bank sign is derived from and pinned against) |
| **Evidence** | Seven client-GameTest groups in `FlyAnimationClientGameTest`, run by `make fabric-client-test`, plus one executed negative control against the reverted metronome |

## Context

The client already receives everything it needs and throws it away.

`FlyEntity` synchronizes the applied five-axis motor vector as entity data every
tick (`FlyEntity.java:34-43`, published through `presentationAction()` at
`FlyEntity.java:116-123`), and `FlyRenderer.extractRenderState` already copied it
into the render state. `NeuroCraftHud` already printed it as five numbers. But
`FlyModel.setupAnim` was, in full:

```java
float flap = (float) Math.sin(state.ageInTicks * 2.4F) * 0.48F;
leftWing.zRot = -0.25F - flap;
rightWing.zRot = 0.25F + flap;
float antennaSweep = (float) Math.sin(state.ageInTicks * 0.18F) * 0.08F;
leftAntenna.yRot = -0.18F + antennaSweep;
rightAntenna.yRot = 0.18F - antennaSweep;
```

A function of `ageInTicks` alone. The wings beat on a metronome whatever the
connectome commands; the six legs and the abdomen were baked once and never moved
again; and the body has no roll axis at all, server-side or client-side. Combined
with `pitch_torque` being hardwired to `0.0` in the pinned wing-differential
projection (`controller.py:443`), the fly rendered as a level brick sliding
around. **Nothing on screen changed when the connectome did something.**

That is a presentation defect with two costs. The operator's standing complaint is
that the fly "doesn't seem smart", and a metronome cannot look otherwise. And every
future steering change was visible only in a trace — there was no way to watch the
body and see whether a command had arrived.

## Decision

Drive the model from the motor vector the client already has, entirely inside
`mod/src/client`, and pin the mapping with client GameTests.

### The mapping, axis by axis

| Motor axis | Visual degree of freedom | Direction |
|---|---|---|
| `forward_thrust`, `vertical_thrust` (magnitude) | wing-beat **amplitude** and **rate** | harder work, harder beat |
| `yaw_torque` | wing **stroke-amplitude and stroke-plane-pitch differential** | the *outside* wing of the turn works harder |
| `yaw_torque` | render-only body **bank** | the body leans *into* the turn |
| `vertical_thrust` | **leg tuck / extension** (fore-aft swing and splay) | a climb tucks, a descent extends |
| `forward_thrust` | **abdomen pitch** | thrust raises it, braking drops it |
| — | antenna idle sweep | unchanged; still `ageInTicks` only |

`lateral_thrust` and `pitch_torque` drive nothing. Both are hardwired to `0.0`
under the pinned wing-differential projection (`controller.py:440`, `:443`), so
neither carries a reading worth drawing on that path, and inventing one would put
motion on screen that no profile commands.

**Every profile benefits.** The mapping reads the published `PrimitiveAction` and
nothing else. It does not know which controller produced it, and no profile,
mode, or control is special-cased anywhere in the client.

**One consequence of the pinned projection, disclosed.** That same projection sets
`forward_thrust` and `vertical_thrust` to the *identical* number
(`clip(common)`, `controller.py:439` and `:441`). The effort channel is
`sqrt(forward^2 + vertical^2)` saturated at unit, so on that profile it is
`sqrt(2) * |common|` and saturates at `|common| = 0.7071` rather than at the rail.
That is a property of the profile, not of this mapping — on the trained-readout
and engineered-encoder projections the two axes are independent — and the effect
is only that a hard-working fly on the pinned profile reaches full wing amplitude
somewhat before it reaches full command.

### Declared presentation constants

Every number below is presentation engineering. None is derived from the
connectome, from fly biology, or from any measurement of the fly's behavior, and
none changes a command.

| Constant | Value | Rationale |
|---|---|---|
| `SMOOTHING_TAU_TICKS` | 3.0 ticks (150 ms) | Commands arrive at 20 Hz and the client draws faster, so the raw channel is a staircase. Slow enough to erase the step, fast enough that a command change is on screen inside a tenth of a second |
| `MAX_SMOOTHING_STEP_TICKS` | 20.0 ticks | Bounds the catch-up after a pause, chunk reload, or dropped frame, so the body eases in rather than snapping |
| `YAW_TORQUE_VISUAL_REFERENCE` | 0.035 | The yaw magnitude drawn at full visual deflection — see *The visual gain* below |
| `MAX_BANK_DEGREES` | 25.0 | A bank a viewer reads as "leaning into the turn" without the model clipping its own shadow |
| `WING_BEAT_RATE_IDLE` | 2.4 rad/tick | The pre-ADR-0016 rate verbatim, so a hovering fly beats exactly as it always did |
| `WING_BEAT_RATE_GAIN` | 1.2 rad/tick | A fly at full effort beats 1.5x as fast as one hovering |
| `WING_BEAT_AMPLITUDE_IDLE` | 0.36 rad (~21 deg) | Below the pre-ADR-0016 fixed 0.48, so an uncommanded fly reads as calm rather than frantic — it still beats through a full stroke |
| `WING_BEAT_AMPLITUDE_GAIN` | 0.24 rad | Maximum stroke is the 0.60 rad (~34 deg) sum |
| `WING_DIFFERENTIAL_AMPLITUDE` | 0.18 rad | Kept strictly below `WING_BEAT_AMPLITUDE_IDLE` so the inside wing can never invert its stroke |
| `WING_DIFFERENTIAL_PITCH` | 0.20 rad | Stroke-plane tilt difference between the two wings at full visual yaw |
| `LEG_TUCK_SWING` | 0.55 rad | Fore-aft leg swing between a full climb and a full descent |
| `LEG_TUCK_SPREAD` | 0.22 rad | Splay change; kept below the 0.65 rad rest splay so the legs never cross the midline |
| `ABDOMEN_PITCH_SWING` | 0.30 rad | Abdomen pitch between full thrust and full braking |
| `SMOOTHER_CACHE_LIMIT` | 64 entries | Bound on the client-side smoother cache, so it cannot grow without limit |

Eight further constants — `WING_REST_SPLAY` 0.25, `WING_REST_PITCH` -0.08,
`LEG_REST_PITCH` -0.18, `LEG_REST_SPLAY` 0.65, `ABDOMEN_REST_PITCH` -0.10,
`ANTENNA_IDLE_RATE` 0.18, `ANTENNA_IDLE_SWEEP` 0.08, `ANTENNA_REST_YAW` 0.18 —
are the pre-ADR-0016 baked-pose values lifted verbatim out of the mesh and the old
`setupAnim` and given names. They are not new choices.

### The visual gain

**This is the finding most likely to be challenged, and it should be.**

The yaw torque the fly actually commands is tiny. ADR 0015 records the eem-v1
sided boundary at `+0.0054` and `-0.0346`; the D-067 arena sizing records the
measured yaw as **0.5-3.5% of the +/-1 rail**. Mapped linearly onto a bank, the
entire committed live band would occupy under one degree of roll and the fly would
stay the level brick it is today. The animation would be technically correct and
would show nothing.

So the yaw channel is rescaled for the screen: it is divided by the declared
`YAW_TORQUE_VISUAL_REFERENCE = 0.035` — the top of that recorded band — and
clamped to `[-1, +1]`. A command at its recorded live maximum draws at full bank;
anything larger saturates. The implied display scaling is about **28.6x**.

Stated as plainly as it can be:

- **The gain scales pixels, not commands.** It lives in `FlyAnimation`, a class
  of pure static functions over primitives, downstream of everything. The number
  that reaches `PrimitiveActuator` is the number the controller emitted,
  unscaled and unmodified. No trace, receipt, identity, or hash sees this value.
- **It adds no information.** It is the same number, drawn larger. A rendered
  bank is a display of `yaw_torque` at a declared magnification, exactly as the
  HUD's `TORQUE YAW %+1.2f` is a display of it at a declared precision.
- **A banking fly is not a hard-steering fly, and must never be described as
  one.** A fly drawn at full 25-degree bank is executing a yaw torque of about
  3.5% of its authority. Any statement of the form "the fly turned sharply",
  "the fly banked hard", or "the steering is much stronger now" that rests on
  watching the rendered body is **wrong**, and no such statement may enter
  `STATUS.md`, a receipt, a decision row, or the paper. The only admissible
  reading of the bank is qualitative and signed: *a yaw command is present, and
  it points this way.* Magnitude comes from the trace.

The same caveat applies, less acutely, to every other channel: wing amplitude,
leg tuck and abdomen pitch are all drawn against their own declared ranges, and
none of them is a measurement.

### The bank sign, and why it is derived once and pinned

The entity has **no roll axis at all**. `PrimitiveAction` has five channels and
none of them is roll; `PrimitiveActuator` cannot roll the body; `FlyEntity` stores
no roll. The bank is a rotation applied to the `PoseStack` inside
`FlyRenderer.applyBodyPose` and it exists nowhere else in the repository.

Its direction depends on a handedness chain that ADR 0015 spent an ADR getting
right, so it is derived here and then pinned by a test that does not trust the
derivation. In `FlyModel` coordinates, after the renderer's `(-0.42, -0.42, +0.42)`
scale and the standard `YP(180 - yaw)` offset, model `+X` is the body's **left**
(consistent with the mesh: `left_wing` sits at `x = +1.6`), model `+Y` is
world-**down**, and model `+Z` is the body's **back**. A positive rotation about
model `+Z` therefore lowers the body's *left* side. Positive `yaw_torque` turns
the body toward its own **right** (ADR 0015: increasing yaw walks the heading
`SOUTH -> WEST -> NORTH -> EAST`, i.e. clockwise, i.e. rightward), and a body
banking *into* a right turn drops its right side. So a positive visual yaw maps to
a **negative** `Axis.ZP` angle.

`assertBankLowersTheInsideOfTheTurn` does not assert that sentence. It builds the
real `PoseStack` through `FlyRenderer.applyBodyPose`, transforms the two wing-tip
model points through the resulting matrix — scale, yaw offset and all — and
asserts that the world-space height of the **right** wing tip is below the left
under a positive yaw command, and mirrored under its negation. If the derivation
above is wrong anywhere in the chain, that test fails.

### Smoothing

Commands are 20 Hz; the client draws faster. Every motor channel is passed through
a frame-rate-independent exponential filter, `1 - exp(-dt / SMOOTHING_TAU_TICKS)`,
so the smoothed value follows the same trajectory whether the client draws at 30
or 300 frames per second. The wing-beat phase is *integrated* rather than computed
from a clock (`phase += dt * rate(effort)`), because the beat rate is
command-dependent and a phase computed as `age * rate` would jump discontinuously
whenever the rate changed.

`dt` comes from the entity's own interpolated tick clock, `tickCount + partialTick`,
so a paused game freezes the animation rather than racing it, and it is clamped
into `[0, MAX_SMOOTHING_STEP_TICKS]`.

The filter state is per-entity and therefore has to live somewhere across frames.
Minecraft reuses one render state per renderer, so it lives in a bounded
`HashMap<Integer, FlyPresentationSmoother>` owned by `FlyRenderer`, keyed by entity
id. The fly is a singleton in every declared scenario; the bound exists only so a
client-side cache cannot grow without limit, and on overflow the cache is dropped
whole, which costs at most one smoothing restart.

## The render-only invariant

`FlyRenderer`'s class comment already asserted that it "consumes synchronized
presentation fields and mutates no entity state". This ADR adds animation and
mutable per-entity smoothing behind that sentence, so the sentence is now load
bearing and is stated as a checkable invariant:

1. **Every call the render path makes on `FlyEntity` is a getter** — `getYRot`,
   `getXRot`, `controllerState`, `presentationAction`, `tickCount`, `getId`.
   There is no setter, no `setDeltaMovement`, no `setPos`, no command emission.
2. **`FlyAnimation` cannot touch entity state**, structurally: it is a final
   class of static methods whose parameters are `float`s and one already-published
   `PrimitiveAction`. It holds no reference to an entity, a level, or the bridge.
3. **`FlyPresentationSmoother` is client-side presentation memory.** It is
   written only from `extractRenderState` and read only by the model and the
   pose. Nothing it holds is transmitted, traced, hashed, or readable by the
   server.
4. **No *measurement* identity moves; one code bundle does.** No connectome,
   adapter, decoder, encoder, calibration, arena, pilot, or embodiment identity
   binds a client render source file. One declared identity does: the
   `neurocraft.gametest_code_bundle.v1` `code_hash`
   (`scripts/prepare_gametest_demo.py`, `_code_bundle()`) hashes all of `mod/src`
   recursively, so these files are inside it and this lane moves it from
   `a59eef49…a330d` to `43881144…ad75`. That bundle had already drifted from every
   committed receipt before this lane (D-068), and no test pins a literal — all
   compute it dynamically — so nothing is newly invalidated; but the absolute
   "no identity moves" would be false and is not claimed. The changed files are
   `mod/src/client/**` and one client GameTest; the mod's control-adjacent server
   sources are untouched by this lane.
5. **`assertRenderExtractionLeavesEntityStateUntouched`** drives the real
   registered `FlyRenderer`'s `extractRenderState` over a real client-side
   `FlyEntity` for 40 sub-tick frames under a neutral command — it does not
   exercise `render()` or `setupAnim` against a live entity, so points 1-3 above
   (verified structurally: every entity interaction in the changed sources is a
   getter, and `FlyAnimation` holds no entity reference by construction) carry
   the rest of the invariant. It asserts that position, both rotations, delta
   movement, published action,
   controller state and tick count are unchanged, while confirming the same call
   did advance the wing beat.

Point 5 is the one that would catch a regression; points 1-4 are what make it
hard to write one.

## Regression pins

Client GameTest lane 1 -> 2 (`FlyAnimationClientGameTest` joins the existing
`FlyClientGameTest`, which is untouched and still produces the renderer-observation
receipt). **The server GameTest lane is unchanged at 26** — this lane adds no
server test and removes none.

- `assertWingBeatTracksThrust` — at a fixed clock and a fixed phase, stroke
  amplitude must widen with thrust effort, must equal the declared mapping, must
  respond identically to the vertical axis as to the forward one, and the beat
  *rate* must advance further per tick under thrust.
- `assertWingBeatStaysAliveAtZeroCommand` — an uncommanded fly must still beat
  through a full down-and-up stroke spanning exactly the declared idle amplitude.
  A hovering fly must not freeze.
- `assertWingDifferentialTracksYawTorque` — symmetric at zero torque; in a right
  turn the outside (left) wing out-strokes the inside one and the two differ in
  stroke-plane pitch; the inside wing never inverts; and the mirrored command
  mirrors the body exactly.
- `assertLegsAndAbdomenTrackTheirAxes` — a climb tucks the legs back and in and a
  descent extends them forward and out; the two banks stay mirrored; all three
  legs on a side share the pose; abdomen pitch tracks forward thrust and returns
  to its declared rest at zero command.
- `assertBankIsZeroSignedAndClamped` — exactly zero bank at zero torque; the
  smallest recorded live magnitude (`0.0054`) still banks several degrees;
  opposite torques bank in opposite directions; the rails and well past them
  clamp to exactly 25 degrees; the declared reference amplitude maps to exactly
  full bank.
- `assertBankLowersTheInsideOfTheTurn` — the end-to-end pose test described above.
- `assertRenderExtractionLeavesEntityStateUntouched` — invariant point 5.

**One negative control, executed** (not merely reasoned). `FlyModel.setupAnim` was
temporarily reverted to the pre-ADR-0016 metronome body — the seven lines quoted
at the top of this ADR, verbatim — and, so that the whole matrix could be read
from one run rather than stopping at the first failure, `runTest` was temporarily
wrapped to run every group and collect its error. `make fabric-client-test` then
printed:

```
NEGATIVE CONTROL MATRIX
FAIL assertWingBeatTracksThrust :: wing stroke must widen with thrust effort at a fixed phase, but hovering=0.45381606 working=0.45381606
FAIL assertWingBeatStaysAliveAtZeroCommand :: a fly with no command must still beat its wings through a full stroke, but the stroke was down=0.45381606 up=0.45381606
FAIL assertWingDifferentialTracksYawTorque :: in a right turn the outside (left) wing must out-stroke the inside one, but left=0.45381606 right=0.45381606
FAIL assertLegsAndAbdomenTrackTheirAxes :: a climbing fly must tuck its legs back relative to a descending one, but climb=-0.18 descent=-0.18
PASS assertBankIsZeroSignedAndClamped
PASS assertBankLowersTheInsideOfTheTurn
```

**All four model-driven groups failed and the two pose/bank groups passed**, which
is the expected partition: the bank groups exercise `FlyAnimation` and
`FlyRenderer.applyBodyPose` and never call `setupAnim`, so reverting the model
cannot move them. `assertRenderExtractionLeavesEntityStateUntouched`, which runs
outside the harness, also passed for the same reason.

The number `0.45381606` is itself the measurement: it is the *identical* stroke the
metronome produced for hovering, for full thrust, for both ends of the beat, and
for both wings in a turn, because `sin(100 * 2.4) * 0.48` does not depend on the
command. `-0.18` is the baked leg pitch, unchanged between a full climb and a full
descent because the old body never posed a leg. Both temporary edits were then
reverted and the suite re-run green.

## Consequences

- **This changes what the fly looks like and nothing else.** It visualizes
  connectome output. It adds no information, changes no action, rotates no
  identity, and alters no measurement. **It must never be described as the fly
  behaving differently, steering differently, or responding differently.** If a
  future live session looks livelier than the last one, that is this ADR and not
  the connectome, unless a trace says otherwise.
- **The declared visual gain is the standing hazard.** A rendered bank is a
  ~28.6x magnification of a real command. It is legitimate presentation and it is
  also exactly the kind of thing that gets misread as "the fly is steering hard".
  The prohibition in *The visual gain* above is the mitigation and applies to
  every future record, receipt, screenshot caption, and paper figure.
- **No receipt is edited and none is superseded.** `EXTERNAL-BRIDGE-01..06`,
  `LIVE-CIRCUIT-01`, the pilots and the embodiment receipts bind server-side and
  Python material that this lane does not touch. The client renderer-observation
  receipt written by `runClientGameTest` (`fly-renderer-observation.json`/`.png`)
  is regenerated on every run into the untracked `mod/build` tree and is not
  committed evidence. **The gap disclosed here is now closed (D-074).** As first
  written, the `neurocraft.renderer_observation.v1` schema carried `automated` and
  `manual_gameplay_observation` flags but no presentation-gain marker, so a
  banked-fly PNG lifted from that path into a figure or a message travelled
  without the visual-gain caveat above. The schema is now
  `neurocraft.renderer_observation.v2` and every receipt carries
  `bank_is_display_scaled: true` beside a `presentation_visual_gain` object naming
  the channel (`yaw_torque`), what it is drawn as (`body_bank_degrees`), the
  reference amplitude, the maximum bank in degrees, the implied gain, and the
  prohibition in *The visual gain* above in prose. Its numbers are read from
  `FlyAnimation.YAW_TORQUE_VISUAL_REFERENCE` and `FlyAnimation.MAX_BANK_DEGREES`
  rather than restated, so a changed presentation constant cannot leave the marker
  describing the old mapping; `runClientGameTest` fails the build if the receipt it
  just wrote is missing any part of the marker, and
  `tests/governance/test_renderer_observation_marker.py` pins the writer, that
  Gradle refusal, and this paragraph. **Two boundaries remain, and they are not
  claims to the contrary.** The marker lives in the JSON: the PNG written beside it
  carries no embedded caveat, so an image separated from its receipt still needs
  the caveat supplied by whoever moves it, and any image exported before this
  change was written under v1 and has no marker at all. The committed
  `RENDERER-CLIENT-01.json` is v1 and is **not** retrofitted — it is history, and
  receipts here are never edited. The receipt's warm-pixel counts also vary run to
  run (155 and 156 were both observed during review), so no record may cite them as
  a measurement.
- **The screenshot delta the existing client GameTest measures still passes** with
  the new animation, but its thresholds are amplitude-sensitive by construction
  (warm-pixel counts and bounds). A future change to the declared amplitudes
  should re-run `make fabric-client-test` rather than assume that test is
  insensitive to them.
- **Live re-observation is pending.** No claim is made here about how the animated
  body reads to a human. Whether a bank at the measured yaw magnitudes actually
  makes the fly look alive is unmeasured and is not asserted.
