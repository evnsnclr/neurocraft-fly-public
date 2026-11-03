# Reproduce the release comparison

This is an automated, embodied diagnostic for the current six-input profile. It
is separate from a human's normal-client installation/play test and from the
14-neuron notebook example. It does not establish biological validity. The
controller still selects and modulates scripted body programs through hand-chosen
readouts. See [model card](MODEL_CARD.md) and [manual experiments](EXPERIMENTS.md).

## Fixed protocol

The plan contains 54 core cases: six sources crossed with original weights,
shuffled weights (seed 7), and no sensory input, repeated in three blocks. Each
block has 18 core cases and a separate restored-original brush probe. Lane order
rotates between blocks. Every case gets a fresh neural epoch; physical preparation
then waits for ten eligible, matched observations with stable light and event
counters. Other sensory history may differ as the body moves.

Approach uses five scripted positions (5.5, 4.5, 3.5, 2.5 and 1.5 blocks) relative
to a marker anchored once at the fly's initial pose. The marker does not follow
later movement. Each requested horizontal position is confirmed through a fresh
server query, followed by a fresh server game-time barrier and a wait for six
observed ticks beyond it. The schedule does not
stop or change when a readout activates. Actual sensor delivery and applied
responses are checked separately. This is not keyboard walking or identical-input
replay; all closed-loop sensory histories remain available in local diagnostics.

Brush and attack use declared synthetic native client dispatch. Food, light and
mob stimuli use prepared world fixtures. Automatic aiming/scene preparation must
remain labeled in videos and results. The diagnostic client must be unobscured
and exclusively controlled during a block; an open menu may cause a refusal.

## Prepare before collecting results

From one frozen source candidate and the documented installed Python environment:

```sh
uv run --frozen python -m scripts.run_release_comparisons plan \
  --output artifacts/release-comparison-01
```

The output directory must be new. Keep `plan.json` and `plan.sha256`; the plan
pins the protocol and runner bytes before any block runs. Editing the runner or
the protocol invalidates it. A changed experiment needs a new plan, with the
earlier attempt retained. One seed does not justify claims about all shuffles.
The first block's preflight writes `models.json` before sending any stimulus.
It binds graph, adapter, dynamics, decoder and observation-schema identities for
each lane. Later blocks must match; the report checks every case and restoration
probe against those identities. Keep the exact source export manifest alongside
the plan; the three runner hashes alone do not identify the whole application.
Before scene preparation, the first newly applied baseline also binds the policy
hash in `policy.json`. Every handover, response-window row and restored probe must
match it. Policy rules are separate from the five model identity hashes.

## Run three fresh sessions

After full-network prerequisites in [INSTALL.md](INSTALL.md), launch the existing
development diagnostic client, not the manually managed production-client route:

```sh
uv run --frozen python -I scripts/run_interactive_demo.py \
  --controller-profile full_mcns_interactions_program_v2 \
  --showcase-world --comparison-lanes --control-seed 7 \
  --action-latency --kernel-threads 8 --game-tick-rate 20 \
  --client-width 1280 --client-height 720 --client-max-fps 60 \
  --diagnostic-tools --capture-fps 1 --capture-seconds 720 \
  --capture-png-compression 6
```

These are requested settings, not achieved performance. Check free disk space and
stop builds/tests before live measurement. Wait for the launcher to report the
interactive session and for the client to show Live. Close game menus. Copy the
new run directory printed by the launcher into the `--run` argument below:

```sh
uv run --frozen python -m scripts.run_release_comparisons run-block \
  --plan artifacts/release-comparison-01 \
  --run artifacts/live-interactive/ACTUAL_SESSION_DIRECTORY --block 0
```

`ACTUAL_SESSION_DIRECTORY` is a placeholder, not a literal existing directory.
Start promptly: the legacy runner retains a 110-second operator reserve and a
570-second hard execution bound inside the 720-second capture. A too-late start
is refused; a deadline preserves remaining cases as unattempted. The program never
relaxes the normal timing budget to obtain a pass.

After block 0 completes, stop that launcher/client cleanly. Repeat the launch for
block 1 and then block 2, passing a different fresh run directory each time. A
run ID cannot serve two blocks. Already-attempted block directories are not reused.
Do not overwrite a failed block with a retake. A terminal bridge fault requires
a fresh session and a separately declared subsequent experiment.

## Read every result

```sh
uv run --frozen python -m scripts.run_release_comparisons report \
  --plan artifacts/release-comparison-01 \
  --output artifacts/release-comparison-01/report.json
```

The report retains all 54 core cases even when a block never started. It separately
reports three restoration probes, source/lane counts, failures, applied targets
and qualified denominators. Output files are created exclusively, never overwritten.

A case requires a valid delivered fixture, coverage of every observation and its
applied receipt in the response window, matching input diagnostics and a verified
input gate. An approach additionally requires all five confirmed positions/holds.
An issued target response must have its complete applied hold. Missing data is
unqualified, not zero. The sidecar logs declared assignments/population drives
and stimulus hashes; it is not a dense sensory vector or hash-linked causal trace.

The underlying `program-decisions.ndjson` keeps its existing v1 schema. A separate
`program-decisions.inputs.ndjson` records command-matched declared stimulation,
including pre-control inputs and executed zero summaries in no-input mode. Writer
failure disables the sidecar without changing control; qualification then reports
missing input evidence. Input rows in the trial ledger are separate from the
legacy two-row-per-command reducer budget.

`run-block` completion means the schedule was observed/restored; use `report` for
the stricter qualification result. A retained shuffled response is valid, and
no-input can still move through fallback CRUISE. A source miss cannot demonstrate
neural suppression. The restored-original brush must qualify and produce its
applied target; configuration restoration alone is reported separately.

Keep original raw diagnostics privately. Publish only a reviewed, sanitized
summary/manifest and explanatory plots with the exact source, plan and artifact
identities. A new plan changes no historical receipt and does not authorize a
public release. Future multiple-seed, sensitivity or targeted-silencing experiments
need their own declared protocols and scope.
