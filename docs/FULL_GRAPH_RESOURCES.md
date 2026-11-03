# Full MaleCNS runtime resource assessment

_Current-machine feasibility and staged optimization decision, 2026-08-17_

**Historical assessment.** The decisions and measurements below describe the
early unsigned implementation on 2026-08-17. They are not installation requirements
or the current scripted-program architecture. The full-network demo later used a
native CPU kernel; see the [current model card](MODEL_CARD.md),
[release scope](RELEASE_SCOPE.md) and [evidence availability](EVIDENCE.md).
The former research-gate ladder was retired as a goal by
[ADR 0019](adr/0019-pivot-to-interactive-connectome-demo.md).

## Decision

The full `male-cns-v1.0-superclass-v1` artifact fits on the existing development
machine. No hardware purchase is justified by current evidence. Capacity is not the
present blocker: the measured three-step unsigned engineering implementation is too
slow for the declared real-time interval, and it is not yet a faithful signed
runtime.

Proceed in three bounded stages: first implement and benchmark an array-native CPU
kernel on the current Mac; then test a graph-resident Metal backend only if the CPU
candidate misses the gate; then use one bounded 24--32 GB CUDA machine as a fallback
only if both local paths fail. Until a backend passes every numerical, causal,
latency, queue, and memory gate, retain the full graph as slowed/offline
`MALECNS_FULL` and use the disclosed MaleCNS-derived fixture/circuit for the live
Minecraft path.

## Evidence classes

### Measured facts

The current host was inspected as an Apple M4 Max with 48 GB unified memory and
approximately 80 GiB of free local storage at inspection time. Free storage is a
time-varying observation, not a guaranteed allocation. The existing ignored full
artifact and source material already reside locally; redistribution is not approved
by the current receipt.

The authoritative [RUNTIME-MCNS-01 receipt](./receipts/RUNTIME-MCNS-01.md) records:

| Measure | Observed value |
|---|---:|
| Retained nodes | 166,700 |
| Retained directed edges | 25,582,938 |
| Eight packed artifact arrays | 619,324,920 B (590.63 MiB) |
| Artifact directory including manifest | 619,331,728 B |
| Cold build 1 / build 2 | 40.9386 s / 39.8532 s |
| Verified hash load 1 / load 2 | 0.2862 s / 0.2988 s |
| Full NumPy layout verification 1 / 2 | 0.9501 s / 0.8366 s |
| Peak whole-builder-process RSS 1 / 2 | 6,050,217,984 B / 6,387,580,928 B |
| Unsigned all-edge smoke mean | 0.229091 s per step |

The two builds were byte-identical across all nine artifact files. Peak RSS is the
whole preprocessing process, not incremental or steady-state controller RSS.

The smoke executed all 25,582,938 retained edges for only three timed steps
(`0.239217`, `0.223034`, and `0.225021` seconds). It used a Python-orchestrated,
chunked NumPy engineering loop, created large temporaries, and omitted
transmitter-derived signs. It did not include a faithful dynamics overlay,
Minecraft, WebSocket transport, trace I/O, a warm-up, percentile statistics, or a
10,000-interval soak. It is evidence that this implementation misses real time; it
is not a lower bound on optimized hardware and is not a faithful full-graph run.

### Calculations from the measured counts

For `N = 166,700` and `E = 25,582,938`, one destination-oriented hot view comprising
`uint64[N+1]` row offsets, `uint32[E]` presynaptic indices, and effective weights has
these raw contiguous sizes:

| Runtime view | Exact bytes | Binary size |
|---|---:|---:|
| `float32` effective weights | 205,997,112 B | 196.45 MiB |
| `float64` effective weights | 308,328,864 B | 294.05 MiB |
| One `float32[N]` state field | 666,800 B | 0.636 MiB |
| One `float64[N]` state field | 1,333,600 B | 1.272 MiB |

If the existing 619,324,920-byte provenance artifact and the new hot view coexist,
their raw byte sums are approximately 787.1 MiB with `float32` weights or 884.7 MiB
with `float64` weights, before state, signs, delays, checkpoints, page residency,
libraries, allocator overhead, and work buffers. This coexistence calculation is not
a measured RSS result. Provenance arrays may remain memory-mapped or outside the hot
process, but they must remain content-addressed and auditable.

The `0.229091 s` smoke mean is:

- `22.91x` slower than a 10 ms budget for one 100 Hz neural step; and
- `32.73x` slower than the arithmetic 7 ms-per-step allowance obtained when five
  10 ms steps must fit inside the plan's 35 ms p99 full-interval target.

Those factors compare a three-sample mean with target budgets, not like-for-like p99
measurements. They size the optimization problem; they cannot qualify a backend.

### Engineering hypotheses to test

- A compiled or otherwise array-native CPU kernel should remove Python edge-chunk
  iteration, repeated `bincount` work, and large per-step allocations. The size of
  that improvement is unknown until benchmarked.
- M4 Max unified memory is ample for a graph-resident hot view, but Metal sparse
  coverage, determinism, launch overhead, and the observed degree distribution may
  limit useful speedup. Dense-GPU capability is not evidence for this sparse kernel.
- A 24--32 GB CUDA device has ample nominal capacity for the calculated hot set, but
  it is only a fallback benchmark resource. It must retain graph/state on device,
  avoid silent CPU fallback, and beat the optimized CPU end to end; device capacity
  alone does not justify rental or purchase.
- Event/frontier execution may help only if measured activity is sparse and the
  selected equations admit omission or lazy updates without changing results. It
  cannot be introduced as an unrecorded approximation.

## Staged execution plan

### Stage 1 — array-native CPU on the existing M4 Max

1. Materialize a content-addressed model-runtime view from the immutable full
   artifact: destination offsets, presynaptic indices, signed/normalized effective
   weights, declared delays, boundary masks, and stable-ID maps.
2. Keep the current single-thread `float64` implementation as the correctness
   oracle. Add a compiled/array-native full-graph candidate with fixed threading and
   no Python object adjacency or per-edge Python loop.
3. Preallocate current state, next state, recurrent accumulation, delay buffers, and
   bounded telemetry workspaces. After warm-up, forbid unbounded per-step allocation.
4. Benchmark `float64` first. Consider `float32` only after the numerical-frequency,
   decoded-action, and replay tolerances pass; record both identities separately.
5. Capture cold view construction, verified warm load, first-touch time, resident
   memory, per-step timing, complete 50 ms intervals, queue growth, and trace
   overhead as distinct measurements.

This stage needs engineering time and disk for immutable benchmark artifacts, not a
new computer.

### Stage 2 — graph-resident Metal candidate

Attempt Metal only after the CPU result and profiler identify the sparse update as
the limiting stage. Keep graph, weights, state, and delay buffers resident in unified
memory; transfer only bounded sensor values, motor outputs, and sampled telemetry.
Time work after explicit device synchronization, record compilation/upload
separately, and detect any CPU fallback as a failed run. The Metal backend must use
the same fixed equations and content identities as the CPU reference.

### Stage 3 — one bounded CUDA fallback

If the optimized CPU and Metal candidates both miss the gate, run the same frozen
benchmark once on a reproducible 24--32 GB CUDA environment. Start with rental or
otherwise temporary access; current evidence does not support a purchase. Pin the
GPU, driver, CUDA, framework/kernel, deterministic settings, clocks, and machine
identity. Keep the full hot view resident and include host/device synchronization in
end-to-end percentiles. A CUDA pass qualifies only that named environment until
reproduced elsewhere.

## Acceptance metrics

Use the benchmark matrix and procedure in
[the historical performance-plan reference](EVIDENCE.md#historical-references): five randomized-order
repetitions, at least 1,000 warm-up intervals, and 10,000 timed 50 ms intervals with
raw artifacts retained. At minimum, require all of the following:

| Gate | Required result |
|---|---|
| Neural interval | 50 ms simulated interval below 35 ms p99 and 45 ms p99.9 |
| Queue/cadence | No growing queue, silently skipped/dropped substeps, or equation change under load |
| Observation to applied action | Below 40 ms p99 over 10,000 intervals |
| Server impact | Callback below 1 ms p99 / 2 ms p99.9; enabled-minus-baseline p99 below 2 ms; no controller-attributable tick above 50 ms |
| Runtime memory | Total process RSS below 2 GiB and incremental RSS no more than `2 * runtime_array_bytes + 256 MiB` |
| Steady allocation | RSS growth below 1% or 16 MiB, whichever is larger, after warm-up |
| Protocol | 4 KiB at 20 Hz loopback below 5 ms p99; endpoint encode/validate below 1 ms p99 |
| Telemetry | At most 250 MiB/hour by default, no tick-thread I/O, no missing mandatory record |
| Recovery | Neutral within one game tick; verified warm full graph ready below 10 s |
| GPU promotion | At least 2x faster p99 end to end than optimized CPU; at most 80% device/unified memory |

Report native neural computation, bridge time, Minecraft tick impact, and encoded or
presentation rates separately. A fast kernel does not imply a fast closed loop.

## Scientific and anti-bypass gates

Performance optimization cannot change the scientific controller:

- Preserve every retained directed edge, its direction, source count, stable source
  identifier, and content identity. Any scientifically motivated filter or topology
  change creates a new artifact and requires an ADR; deadline pressure is not such a
  motivation.
- Bind graph, sign, dynamics, delay, sensory-adapter, decoder, observation-schema,
  and arena identities at startup and fail closed on mismatch.
- Use only named sensory boundary populations for input and named motor/descending
  pools for output. No target steering, PID/heading hold, vanilla goal, behavior
  tree, scripted fallback, LLM, hidden learned policy, or semantic actuator may
  enter the Faithful action path.
- Run the same equations, exact configured number of substeps, intervention
  semantics, and primitive-motor decoder on every backend. No edge pruning, stale
  hold-last behavior, silent CPU fallback, approximation, or adaptive step dropping
  is allowed.
- Establish numerical agreement against the `float64` reference, exact or declared
  tolerant replay, trace integrity, fail-neutral behavior, and every applicable
  AB-01 through AB-12 test. In particular, full-graph speed cannot substitute for
  the mandatory anti-bypass mutants.

The optimization lane can establish a named full-runtime G2/performance subgate.
Only a traced Minecraft closed loop plus the causal gates can support G3, and only a
10,000-interval pass can add the G3-R real-time qualifier. Until then, the accurate
wording remains **slowed/offline full MaleCNS graph** and **live MaleCNS-derived
fixture/circuit**. See [ADR 0002](./adr/0002-neural-dynamics.md),
[ADR 0004](./adr/0004-causal-trace-and-replay.md),
[ADR 0005](./adr/0005-fixture-and-claim-gates.md), and
the historical project plan (not distributed; see
[historical references](EVIDENCE.md#historical-references)).
