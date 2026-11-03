# Baseline boundary

`scripted_baseline` is an explicit non-neural control with the same raw physical sensor and primitive-action budgets as the closed loop. It is never a fallback for Faithful mode, never emits semantic actions, and must be reported as `SCRIPTED_BASELINE` in manifests, traces, dashboards, and results.

The primary negative-control matrix also includes fixed-support weight shuffling, directed degree-preserving rewiring, lesions, a null graph, frozen neural state, and a separately labelled synthetic graph. Matching arena, reset, seed, sensor schema, actuator bounds, tick budget, and evaluation code is mandatory.
