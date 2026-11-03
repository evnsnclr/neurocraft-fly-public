# Neural service boundary

This package owns the authenticated, loopback-only protocol endpoint. Its default
`DeterministicNeutralController` is a **synthetic transport test double**: it always
emits a neutral primitive action and cannot support a Minecraft, neural-engine, or
scientific-fidelity claim. The integrated neural runtime must be injected through the
`Controller` interface.

Every issued action remains pending until Minecraft returns a bound
`ActuationReceipt`; forged, reordered, expired, or mismatched receipts fail closed.
Callers may inject a `receipt_sink` to append authoritative application evidence.

Authenticated `ControlCommand` frames implement pause, resume, one observation step,
and reset. Paused, replaying, and faulted states emit all-zero primitive actions
without advancing the neural interval. Reset succeeds only for a controller that
explicitly implements `ResettableController`. Recording and replay lifecycle commands
are delegated to an injected `ControlHandler`; without one, the service returns a
rejected `ControlAck` and enters the neutral paused state.

The run token is supplied only through `NEUROCRAFT_RUN_TOKEN`; it is absent from CLI
arguments, object representations, logs, traces, and identity files. Every reconnect
requires the caller to provide a new `session_epoch` identity.
