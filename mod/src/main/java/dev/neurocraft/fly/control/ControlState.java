package dev.neurocraft.fly.control;

/** Read-only state returned by the neural service for an operator control request. */
public enum ControlState {
    RUNNING,
    PAUSED,
    STEP_ARMED,
    RECORDING,
    REPLAYING,
    FAULTED
}
