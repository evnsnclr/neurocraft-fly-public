package dev.neurocraft.fly.control;

/** Frozen operator control operations; none are movement-policy or semantic motor commands. */
public enum ControlOperation {
    PAUSE,
    RESUME,
    SINGLE_STEP,
    RESET,
    RECORD_START,
    RECORD_STOP,
    REPLAY_START,
    REPLAY_STOP,
    LESION_APPLY,
    LESION_CLEAR,
    STIMULATION_APPLY,
    STIMULATION_CLEAR
}
