package dev.neurocraft.fly.control;

/** Closed set of frames accepted from the neural service. */
public sealed interface InboundFrame
        permits ActionFrame, ProgramActionFrame, ControlAckFrame, FaultFrame, HelloAck {}
