package dev.neurocraft.fly.control;

import dev.neurocraft.fly.body.ProgramExecution;
import dev.neurocraft.fly.entity.ActuationReceipt;
import java.util.Optional;
import java.util.OptionalLong;

/** Permanently neutral bridge used when required run configuration is absent or invalid. */
public final class NeutralBridge implements ControllerBridge {
    private final ControllerFault state;

    public NeutralBridge(ControllerFault state) {
        if (!state.isPaused()) {
            throw new IllegalArgumentException("neutral bridge requires a paused fault state");
        }
        this.state = state;
    }

    @Override
    public void start() {}

    @Override
    public void offerObservation(PhysicalSensorSnapshot snapshot) {}

    @Override
    public Optional<CommandFrame> pollAction() {
        return Optional.empty();
    }

    @Override
    public OptionalLong sentObservationTick(long observationSequence) {
        return OptionalLong.empty();
    }

    @Override
    public void offerActuationReceipt(
            ActionFrame command,
            PhysicalSensorSnapshot worldBefore,
            ActuationReceipt receipt,
            ValidationResult decision) {}

    @Override
    public void offerProgramReceipt(
            ProgramActionFrame command,
            PhysicalSensorSnapshot worldBefore,
            Optional<ProgramExecution> execution,
            ValidationResult decision) {}

    @Override
    public OptionalLong requestControl(
            long issuedTick, ControlOperation operation, String contentId, String contentHash) {
        return OptionalLong.empty();
    }

    @Override
    public Optional<ControlAckFrame> pollControlAck() {
        return Optional.empty();
    }

    @Override
    public Optional<ControlAckFrame> pollControlAck(
            long controlSequence, ControlOperation operation) {
        return Optional.empty();
    }

    @Override
    public long sentActuationReceiptCount() {
        return 0;
    }

    @Override
    public long sentProgramReceiptCount() {
        return 0;
    }

    @Override
    public OptionalLong lastSentActuationReceiptCommandSequence() {
        return OptionalLong.empty();
    }

    @Override
    public Optional<ControllerFault> pendingTerminalFault() {
        return Optional.empty();
    }

    @Override
    public void failClosed(ControllerFault fault) {}

    @Override
    public ControllerFault state() {
        return state;
    }

    @Override
    public void close() {}
}
