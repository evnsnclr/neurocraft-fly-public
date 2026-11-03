package dev.neurocraft.fly.control;

import dev.neurocraft.fly.body.ProgramExecution;
import dev.neurocraft.fly.entity.ActuationReceipt;
import java.util.Optional;
import java.util.OptionalLong;

/** Transport boundary used by the server-authoritative entity. */
public interface ControllerBridge extends AutoCloseable {
    void start();

    void offerObservation(PhysicalSensorSnapshot snapshot);

    Optional<CommandFrame> pollAction();

    /** Lookup from an action's observation sequence to the exact frame sent on this connection. */
    OptionalLong sentObservationTick(long observationSequence);

    void offerActuationReceipt(
            ActionFrame command,
            PhysicalSensorSnapshot worldBefore,
            ActuationReceipt receipt,
            ValidationResult decision);

    /**
     * Sends the authoritative report for one admitted {@code program_action}.
     *
     * <p>A program tick has no honest {@code actuation_receipt} (ADR 0020 section 3), so it gets its
     * own frame. {@code execution} is empty exactly when no program ran: a rejected envelope. A
     * refused-but-executed tick carries the execution that actually happened, which is what makes
     * the {@code reason=operator} receipt truthful rather than a fabricated HOVER.
     */
    void offerProgramReceipt(
            ProgramActionFrame command,
            PhysicalSensorSnapshot worldBefore,
            Optional<ProgramExecution> execution,
            ValidationResult decision);

    OptionalLong requestControl(
            long issuedTick, ControlOperation operation, String contentId, String contentHash);

    Optional<ControlAckFrame> pollControlAck();

    /** Removes only the acknowledgement bound to the requested control key. */
    Optional<ControlAckFrame> pollControlAck(long controlSequence, ControlOperation operation);

    /** Count of canonical actuation receipt frames successfully handed to the WebSocket. */
    long sentActuationReceiptCount();

    /** Count of canonical program receipt frames successfully handed to the WebSocket. */
    long sentProgramReceiptCount();

    OptionalLong lastSentActuationReceiptCommandSequence();

    /** Terminal validation fault held open only until its authoritative receipt is sent. */
    Optional<ControllerFault> pendingTerminalFault();

    /** Propagates final server-thread validation failure to the transport's latched neutral state. */
    void failClosed(ControllerFault fault);

    ControllerFault state();

    @Override
    void close();
}
