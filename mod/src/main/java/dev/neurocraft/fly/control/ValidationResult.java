package dev.neurocraft.fly.control;

import java.util.Objects;
import java.util.Optional;

/** Result of final server-thread validation. Rejections always carry a neutral action. */
public record ValidationResult(
        boolean accepted,
        RawPrimitiveAction rawAction,
        PrimitiveAction appliedAction,
        ControllerFault state,
        Optional<CommandFrame> command) {
    public ValidationResult {
        Objects.requireNonNull(rawAction, "rawAction");
        Objects.requireNonNull(appliedAction, "appliedAction");
        Objects.requireNonNull(state, "state");
        Objects.requireNonNull(command, "command");
        if (accepted && command.isEmpty()) {
            throw new IllegalArgumentException("accepted command must retain its causal envelope");
        }
        if (!accepted && !appliedAction.isNeutral()) {
            throw new IllegalArgumentException("rejected command must apply neutral motors");
        }
    }

    public static ValidationResult accepted(ActionFrame command) {
        Objects.requireNonNull(command, "command");
        return new ValidationResult(
                true,
                command.rawAction(),
                command.rawAction().clamped(),
                ControllerFault.ACTIVE,
                Optional.of(command));
    }

    /**
     * Accepts a body-program selection.
     *
     * <p>A program frame requests <strong>no</strong> primitive motors, so both motor fields are
     * the neutral vector. That is not a formality: it is what makes the fall-through arm of
     * {@code FlyEntity.tick} safe. If the readout lane declines the frame for a reason that leaves
     * the body to another lane, the accepted-command arm applies {@code appliedAction} — and it
     * must be neutral, because there is no motor request on this frame to apply.
     */
    public static ValidationResult accepted(ProgramActionFrame command) {
        Objects.requireNonNull(command, "command");
        return new ValidationResult(
                true,
                RawPrimitiveAction.ZERO,
                PrimitiveAction.NEUTRAL,
                ControllerFault.ACTIVE,
                Optional.of(command));
    }

    /** Dispatches to the kind-specific factory above; the two carry different motor semantics. */
    public static ValidationResult accepted(CommandFrame command) {
        Objects.requireNonNull(command, "command");
        return command instanceof ActionFrame action
                ? accepted(action)
                : accepted((ProgramActionFrame) command);
    }

    public static ValidationResult rejected(ControllerFault fault) {
        return new ValidationResult(
                false,
                RawPrimitiveAction.ZERO,
                PrimitiveAction.NEUTRAL,
                fault,
                Optional.empty());
    }

    public static ValidationResult rejected(ActionFrame command, ControllerFault fault) {
        Objects.requireNonNull(command, "command");
        return new ValidationResult(
                false,
                command.rawAction(),
                PrimitiveAction.NEUTRAL,
                fault,
                Optional.of(command));
    }

    /** A rejected program frame requested no motors, so it retains the zero raw vector. */
    public static ValidationResult rejected(ProgramActionFrame command, ControllerFault fault) {
        Objects.requireNonNull(command, "command");
        return new ValidationResult(
                false,
                RawPrimitiveAction.ZERO,
                PrimitiveAction.NEUTRAL,
                fault,
                Optional.of(command));
    }

    /** Dispatches to the kind-specific factory above. */
    public static ValidationResult rejected(CommandFrame command, ControllerFault fault) {
        Objects.requireNonNull(command, "command");
        return command instanceof ActionFrame action
                ? rejected(action, fault)
                : rejected((ProgramActionFrame) command, fault);
    }
}
