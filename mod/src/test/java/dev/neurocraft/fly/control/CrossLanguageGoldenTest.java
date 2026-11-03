package dev.neurocraft.fly.control;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramExecution;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.body.ProgramRefusalReason;
import dev.neurocraft.fly.body.ReadoutValues;
import dev.neurocraft.fly.entity.ActuationReceipt;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Structural compatibility checks against the Python package's shared canonical golden frames. */
final class CrossLanguageGoldenTest {
    private static final String TOKEN = "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA";
    /** Unpadded base64url of the 32 bytes 0x00..0x1f, pinned by the Python golden. */
    private static final String CLIENT_NONCE = "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8";
    private static final ControllerIdentity IDENTITY = new ControllerIdentity(
            "demo-001",
            1,
            "fly-01",
            "1".repeat(64),
            "2".repeat(64),
            "3".repeat(64),
            "4".repeat(64),
            "5".repeat(64));

    @Test
    void parsesPythonGoldenInboundFrames() throws Exception {
        HelloAck ack = assertInstanceOf(
                HelloAck.class, ControlProtocol.decodeInbound(golden("hello_ack")));
        ActionFrame action = assertInstanceOf(
                ActionFrame.class, ControlProtocol.decodeInbound(golden("action")));
        FaultFrame fault = assertInstanceOf(
                FaultFrame.class, ControlProtocol.decodeInbound(golden("fault")));
        ControlAckFrame controlAck = assertInstanceOf(
                ControlAckFrame.class, ControlProtocol.decodeInbound(golden("control_ack")));

        assertEquals(IDENTITY, ack.identity());
        assertEquals(IDENTITY, action.identity());
        assertEquals(0, action.commandSequence());
        assertEquals(100, action.observationTick());
        assertEquals(102, action.targetTick());
        assertEquals(0.25, action.rawAction().forwardThrust());
        assertEquals("STALE", fault.code());
        assertEquals(0L, fault.relatedObservationSequence());
        assertEquals(ControlOperation.PAUSE, controlAck.operation());
        assertEquals(ControlState.PAUSED, controlAck.state());
        assertEquals(100, controlAck.effectiveTick());
    }

    @Test
    void JavaHelloIsStructurallyIdenticalToPythonGolden() throws Exception {
        assertStructuralEquals(
                golden("hello"), ControlProtocol.encodeHello(IDENTITY, TOKEN, CLIENT_NONCE));
    }

    @Test
    void JavaVerifiesThePythonGoldenHelloAckProof() throws Exception {
        HelloAck ack = assertInstanceOf(
                HelloAck.class, ControlProtocol.decodeInbound(golden("hello_ack")));

        assertTrue(ControlProtocol.verifyHelloAck(ack, TOKEN, IDENTITY, CLIENT_NONCE));
        assertFalse(ControlProtocol.verifyHelloAck(
                ack, TOKEN, IDENTITY, ControlProtocol.generateClientNonce()));
    }

    @Test
    void JavaObservationIsStructurallyIdenticalToPythonGolden() throws Exception {
        PhysicalSensorSnapshot snapshot = new PhysicalSensorSnapshot(
                100,
                "fly-01",
                1.0,
                64.0,
                -2.0,
                0.005,
                0.0,
                -0.01,
                Math.toDegrees(0.25),
                Math.nextUp(Math.toDegrees(-0.1)),
                false,
                false,
                false,
                false,
                false,
                12,
                10,
                8.0,
                8.0,
                6.0,
                5.0,
                8.0,
                2.5,
                3.0);

        assertStructuralEquals(
                golden("observation"),
                ControlProtocol.encodeObservation(IDENTITY, snapshot, 0, 123_456_789, 0));
    }

    @Test
    void JavaTypedObservationIsStructurallyIdenticalToPythonGolden() throws Exception {
        PhysicalSensorSnapshot snapshot = new PhysicalSensorSnapshot(
                100,
                "fly-01",
                1.0,
                64.0,
                -2.0,
                0.005,
                0.0,
                -0.01,
                Math.toDegrees(0.25),
                Math.nextUp(Math.toDegrees(-0.1)),
                false,
                false,
                false,
                false,
                false,
                12,
                10,
                8.0,
                8.0,
                6.0,
                5.0,
                8.0,
                2.5,
                3.0, java.util.List.of(0, 1, 2, 0, 1, 2));

        assertStructuralEquals(
                golden("observation_v2"),
                ControlProtocol.encodeObservation(new ControllerIdentity(
                        IDENTITY.runId(), IDENTITY.sessionEpoch(), IDENTITY.entityId(),
                        IDENTITY.graphHash(), IDENTITY.adapterHash(), IDENTITY.dynamicsHash(),
                        IDENTITY.decoderHash(), ControlProtocol.OBSERVATION_SCHEMA_V2_SHA256),
                        snapshot, 0, 123_456_789, 0));
    }

    @Test
    void JavaActuationReceiptIsStructurallyIdenticalToPythonGolden() throws Exception {
        ActionFrame action = assertInstanceOf(
                ActionFrame.class, ControlProtocol.decodeInbound(golden("action")));
        ActuationReceipt receipt = new ActuationReceipt(
                action.rawAction(),
                action.rawAction().clamped(),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0);

        assertStructuralEquals(
                golden("actuation_receipt"),
                ControlProtocol.encodeActuationReceipt(
                        IDENTITY,
                        action,
                        goldenSnapshot(102),
                        receipt,
                        ValidationResult.accepted(action)));
    }

    @Test
    void JavaDelayedRejectedReceiptIsStructurallyIdenticalToPythonGolden() throws Exception {
        ActionFrame action = assertInstanceOf(
                ActionFrame.class, ControlProtocol.decodeInbound(golden("action")));
        ActuationReceipt receipt = new ActuationReceipt(
                action.rawAction(),
                PrimitiveAction.NEUTRAL,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0);

        assertStructuralEquals(
                golden("actuation_receipt_rejected"),
                ControlProtocol.encodeActuationReceipt(
                        IDENTITY,
                        action,
                        goldenSnapshot(105),
                        receipt,
                        ValidationResult.rejected(action, ControllerFault.STALE_COMMAND)));
    }

    @Test
    void JavaControlCommandIsStructurallyIdenticalToPythonGolden() throws Exception {
        ControlCommandFrame command =
                new ControlCommandFrame(IDENTITY, 0, 100, ControlOperation.PAUSE, null, null);

        assertStructuralEquals(
                golden("control_command"), ControlProtocol.encodeControlCommand(command));
    }

    @Test
    void parsesPythonInterventionControlAckGolden() throws Exception {
        ControlAckFrame acknowledgement = assertInstanceOf(
                ControlAckFrame.class,
                ControlProtocol.decodeInbound(golden("intervention_control_ack")));

        assertEquals(IDENTITY, acknowledgement.identity());
        assertEquals(1, acknowledgement.controlSequence());
        assertEquals(105, acknowledgement.issuedTick());
        assertEquals(ControlOperation.LESION_APPLY, acknowledgement.operation());
        assertEquals("left-sensory-lesion-v1", acknowledgement.contentId());
        assertEquals("6".repeat(64), acknowledgement.contentHash());
        assertEquals(ControlState.PAUSED, acknowledgement.state());
    }

    @Test
    void parsesPythonResetAckAtTheGovernedFutureTick() throws Exception {
        ControlAckFrame acknowledgement = assertInstanceOf(
                ControlAckFrame.class,
                ControlProtocol.decodeInbound(golden("reset_control_ack")));

        assertEquals(ControlOperation.RESET, acknowledgement.operation());
        assertEquals(ControlState.RECORDING, acknowledgement.state());
        assertEquals(acknowledgement.issuedTick() + 2, acknowledgement.effectiveTick());
    }

    @Test
    void JavaInterventionCommandIsStructurallyIdenticalToPythonGolden() throws Exception {
        ControlCommandFrame command = new ControlCommandFrame(
                IDENTITY,
                1,
                105,
                ControlOperation.LESION_APPLY,
                "left-sensory-lesion-v1",
                "6".repeat(64));

        assertStructuralEquals(
                golden("intervention_control_command"),
                ControlProtocol.encodeControlCommand(command));
    }

    @Test
    void parsesThePythonGoldenProgramAction() throws Exception {
        ProgramActionFrame frame = assertInstanceOf(
                ProgramActionFrame.class, ControlProtocol.decodeInbound(golden("program_action")));

        assertEquals(IDENTITY, frame.identity());
        assertEquals(0, frame.commandSequence());
        assertEquals(100, frame.observationTick());
        assertEquals(102, frame.targetTick());
        assertEquals(102, frame.validThroughTick());
        assertEquals(0, frame.neuralStepStart());
        assertEquals(5, frame.neuralStepEnd());
        assertEquals(BodyProgramId.CRUISE, frame.program());
        assertEquals(new Modulation(0.7, 1.0, 1.0, 0.0), frame.modulation());
        assertEquals(new ProgramParameters(-0.25), frame.parameters());
        assertEquals(new ReadoutValues(0.0, -0.25, 0.61), frame.readout());
        assertEquals("7".repeat(64), frame.policyHash());
    }

    @Test
    void JavaAppliedProgramReceiptIsStructurallyIdenticalToPythonGolden() throws Exception {
        ProgramActionFrame frame = assertInstanceOf(
                ProgramActionFrame.class, ControlProtocol.decodeInbound(golden("program_action")));
        ProgramExecution execution = new ProgramExecution(
                frame.program(),
                frame.modulation(),
                frame.parameters(),
                PrimitiveAction.clamped(0.35, 0.0, 0.1, -0.05, 0.0),
                Optional.empty());

        assertStructuralEquals(
                golden("program_receipt"),
                ControlProtocol.encodeProgramReceipt(
                        IDENTITY,
                        frame,
                        goldenSnapshot(102),
                        Optional.of(execution),
                        ValidationResult.accepted(frame)));
    }

    /**
     * An {@code unregistered} refusal ran the declared HOVER fallback, under the modulation and
     * parameters the frame asked for, because the runtime sets them before it consults the registry.
     */
    @Test
    void JavaUnregisteredRefusalReceiptIsStructurallyIdenticalToPythonGolden() throws Exception {
        ProgramActionFrame frame = programFrame(
                BodyProgramId.FEED,
                new Modulation(0.7, 1.0, 1.0, 0.0),
                new ProgramParameters(-0.25));
        ProgramExecution execution = new ProgramExecution(
                BodyProgramId.HOVER,
                frame.modulation(),
                frame.parameters(),
                PrimitiveAction.clamped(0.0, 0.0, 0.08, 0.0, 0.0),
                Optional.of(ProgramRefusalReason.UNREGISTERED));

        assertStructuralEquals(
                golden("program_receipt_refused"),
                ControlProtocol.encodeProgramReceipt(
                        IDENTITY,
                        frame,
                        goldenSnapshot(102),
                        Optional.of(execution),
                        ValidationResult.accepted(frame)));
    }

    /**
     * An {@code operator} refusal is the one refusal that did <strong>not</strong> run HOVER: an
     * operator hold owned the tick, so the receipt reports the operator's own program, under the
     * operator's own modulation, with the motors that program actually emitted. Pinning HOVER here
     * would make the receipt assert a program ran that did not.
     */
    @Test
    void JavaOperatorRefusalReceiptReportsTheProgramThatActuallyRan() throws Exception {
        ProgramActionFrame frame = programFrame(
                BodyProgramId.ESCAPE_JUMP,
                new Modulation(1.0, 1.0, 1.0, 0.8),
                ProgramParameters.NEUTRAL);
        ProgramExecution execution = new ProgramExecution(
                BodyProgramId.CRUISE,
                new Modulation(0.5, 0.5, 1.0, 0.0),
                ProgramParameters.NEUTRAL,
                PrimitiveAction.clamped(0.275, 0.0, 0.0, 0.0625, 0.0),
                Optional.of(ProgramRefusalReason.OPERATOR));

        assertStructuralEquals(
                golden("program_receipt_refused_operator"),
                ControlProtocol.encodeProgramReceipt(
                        IDENTITY,
                        frame,
                        goldenSnapshot(102),
                        Optional.of(execution),
                        ValidationResult.accepted(frame)));
    }

    @Test
    void JavaRejectedProgramReceiptIsStructurallyIdenticalToPythonGolden() throws Exception {
        ProgramActionFrame frame = assertInstanceOf(
                ProgramActionFrame.class, ControlProtocol.decodeInbound(golden("program_action")));

        assertStructuralEquals(
                golden("program_receipt_rejected"),
                ControlProtocol.encodeProgramReceipt(
                        IDENTITY,
                        frame,
                        goldenSnapshot(105),
                        Optional.empty(),
                        ValidationResult.rejected(frame, ControllerFault.STALE_COMMAND)));
    }

    /** The golden envelope, with only the program-specific fields varied. */
    private static ProgramActionFrame programFrame(
            BodyProgramId program, Modulation modulation, ProgramParameters parameters) {
        return new ProgramActionFrame(
                IDENTITY,
                0,
                0,
                100,
                102,
                102,
                0,
                5,
                "6".repeat(64),
                program,
                modulation,
                parameters,
                new ReadoutValues(0.0, -0.25, 0.61),
                "7".repeat(64));
    }

    private static String golden(String name) throws Exception {
        Path path = Path.of("..", "packages", "protocol", "golden", name + ".json")
                .toAbsolutePath()
                .normalize();
        return Files.readString(path);
    }

    private static PhysicalSensorSnapshot goldenSnapshot(long gameTick) {
        return new PhysicalSensorSnapshot(
                gameTick,
                "fly-01",
                1.0,
                64.0,
                -2.0,
                0.005,
                0.0,
                -0.01,
                Math.toDegrees(0.25),
                Math.nextUp(Math.toDegrees(-0.1)),
                false,
                false,
                false,
                false,
                false,
                12,
                10,
                8.0,
                8.0,
                6.0,
                5.0,
                8.0,
                2.5,
                3.0);
    }

    private static void assertStructuralEquals(String expectedJson, String actualJson)
            throws Exception {
        Map<String, Object> expected =
                StrictJson.parseObject(expectedJson, ControlProtocol.MAX_MESSAGE_BYTES);
        Map<String, Object> actual =
                StrictJson.parseObject(actualJson, ControlProtocol.MAX_MESSAGE_BYTES);
        assertEquals(expected, actual);
    }
}
