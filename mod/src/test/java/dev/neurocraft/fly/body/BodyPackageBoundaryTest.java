package dev.neurocraft.fly.body;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;

/**
 * ADR 0020 regression pin: the body-program package cannot reach the bridge, and holds no source of
 * non-determinism.
 *
 * <p>This reads the package's own source text rather than its bytecode, because the property being
 * pinned is an architectural boundary a future edit could cross by accident. Two things are
 * asserted: every project import is on a declared allowlist, and a set of names that would let a
 * program see or touch the neural lane never appears anywhere in these files.
 */
final class BodyPackageBoundaryTest {
    private static final Path BODY_PACKAGE = Path.of("src/main/java/dev/neurocraft/fly/body");

    /** This file names the method it is pinning, so it must exclude itself from the scan. */
    private static final Path THIS_TEST_SOURCE =
            Path.of("src/test/java/dev/neurocraft/fly/body/BodyPackageBoundaryTest.java");

    /**
     * Project types a body program may depend on.
     *
     * <ul>
     *   <li>{@code PhysicalSensorSnapshot} — the authoritative sensor sample it reads.
     *   <li>{@code PrimitiveAction} / {@code RawPrimitiveAction} — the bounded motor vector it
     *       emits, so a program is clamped by exactly the rails the neural lane is.
     *   <li>{@code BodyFrameMath} — the single ADR 0015 statement of body-frame handedness. A
     *       program that restated it would be a second place for that sign to be wrong.
     *   <li>{@code PrimitiveActuator} — its declared thrust delta, used for bounded velocity feedback.
     *   <li>{@code FlyEntity} — declared body constants and the context sampler's physical
     *       bounding box, entity identity, and vanilla collision context. The
     *       forbidden-token list below blocks every accessor that would let a program read or
     *       touch the controller through the entity handle.
     * </ul>
     */
    private static final Set<String> ALLOWED_PROJECT_IMPORTS = Set.of(
            "dev.neurocraft.fly.control.PhysicalSensorSnapshot",
            "dev.neurocraft.fly.control.PrimitiveAction",
            "dev.neurocraft.fly.control.RawPrimitiveAction",
            "dev.neurocraft.fly.entity.BodyFrameMath",
            "dev.neurocraft.fly.entity.PrimitiveActuator",
            "dev.neurocraft.fly.entity.FlyEntity");

    /**
     * The exact JDK types a body program may import. Enumerated rather than prefixed so a future
     * edit cannot quietly pull in {@code java.util.concurrent}, {@code java.time},
     * {@code java.io}, or anything else that would make a program's motor sequence depend on
     * something other than its arguments.
     */
    private static final Set<String> ALLOWED_JDK_IMPORTS = Set.of(
            "java.util.ArrayDeque",
            "java.util.EnumMap",
            "java.util.List",
            "java.util.Locale",
            "java.util.Map",
            "java.util.Objects",
            "java.util.Optional",
            "java.util.OptionalLong",
            "java.util.Set");

    /**
     * Non-JDK, non-project package prefixes. Vanilla is allowed because lanes C, G, and H sample
     * the world through it; logging is allowed because the transition log is an honesty surface.
     */
    private static final List<String> ALLOWED_EXTERNAL_PREFIXES =
            List.of("net.minecraft.", "org.slf4j.");

    /**
     * Names that must not appear anywhere in the package.
     *
     * <p>The first five would give a program a way to see or influence the neural lane; the last
     * three are the sources of non-determinism that would make a program's motor sequence
     * unreproducible.
     */
    private static final List<String> FORBIDDEN_TOKENS = List.of(
            "NeuralBridge",
            "ControlProtocol",
            "ActionFrame",
            "CommandValidator",
            "FlyControllerRuntime",
            // FlyEntity is on the allowlist for its declared constants. These are the accessors
            // that would turn that allowance into a path to the neural lane: the last admitted
            // command, the last actuation receipt, another fly's program runtime, the
            // server-only absolute reset, and the controller's fault state.
            "lastAcceptedCommand",
            "lastActuation",
            "bodyPrograms",
            "resetForFixture",
            "controllerState",
            "java.util.Random",
            "Math.random",
            "System.currentTimeMillis",
            "System.nanoTime");

    @Test
    void bodyProgramsCannotReachTheBridge() throws IOException {
        List<Path> sources = sources();
        assertTrue(sources.size() >= 10, "body package source set collapsed: " + sources);
        for (Path source : sources) {
            String text = Files.readString(source, StandardCharsets.UTF_8);
            for (String forbidden : FORBIDDEN_TOKENS) {
                assertFalse(
                        text.contains(forbidden),
                        source + " references " + forbidden + ", which the body package may not");
            }
            for (String imported : projectImports(text)) {
                assertTrue(
                        imported.startsWith("dev.neurocraft.fly.body.")
                                || ALLOWED_PROJECT_IMPORTS.contains(imported),
                        source + " imports " + imported + ", which is not on the body allowlist");
            }
            for (String imported : externalImports(text)) {
                assertTrue(
                        ALLOWED_JDK_IMPORTS.contains(imported)
                                || ALLOWED_EXTERNAL_PREFIXES.stream()
                                        .anyMatch(imported::startsWith),
                        source + " imports " + imported + ", which is not an allowed dependency");
            }
        }
    }

    /**
     * {@code engageScripted} is the one method that can hand the motor vector to a scripted
     * program, and the PAUSE gate is enforced by its call sites rather than by visibility. This
     * pins that set: the accepted PAUSE transaction and the runtime local-selection path,
     * which requires a retained grant from that transaction and rechecks it at the next tick.
     */
    @Test
    void scriptedSelectionsRequireTheStagedPauseOrItsRetainedRuntimeGrant() throws IOException {
        Set<String> permitted = Set.of(
                // Initial PAUSE and subsequent selections under its retained authority grant.
                "mod/src/main/java/dev/neurocraft/fly/StagedProgramAuthority.java",
                "mod/src/main/java/dev/neurocraft/fly/FlyControllerRuntime.java",
                // Per-program and preemption pins drive the runtime directly, exactly as the
                // validator pins drive CommandValidator directly; the default GameTest lane has
                // no bridge, so an acknowledgement cannot be injected through the real path.
                "mod/src/gametest/java/dev/neurocraft/fly/gametest/FlyEntityGameTests.java",
                // The second default-lane entrypoint, holding the per-program pins (ADR 0020
                // lane B onward). Same reason as its sibling above.
                "mod/src/gametest/java/dev/neurocraft/fly/gametest/BodyProgramGameTests.java",
                // Lane D's entrypoint. One pin there drives a whole ESCAPE_JUMP through the
                // runtime so the exit into CRUISE is a real transition rather than a name in a
                // record; that needs the same direct engagement, for the same reason.
                "mod/src/gametest/java/dev/neurocraft/fly/gametest/EscapeJumpProgramGameTests.java",
                // Lane E2's entrypoint (the CRUISE floor band, R-31). Its pins drive CRUISE
                // through the runtime exactly as BodyProgramGameTests does, and one drives three
                // whole ESCAPE_JUMPs through it so each exit into CRUISE is a real transition.
                "mod/src/gametest/java/dev/neurocraft/fly/gametest/CruiseFloorBandGameTests.java",
                // Native tracking input is tested against actual CRUISE/ESCAPE motion in
                // a bridge-free client world. Only this named test fixture bypasses PAUSE;
                // the production DiagnosticClientActions helper never selects a program.
                "mod/src/gametest/java/dev/neurocraft/fly/control/DiagnosticClientActionsGameTest.java",
                // Pure trial-seed regression exercises authority changes without a bridge.
                "mod/src/test/java/dev/neurocraft/fly/body/BodyTrialSeedTest.java");
        List<String> callers = callersOf("engageScripted(");
        assertEquals(
                permitted,
                Set.copyOf(callers),
                "engageScripted call sites changed outside the body package: " + callers);
    }

    /**
     * {@code advanceReadout} is the one method that can hand the motor vector to a readout-selected
     * program, and like {@code engageScripted} its gate is its call sites rather than its
     * visibility. This pins that set: exactly one production caller, {@code FlyEntity}, on the
     * authoritative entity tick, and only when the validator admitted a {@code program_action}.
     *
     * <p>A second production caller would be a second place the readout lane could be entered, and
     * the whole argument that "the connectome's readouts selected this program" rests on there
     * being exactly one.
     */
    @Test
    void onlyTheEntityTickCanEngageReadoutAuthority() throws IOException {
        Set<String> permitted = Set.of(
                // The authoritative entity tick: the whole of the production path.
                "mod/src/main/java/dev/neurocraft/fly/entity/FlyEntity.java",
                // The readout-lane pins drive the runtime directly, exactly as the operator-lane
                // pins drive it directly. Two of them drive the real tick path instead, through a
                // recording bridge, because a claim about the wiring between the validator and the
                // body cannot be made from either end alone.
                "mod/src/gametest/java/dev/neurocraft/fly/gametest/ReadoutProgramGameTests.java",
                // The selection-bookkeeping pins drive the runtime as the pure state machine it
                // is, with no world at all: a refused selection must be logged once and held, and
                // that is six calls in a loop rather than six ticks of a server.
                "mod/src/test/java/dev/neurocraft/fly/body/BodyProgramRuntimeReadoutTest.java",
                // Pure trial-seed regression compares actual executions at different world ticks.
                "mod/src/test/java/dev/neurocraft/fly/body/BodyTrialSeedTest.java",
                // The client HUD honesty pin. One case there has to draw the published channel of
                // a runtime that really executed a readout-selected program under a turn bias,
                // because the claim is that the number on screen is the one the body ran under and
                // not the one the mapping asked for. Building the channel by hand would assert
                // that against the same literal it came from.
                "mod/src/gametest/java/dev/neurocraft/fly/gametest/HudHonestyClientGameTest.java");
        assertEquals(
                permitted,
                Set.copyOf(callersOf("advanceReadout(")),
                "advanceReadout call sites changed outside the body package");
    }

    private static List<String> callersOf(String token) throws IOException {
        List<String> callers = new ArrayList<>();
        for (Path root : List.of(
                Path.of("src/main/java"),
                Path.of("src/client/java"),
                Path.of("src/gametest/java"),
                Path.of("src/test/java"))) {
            try (Stream<Path> walk = Files.walk(root)) {
                for (Path source : walk.filter(path -> path.toString().endsWith(".java")).toList()) {
                    if (source.startsWith(BODY_PACKAGE) || source.equals(THIS_TEST_SOURCE)) {
                        continue;
                    }
                    if (Files.readString(source, StandardCharsets.UTF_8).contains(token)) {
                        callers.add("mod/" + source.toString().replace('\\', '/'));
                    }
                }
            }
        }
        return callers;
    }

    private static List<Path> sources() throws IOException {
        try (Stream<Path> walk = Files.walk(BODY_PACKAGE)) {
            return walk.filter(path -> path.getFileName().toString().endsWith(".java"))
                    .sorted()
                    .toList();
        }
    }

    private static List<String> projectImports(String text) {
        return importedTypes(text).stream()
                .filter(name -> name.startsWith("dev.neurocraft."))
                .toList();
    }

    private static List<String> externalImports(String text) {
        return importedTypes(text).stream()
                .filter(name -> !name.startsWith("dev.neurocraft."))
                .toList();
    }

    private static List<String> importedTypes(String text) {
        return text.lines()
                .map(String::strip)
                .filter(line -> line.startsWith("import "))
                .map(line -> line.substring("import ".length()).replace("static ", ""))
                .map(line -> line.endsWith(";") ? line.substring(0, line.length() - 1) : line)
                .map(String::strip)
                .toList();
    }
}
