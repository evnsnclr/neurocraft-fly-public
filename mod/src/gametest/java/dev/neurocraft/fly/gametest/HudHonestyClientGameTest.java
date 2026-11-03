package dev.neurocraft.fly.gametest;

import dev.neurocraft.fly.body.BodyProgramChannel;
import dev.neurocraft.fly.body.BodyProgramContext;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.BodyProgramRuntime;
import dev.neurocraft.fly.body.BodyProgramState;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.body.ReadoutChannel;
import dev.neurocraft.fly.body.ReadoutValues;
import dev.neurocraft.fly.body.SelectionSource;
import dev.neurocraft.fly.body.programs.CruiseProgram;
import dev.neurocraft.fly.client.NeuroCraftHud;
import dev.neurocraft.fly.control.ControllerFault;
import dev.neurocraft.fly.control.BrainDisplayFrame;
import dev.neurocraft.fly.control.BrainDisplayPresentation;
import dev.neurocraft.fly.control.BrainDisplayState;
import dev.neurocraft.fly.control.ControllerIdentity;
import dev.neurocraft.fly.control.ComparisonLanePresentation;
import dev.neurocraft.fly.control.PhysicalSensorSnapshot;
import dev.neurocraft.fly.control.PrimitiveAction;
import java.util.List;
import java.util.LinkedHashMap;
import java.util.Optional;
import java.util.stream.Collectors;
import net.fabricmc.fabric.api.client.gametest.v1.FabricClientGameTest;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.minecraft.client.gui.Font;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The HUD's honesty strings, pinned as strings rather than as pixels -- and, once, as pixels.
 *
 * <p>A frame grab travels without its caption (D-074 / ADR 0016), so the disclaimer has to be
 * <em>inside</em> the drawn line. Under READOUT that means two things a screenshot must carry on its
 * face: that the readout-to-program mapping is hand-chosen engineering (ADR 0019), and that the fly
 * is not deciding anything. This test fails if either clause is dropped or reworded.
 *
 * <p>It also fails if a line ever blends the lanes: a READOUT line must never carry the operator
 * lane's "CONNECTOME NOT DRIVING", an OPERATOR line must never claim a readout selected it, and the
 * declared HOVER fallback must never be drawn with an arrow from a readout that did not select it.
 *
 * <p>And it fails if any drawn line is wider than the panel it is drawn in, measured with the
 * client's real font. Words that are on the HUD but past the panel's edge are words the viewer may
 * never see: that was the case live (lane L, 2026-09-04), and it is the one failure a string pin
 * alone cannot catch.
 *
 * <p><strong>Every value is pushed through the synchronized channel words first.</strong> The HUD
 * only ever sees decoded values, so pinning the strings against the runtime's exact ones would pin
 * a rendering the client can never produce.
 */
public final class HudHonestyClientGameTest implements FabricClientGameTest {
    private static final Logger LOGGER = LoggerFactory.getLogger("neurocraft-fly-hud-test");
    private static final ReadoutValues DRIVING =
            published(new ReadoutValues(0.4213, -0.0120, 0.6149));

    /** The honesty clauses the READOUT lane must draw, whatever else it draws. */
    private static final String MAPPING_DISCLAIMER =
            "READOUT MAPPING: HAND-CHOSEN (ADR 0019) · NOT THE FLY DECIDING";

    /**
     * Every frame the fit check lays out: the readout lane on each program it can show (two
     * selections and the fallback), then the operator and default lanes beside it. All carry the
     * non-neutral modulation the {@link #channel} helper packs, so the MODULATION line is drawn
     * too.
     */
    private static final List<BodyProgramChannel> FIT_FRAMES = List.of(
            channel(SelectionSource.READOUT, BodyProgramId.ESCAPE_JUMP),
            channel(SelectionSource.READOUT, BodyProgramId.CRUISE),
            channel(SelectionSource.READOUT, BodyProgramId.HOVER),
            channel(SelectionSource.OPERATOR, BodyProgramId.WALK),
            channel(SelectionSource.DEFAULT, BodyProgramId.HOVER));

    @Override
    public void runTest(ClientGameTestContext context) {
        assertHudNamesTheReadoutSourceMappingAndDrivingReadout();
        assertTheFallbackIsNotDrawnAsASelection();
        assertTheOperatorAndDefaultLinesAreUnchanged();
        assertNoLineBlendsTheLanes();
        assertTheModulationLineCarriesTheSignedBias();
        assertTheBiasTheBodyRanUnderIsTheBiasTheHudDraws();
        assertEveryDrawnLineFitsInsideThePanel(context);
        assertShowcaseFitsAndKeepsItsFacts(context);
        assertComparisonShowcaseFitsAndNamesOnlyAuthenticatedLanes(context);
        assertLine("WORLD RATE: 10.0 TPS · HALF-SPEED DIAGNOSTIC", NeuroCraftHud.playbackRateLine(10.0F));
        assertLine("WORLD RATE: 20.0 TPS · NORMAL TARGET", NeuroCraftHud.playbackRateLine(20.0F));
    }

    private static void assertComparisonShowcaseFitsAndNamesOnlyAuthenticatedLanes(
            ClientGameTestContext context) {
        long epoch = 1_783_000_000_000_000_007L;
        var identity = new ControllerIdentity("hud-test", epoch, "fly", "a".repeat(64),
                "b".repeat(64), "c".repeat(64), "d".repeat(64), "e".repeat(64));
        var means = new LinkedHashMap<String, BrainDisplayFrame.Population>();
        for (String name : BrainDisplayFrame.POPULATION_NAMES) {
            means.put(name, new BrainDisplayFrame.Population(-1.0,
                    name.equals("lc4_left") ? 71 : name.equals("lc4_right") ? 55 : 1));
        }
        var sample = new BrainDisplayFrame(identity, 123456, 1234567, 500, 505,
                "f".repeat(64), "neurocraft.eem.encoder.loom.v2", "1".repeat(64), means);
        context.runOnClient(client -> {
            for (int width : new int[] {320, 347, client.getWindow().getGuiScaledWidth()}) {
                for (String name : List.of("original", "shuffled_weight", "no_input")) {
                    for (var status : List.of(ComparisonLanePresentation.Status.ACTIVE,
                            ComparisonLanePresentation.Status.SWITCHING, ComparisonLanePresentation.Status.FAILED)) {
                        boolean active = status == ComparisonLanePresentation.Status.ACTIVE;
                        boolean original = name.equals("original");
                        var comparison = active ? new ComparisonLanePresentation(status, name, epoch,
                                Optional.of(identity), name, original ? "faithful" : name,
                                original ? null : "8".repeat(64), original ? null : "18446744073709551615")
                                : ComparisonLanePresentation.waiting(status, name, epoch);
                        var display = new BrainDisplayPresentation(active ? BrainDisplayState.Status.MATCHED
                                : BrainDisplayState.Status.STALE, active ? 500 : 1500,
                                active ? Optional.of(sample) : Optional.empty());
                        for (var frame : FIT_FRAMES) {
                            var lines = NeuroCraftHud.wrapLines(client.font, NeuroCraftHud.showcaseLines(
                                    active ? ControllerFault.ACTIVE : ControllerFault.STARTING,
                                    frame, DRIVING, PrimitiveAction.NEUTRAL, display, 20.0F, comparison), width);
                            String text = lines.stream().map(NeuroCraftHud.Line::text).collect(Collectors.joining(" "));
                            for (String required : new String[] {name, Long.toString(epoch), "95.66x", "403.70x",
                                    "OFFLINE ESCAPE-ARC-01", "~300x", "35.1x", "4 neural steps",
                                    "NOT LIVE REACTION TIMING", "HAND-CHOSEN", "SCRIPTED",
                                    "128-NEURON FIXTURE", "SYNTHETIC IMITATION"}) {
                                if (!text.contains(required)) {
                                    throw new AssertionError("comparison dropped fact: " + required);
                                }
                            }
                            if (text.contains("LANE ACTIVE:") != active) {
                                throw new AssertionError("requested lane was mislabeled active: " + text);
                            }
                            if (active && (!text.contains("GRAPH aaaaaaaa") || !text.contains("ENC loom.v2"))) {
                                throw new AssertionError("comparison lost source graph/encoder identity: " + text);
                            }
                            int right = NeuroCraftHud.panelRight(client.font, lines, width);
                            for (var line : lines) {
                                if (NeuroCraftHud.TEXT_LEFT + client.font.width(line.text()) >= right) {
                                    throw new AssertionError("comparison clipped: " + line.text());
                                }
                            }
                            if (NeuroCraftHud.panelBottom(lines.size()) > 240) {
                                throw new AssertionError("comparison exceeds 240px: " + lines.size() + " " + text);
                            }
                        }
                    }
                }
            }
        });
    }

    private static void assertShowcaseFitsAndKeepsItsFacts(ClientGameTestContext context) {
        var identity = new ControllerIdentity("hud-test", 1, "fly", "a".repeat(64),
                "b".repeat(64), "c".repeat(64), "d".repeat(64), "e".repeat(64));
        var means = new LinkedHashMap<String, BrainDisplayFrame.Population>();
        for (String name : BrainDisplayFrame.POPULATION_NAMES) {
            means.put(name, new BrainDisplayFrame.Population(-1.0,
                    name.equals("lc4_left") ? 71 : name.equals("lc4_right") ? 55 : 1));
        }
        var sample = new BrainDisplayFrame(identity, 123456, 1234567, 500, 505,
                "f".repeat(64), "neurocraft.eem.encoder.loom.v2", "1".repeat(64), means);
        var matched = new BrainDisplayPresentation(BrainDisplayState.Status.MATCHED,
                500, Optional.of(sample));
        var stale = new BrainDisplayPresentation(BrainDisplayState.Status.STALE,
                1500, Optional.empty());
        context.runOnClient(client -> {
            for (int width : new int[] {320, 347, client.getWindow().getGuiScaledWidth()}) {
                for (var frame : FIT_FRAMES) {
                    for (var display : List.of(matched, stale)) {
                        var lines = NeuroCraftHud.wrapLines(client.font, NeuroCraftHud.showcaseLines(
                                ControllerFault.ACTIVE, frame, DRIVING, PrimitiveAction.NEUTRAL,
                                display, 10.0F), width);
                        String text = lines.stream().map(NeuroCraftHud.Line::text)
                                .collect(Collectors.joining(" "));
                        for (String required : new String[] {"HALF-SPEED DIAGNOSTIC", "95.66x",
                                "403.70x", "OFFLINE ESCAPE-ARC-01", "~300x", "35.1x",
                                "4 neural steps", "NOT LIVE REACTION TIMING",
                                "128-NEURON FIXTURE", "SYNTHETIC IMITATION"}) {
                            if (!text.contains(required)) {
                                throw new AssertionError("showcase dropped required fact: " + required);
                            }
                        }
                        int right = NeuroCraftHud.panelRight(client.font, lines, width);
                        for (var line : lines) {
                            if (NeuroCraftHud.TEXT_LEFT + client.font.width(line.text()) >= right) {
                                throw new AssertionError("showcase clipped: " + line.text());
                            }
                        }
                        if (NeuroCraftHud.panelBottom(lines.size()) > 240) {
                            throw new AssertionError("showcase exceeds 240px height: " + lines.size());
                        }
                        if (display == stale && text.contains("-1.0e+00")) {
                            throw new AssertionError("stale activity was rendered as current");
                        }
                        if (frame.source() == SelectionSource.OPERATOR
                                && !text.contains("PAUSED · CONNECTOME NOT STEPPING")) {
                            throw new AssertionError("operator frame lost pause label");
                        }
                    }
                }
            }
        });
    }

    /**
     * The escape case names TTMn on the program line and the other two beneath it; the cruise case
     * names the wing term. Both carry the whole mapping disclaimer, on a line of its own.
     */
    private static void assertHudNamesTheReadoutSourceMappingAndDrivingReadout() {
        assertLine(
                "BODY PROGRAM: ESCAPE_JUMP · READOUT ← TTMn 0.42",
                NeuroCraftHud.bodyProgramLine(
                        channel(SelectionSource.READOUT, BodyProgramId.ESCAPE_JUMP), DRIVING));
        assertLines(
                List.of(MAPPING_DISCLAIMER, "DNa01 -0.01  WING +0.61"),
                NeuroCraftHud.readoutMappingLines(BodyProgramId.ESCAPE_JUMP, DRIVING));
        assertLine(
                "BODY PROGRAM: CRUISE · READOUT ← WING +0.61",
                NeuroCraftHud.bodyProgramLine(
                        channel(SelectionSource.READOUT, BodyProgramId.CRUISE), DRIVING));
        assertLines(
                List.of(MAPPING_DISCLAIMER, "TTMn 0.42  DNa01 -0.01"),
                NeuroCraftHud.readoutMappingLines(BodyProgramId.CRUISE, DRIVING));
    }

    /**
     * HOVER on the readout lane is the declared fallback the authority ran after refusing the
     * mapping's request. No readout selected it, so no readout may be named beside it: the line
     * says FALLBACK and carries no arrow, and all three scalars move to the values line.
     */
    private static void assertTheFallbackIsNotDrawnAsASelection() {
        String line = NeuroCraftHud.bodyProgramLine(
                channel(SelectionSource.READOUT, BodyProgramId.HOVER), DRIVING);
        assertLine("BODY PROGRAM: HOVER · READOUT · FALLBACK", line);
        if (line.contains("←")) {
            throw new AssertionError("the declared fallback was drawn as a selection: " + line);
        }
        List<NeuroCraftHud.Line> showcase = NeuroCraftHud.showcaseLines(
                ControllerFault.ACTIVE, channel(SelectionSource.READOUT, BodyProgramId.HOVER),
                DRIVING, PrimitiveAction.NEUTRAL,
                new BrainDisplayPresentation(BrainDisplayState.Status.UNAVAILABLE, 0,
                        Optional.empty()), 20.0F);
        if (showcase.stream().noneMatch(item -> item.text().equals(
                "BODY: HOVER · READOUT · FALLBACK"))) {
            throw new AssertionError("showcase hid the declared fallback: " + showcase);
        }
        assertLines(
                List.of(MAPPING_DISCLAIMER, "TTMn 0.42  DNa01 -0.01  WING +0.61"),
                NeuroCraftHud.readoutMappingLines(BodyProgramId.HOVER, DRIVING));
    }

    private static void assertTheOperatorAndDefaultLinesAreUnchanged() {
        assertLine(
                "BODY PROGRAM: HOVER · DEFAULT",
                NeuroCraftHud.bodyProgramLine(
                        channel(SelectionSource.DEFAULT, BodyProgramId.HOVER),
                        ReadoutValues.ZERO));
        assertLine(
                "BODY PROGRAM: WALK · OPERATOR — SCRIPTED, CONNECTOME NOT DRIVING",
                NeuroCraftHud.bodyProgramLine(
                        channel(SelectionSource.OPERATOR, BodyProgramId.WALK), ReadoutValues.ZERO));
    }

    private static void assertNoLineBlendsTheLanes() {
        String readoutLine = NeuroCraftHud.bodyProgramLine(
                channel(SelectionSource.READOUT, BodyProgramId.CRUISE), DRIVING);
        if (readoutLine.contains("CONNECTOME NOT DRIVING") || readoutLine.contains("OPERATOR")) {
            throw new AssertionError(
                    "a READOUT line carried the operator lane's disclaimer: " + readoutLine);
        }
        String operatorLine = NeuroCraftHud.bodyProgramLine(
                channel(SelectionSource.OPERATOR, BodyProgramId.WALK), DRIVING);
        if (operatorLine.contains("READOUT") || operatorLine.contains("TTMn")) {
            throw new AssertionError(
                    "an OPERATOR line claimed a readout selected it: " + operatorLine);
        }
    }

    private static void assertTheModulationLineCarriesTheSignedBias() {
        BodyProgramChannel biased = channel(SelectionSource.READOUT, BodyProgramId.CRUISE);
        assertLine(
                "MODULATION: SPD 0.7 TRN 1.0 AMP 1.0 URG 0.0 BIAS -0.3",
                NeuroCraftHud.modulationLine(biased.modulation(), biased.parameters()));
        BodyProgramChannel neutral = BodyProgramChannel.decode(BodyProgramChannel.DEFAULT.packed());
        assertLine(
                "MODULATION: SPD 1.0 TRN 1.0 AMP 1.0 URG 0.0 BIAS +0.0",
                NeuroCraftHud.modulationLine(neutral.modulation(), neutral.parameters()));
    }

    /**
     * <strong>The BIAS field, end to end from the program that actually read it.</strong>
     *
     * <p>{@link #assertTheModulationLineCarriesTheSignedBias} pins channel-to-line from a
     * hand-built channel, which is the right shape for a rendering pin and says nothing about
     * where the number came from. This pins the other half: a turn bias is handed to the runtime
     * on a program frame, CRUISE is stepped with it, and the value the HUD draws comes out of
     * <em>that runtime's</em> published channel word rather than out of the same literal the
     * assertion was written from.
     *
     * <p>The bias is proven to have reached the program before the string is checked -- the same
     * frame run at zero bias must produce a different yaw torque -- so a regression that quietly
     * stopped threading the parameter into {@code step} would fail here rather than continue to
     * draw a number nothing used. That is the specific dishonesty this test exists to catch: a
     * HUD reporting what the mapping asked for as though it were what the body did.
     */
    private static void assertTheBiasTheBodyRanUnderIsTheBiasTheHudDraws() {
        ProgramParameters requested = new ProgramParameters(0.7);
        double biasedYaw = cruiseYawUnder(requested);
        double unbiasedYaw = cruiseYawUnder(ProgramParameters.NEUTRAL);
        if (biasedYaw == unbiasedYaw) {
            throw new AssertionError(
                    "the turn bias never reached CRUISE, so the HUD would be drawing a number the"
                            + " body did not use; yaw was " + biasedYaw + " either way");
        }

        BodyProgramRuntime runtime = new BodyProgramRuntime();
        runtime.advanceReadout(
                hudSnapshot(),
                BodyProgramId.CRUISE,
                Modulation.NEUTRAL,
                requested,
                ReadoutValues.ZERO,
                "7".repeat(64));
        BodyProgramChannel drawn = BodyProgramChannel.decode(runtime.channel().packed());
        assertLine(
                "MODULATION: SPD 1.0 TRN 1.0 AMP 1.0 URG 0.0 BIAS +0.7",
                NeuroCraftHud.modulationLine(drawn.modulation(), drawn.parameters()));
    }

    /**
     * <strong>Every drawn line fits inside the panel it is drawn in, measured with the real
     * font.</strong>
     *
     * <p>For each frame in {@link #FIT_FRAMES}, under every controller state the CONTROL: line can
     * name, the exact lines the HUD would draw are laid out and each is measured with the client's
     * {@link Font}; its right edge must fall strictly inside the panel's right edge, and the panel
     * itself must fit inside the GUI-scaled window this test runs in. The readout frames carry all
     * three readout values non-zero, and each must draw both honesty clauses and all three
     * population names somewhere in its lines -- "fits" would be a hollow property if fitting were
     * achieved by dropping words.
     *
     * <p>Pinned because it was false. {@code READOUT MAPPING: ... · TTMn 0.01 DNa01 +0.34 WING
     * +0.03} was drawn into a fixed 380-px panel; at the default GUI scale in a 1,708-px window it
     * overran the panel and the window edge cut it after {@code TTMn 0.01 DNa}, so two of the three
     * values never reached the screen in five minutes of recording (lane L, 2026-09-04). The window
     * check speaks only for the GUI scale it ran at; the failure message names that scale.
     */
    private static void assertEveryDrawnLineFitsInsideThePanel(ClientGameTestContext context) {
        context.runOnClient(client -> {
            Font font = client.font;
            for (int guiWidth : new int[] {320, 347, client.getWindow().getGuiScaledWidth()}) {
                int widestRight = 0;
                for (BodyProgramChannel frame : FIT_FRAMES) {
                    for (ControllerFault state : ControllerFault.values()) {
                        List<NeuroCraftHud.Line> original = NeuroCraftHud.panelLines(
                                state, frame, DRIVING, PrimitiveAction.NEUTRAL);
                        List<NeuroCraftHud.Line> lines =
                                NeuroCraftHud.wrapLines(font, original, guiWidth);
                        int right = NeuroCraftHud.panelRight(font, lines, guiWidth);
                        widestRight = Math.max(widestRight, right);
                        String before = original.stream().map(NeuroCraftHud.Line::text)
                                .collect(Collectors.joining(" ")).replaceAll("\\s+", " ");
                        String after = lines.stream().map(NeuroCraftHud.Line::text)
                                .collect(Collectors.joining(" ")).replaceAll("\\s+", " ");
                        if (!before.equals(after)) {
                            throw new AssertionError("HUD wrapping lost words: " + after);
                        }
                        for (NeuroCraftHud.Line line : lines) {
                            int textRight = NeuroCraftHud.TEXT_LEFT + font.width(line.text());
                            if (textRight >= right || right > guiWidth) {
                                throw new AssertionError("HUD does not fit width " + guiWidth
                                        + ": text=" + textRight + ", panel=" + right
                                        + ": " + line.text());
                            }
                        }
                        if (frame.source() == SelectionSource.READOUT) {
                            assertReadoutFrameCarriesEveryWord(lines);
                        }
                    }
                }
                LOGGER.info("HUD fit check passed at width {}: panel right {} px",
                        guiWidth, widestRight);
            }
        });
    }

    /** Fitting by omission is not fitting: both clauses and all three names must be drawn. */
    private static void assertReadoutFrameCarriesEveryWord(List<NeuroCraftHud.Line> lines) {
        String drawn =
                lines.stream().map(NeuroCraftHud.Line::text).collect(Collectors.joining(" ")).replaceAll("\\s+", " ");
        for (String required : new String[] {
            "HAND-CHOSEN (ADR 0019)", "NOT THE FLY DECIDING", "TTMn", "DNa01", "WING"
        }) {
            if (!drawn.contains(required)) {
                throw new AssertionError(
                        "a READOUT frame no longer draws \"" + required + "\":\n" + drawn);
            }
        }
    }

    /** CRUISE's yaw torque on its first tick under one bias, in the same context as above. */
    private static double cruiseYawUnder(ProgramParameters parameters) {
        return new CruiseProgram()
                .step(
                        BodyProgramState.entering(BodyProgramId.CRUISE, 23L),
                        BodyProgramContext.sensorsOnly(hudSnapshot()),
                        Modulation.NEUTRAL,
                        parameters)
                .motors()
                .yawTorque();
    }

    /** Open air: every ray at its clamp, so the bias is the only yaw source. */
    private static PhysicalSensorSnapshot hudSnapshot() {
        return new PhysicalSensorSnapshot(
                23L,
                "00000000-0000-0000-0000-000000000001",
                0.0, 64.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0,
                false, false, false, false, false,
                0, 15,
                6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0);
    }

    /** What the client actually receives: the channel word, decoded. */
    private static BodyProgramChannel channel(SelectionSource source, BodyProgramId program) {
        return BodyProgramChannel.decode(new BodyProgramChannel(
                        source,
                        program,
                        new Modulation(0.7, 1.0, 1.0, 0.0),
                        new ProgramParameters(-0.25))
                .packed());
    }

    private static ReadoutValues published(ReadoutValues exact) {
        return ReadoutChannel.decode(new ReadoutChannel(exact).packed()).readout();
    }

    private static void assertLine(String expected, String actual) {
        if (!expected.equals(actual)) {
            throw new AssertionError(
                    "HUD line changed\n  expected: " + expected + "\n  actual:   " + actual);
        }
    }

    private static void assertLines(List<String> expected, List<String> actual) {
        if (!expected.equals(actual)) {
            throw new AssertionError(
                    "HUD lines changed\n  expected: " + expected + "\n  actual:   " + actual);
        }
    }
}
