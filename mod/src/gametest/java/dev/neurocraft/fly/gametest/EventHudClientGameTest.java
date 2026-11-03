package dev.neurocraft.fly.gametest;

import dev.neurocraft.fly.body.BodyProgramChannel;
import dev.neurocraft.fly.body.BodyProgramId;
import dev.neurocraft.fly.body.Modulation;
import dev.neurocraft.fly.body.ProgramParameters;
import dev.neurocraft.fly.body.ReadoutValues;
import dev.neurocraft.fly.body.SelectionSource;
import dev.neurocraft.fly.client.NeuroCraftHud;
import dev.neurocraft.fly.client.EventReplayScreen;
import dev.neurocraft.fly.control.*;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Optional;
import net.minecraft.client.gui.Font;
import net.fabricmc.fabric.api.client.gametest.v1.FabricClientGameTest;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;

/** Actual-font layout and issued/applied honesty checks; fixture means are synthetic. */
public final class EventHudClientGameTest implements FabricClientGameTest {
    @Override public void runTest(ClientGameTestContext context) {
        final EventDisplayFrame event;
        try (var stream = getClass().getResourceAsStream("/neurocraft/event-display.json")) {
            if (stream == null) { throw new AssertionError("missing Python diagnostic fixture"); }
            event = EventDisplayFrame.parse(new String(stream.readAllBytes(), StandardCharsets.UTF_8));
        } catch (Exception exception) { throw new AssertionError(exception); }
        var populations = new LinkedHashMap<String, BrainDisplayFrame.Population>();
        for (String name : BrainDisplayFrame.POPULATION_NAMES) {
            populations.put(name, new BrainDisplayFrame.Population(-0.00001, 1));
        }
        var brain = new BrainDisplayFrame(event.identity(), 5, 5, 25, 30, "e".repeat(64),
                "neurocraft.events.encoder.v1", "3".repeat(64), populations);
        var brainPresentation = new BrainDisplayPresentation(BrainDisplayState.Status.MATCHED,
                500, Optional.of(brain));
        var eventPresentation = EventDisplayPresentation.parseOrInvalid(new EventDisplayPresentation(
                EventDisplayState.Status.MATCHED, 500, Optional.of(event)).encode());
        context.runOnClient(client -> {
            assertInteractionHud(client.font);
            var retainedChannel = new BodyProgramChannel(SelectionSource.READOUT, BodyProgramId.FEED,
                    Modulation.NEUTRAL, ProgramParameters.NEUTRAL);
            // Deliberately retain matched packets while only the authoritative state changes.
            for (var fault : ControllerFault.values()) {
                for (var comparison : List.of(ComparisonLanePresentation.unavailable(),
                        ComparisonLanePresentation.waiting(ComparisonLanePresentation.Status.SWITCHING,
                                "no_input", event.identity().sessionEpoch()),
                        ComparisonLanePresentation.waiting(ComparisonLanePresentation.Status.FAILED,
                                "no_input", event.identity().sessionEpoch()))) {
                    var lines = NeuroCraftHud.showcaseLines(fault, retainedChannel,
                            new ReadoutValues(0, 0, 0), PrimitiveAction.NEUTRAL,
                            brainPresentation, 20, comparison, eventPresentation);
                    String text = String.join(" ", lines.stream().map(NeuroCraftHud.Line::text).toList());
                    boolean permitted = fault == ControllerFault.ACTIVE
                            && comparison.status() == ComparisonLanePresentation.Status.UNAVAILABLE;
                    if (text.contains("APPLIED FEED") != permitted
                            || text.contains("+8.0e-04") != permitted
                            || text.contains("-1.0e-05") != permitted) {
                        throw new AssertionError("retained sample escaped state gate: " + text);
                    }
                    var control = lines.stream().filter(line -> line.text().startsWith("CONTROL "))
                            .findFirst().orElseThrow();
                    if (!permitted && control.color() != 0xFFFF5B5B) {
                        throw new AssertionError("unavailable control is colored active");
                    }
                    if (!permitted) {
                        var eventLine = lines.stream().filter(line -> line.text().startsWith("EVENT "))
                                .findFirst().orElseThrow();
                        if (!eventLine.text().equals("EVENT UNAVAILABLE · PAUSED/FAULTED")
                                || eventLine.color() != 0xFFFF5B5B) {
                            throw new AssertionError("inactive event display lost its fault styling");
                        }
                    }
                }
            }
            // A running body may lack a matching sidecar without losing control or having zero activity.
            for (var program : List.of(BodyProgramId.CRUISE, BodyProgramId.ESCAPE_JUMP)) {
                for (var status : EventDisplayState.Status.values()) {
                    if (status == EventDisplayState.Status.MATCHED || status == EventDisplayState.Status.DISABLED) {
                        continue;
                    }
                    var channel = new BodyProgramChannel(SelectionSource.READOUT, program,
                            Modulation.NEUTRAL, ProgramParameters.NEUTRAL);
                    var lines = NeuroCraftHud.showcaseLines(ControllerFault.ACTIVE, channel,
                            ReadoutValues.ZERO, PrimitiveAction.NEUTRAL, brainPresentation, 20,
                            ComparisonLanePresentation.unavailable(),
                            new EventDisplayPresentation(status, 0, Optional.empty()));
                    var eventLine = lines.stream().filter(line -> line.text().startsWith("EVENT "))
                            .findFirst().orElseThrow();
                    boolean waiting = status == EventDisplayState.Status.WAITING_FOR_APPLIED_COMMAND;
                    String expected = waiting ? "EVENT DISPLAY · awaiting applied match" : "EVENT DISPLAY · " + status;
                    if (!eventLine.text().equals(expected)
                            || eventLine.color() != (waiting ? 0xFFD7E0E7 : 0xFFF2D38A)) {
                        throw new AssertionError("diagnostic status looks like a control fault: " + eventLine);
                    }
                    String text = String.join(" ", lines.stream().map(NeuroCraftHud.Line::text).toList());
                    if (!text.contains("MN9 / antennal motors: unavailable")
                            || !text.contains("-1.0e-05") || text.contains("APPLIED")
                            || text.contains("TRIGGER") || text.contains("+8.0e-04")) {
                        throw new AssertionError("missing event evidence acquired unsupported meaning: " + text);
                    }
                    for (int width : new int[] {320, 347, 427}) {
                        var wrapped = NeuroCraftHud.wrapLines(client.font, lines, width);
                        int right = NeuroCraftHud.panelRight(client.font, wrapped, width);
                        for (var line : wrapped) {
                            if (NeuroCraftHud.TEXT_LEFT + client.font.width(line.text()) >= right) {
                                throw new AssertionError("event status clipped: " + line.text());
                            }
                        }
                        if (NeuroCraftHud.panelBottom(wrapped.size()) > 240) {
                            throw new AssertionError("event status obscures the required scope facts");
                        }
                    }
                }
            }
            for (int width : new int[] {320, 347, 427}) {
                for (String lane : List.of("original", "shuffled_weight", "no_input")) {
                    for (var source : List.of(SelectionSource.READOUT, SelectionSource.OPERATOR)) {
                        for (var status : List.of(ComparisonLanePresentation.Status.ACTIVE,
                                ComparisonLanePresentation.Status.SWITCHING, ComparisonLanePresentation.Status.FAILED)) {
                            boolean active = status == ComparisonLanePresentation.Status.ACTIVE;
                            var comparison = active ? new ComparisonLanePresentation(status, lane,
                                    event.identity().sessionEpoch(), Optional.of(event.identity()), lane,
                                    lane.equals("original") ? "faithful" : lane,
                                    lane.equals("original") ? null : "8".repeat(64),
                                    lane.equals("original") ? null : "18446744073709551615")
                                    : ComparisonLanePresentation.waiting(status, lane, event.identity().sessionEpoch());
                            var channel = BodyProgramChannel.decode(new BodyProgramChannel(source,
                                    BodyProgramId.FEED, Modulation.NEUTRAL, ProgramParameters.NEUTRAL).packed());
                            var lines = NeuroCraftHud.wrapLines(client.font, NeuroCraftHud.showcaseLines(
                                    active ? ControllerFault.ACTIVE : ControllerFault.STARTING,
                                    channel, new ReadoutValues(0, 0, 0), PrimitiveAction.NEUTRAL,
                                    brainPresentation, 20, comparison,
                                    active ? eventPresentation : EventDisplayPresentation.unavailable()), width);
                            String text = String.join(" ", lines.stream().map(NeuroCraftHud.Line::text).toList());
                            for (String fact : List.of(lane, "95.66x", "403.70x", "OFFLINE ESCAPE-ARC-01",
                                    "~300x", "35.1x", "4 neural steps", "NOT LIVE REACTION TIMING",
                                    "HAND-CHOSEN", "SCRIPTED", "128-NEURON FIXTURE", "SYNTHETIC IMITATION")) {
                                if (!text.contains(fact)) { throw new AssertionError("lost fact " + fact); }
                            }
                            int right = NeuroCraftHud.panelRight(client.font, lines, width);
                            for (var line : lines) {
                                if (NeuroCraftHud.TEXT_LEFT + client.font.width(line.text()) >= right) {
                                    throw new AssertionError("event HUD clipped: " + line.text());
                                }
                            }
                            if (NeuroCraftHud.panelBottom(lines.size()) > 240) {
                                throw new AssertionError("event HUD exceeds 240px: " + lines.size() + " " + text);
                            }
                            if (active && source == SelectionSource.READOUT) {
                                for (String label : List.of("EVENT #5 APPLIED FEED", "ISSUED #4 MN9 score 5.0e-04",
                                        "MN9 L +8.0e-04 R +2.0e-04", "Ant. motor L", "WHY landed feed")) {
                                    if (!text.contains(label)) { throw new AssertionError("missing evidence " + label); }
                                }
                                if (text.contains("APPLIED LAND")) { throw new AssertionError("issued origin became applied"); }
                            } else if (text.contains("APPLIED FEED") || text.contains("+8.0e-04")) {
                                throw new AssertionError("unavailable or paused means became current");
                            }
                        }
                    }
                }
            }
        });
    }

    private static EventDisplayFrame interactionFrame(String source, String program, boolean held) {
        // The base is production Python small-engine evidence, not a measured Minecraft session.
        // Source variants below are explicitly synthetic presentation boundary probes.
        try (var stream = EventHudClientGameTest.class.getResourceAsStream("/neurocraft/interaction-display-v2.json")) {
            if (stream == null) { throw new AssertionError("missing V2 Python diagnostic fixture"); }
            JsonObject root = JsonParser.parseString(new String(stream.readAllBytes(), StandardCharsets.UTF_8))
                    .getAsJsonObject();
            root.addProperty("program", program); root.addProperty("reason", "fixture_selection");
            var details = root.getAsJsonObject("interaction"); details.addProperty("source", source);
            if (source == null) {
                root.add("issued_trigger", com.google.gson.JsonNull.INSTANCE);
            } else {
                var gate = details.getAsJsonObject("scores").getAsJsonObject(source);
                if (source.equals("attack")) {
                    details.addProperty("raw_ttmn", -0.5); gate.addProperty("raw", -0.5);
                    gate.addProperty("baseline", -0.25); gate.addProperty("score", 0.25);
                } else {
                    gate.addProperty("baseline", 0.0);
                    gate.addProperty("score", Math.abs(gate.get("raw").getAsDouble()));
                }
                gate.addProperty("event_age", 0); gate.addProperty("baseline_ready", true);
                gate.addProperty("eligible", true);
                var trigger = root.getAsJsonObject("issued_trigger");
                trigger.addProperty("source", source); trigger.add("formula", gate.get("formula"));
                trigger.add("readout", gate.get("readout")); trigger.addProperty("issued_program", program);
                var issued = trigger.getAsJsonObject("issued_command"); issued.addProperty("program", program);
                issued.getAsJsonObject("readout").add("ttmn_jump", root.getAsJsonObject("readout").get("ttmn_jump"));
                var plain = gate.deepCopy(); plain.remove("formula"); plain.remove("readout");
                trigger.add("score", plain);
            }
            if (held) {
                // The qualifying issued trigger stays unchanged while current activity returns to baseline.
                for (String key : List.of("command_seq", "observation_seq", "observation_tick",
                        "target_tick", "valid_through_tick")) {
                    root.addProperty(key, root.get(key).getAsLong() + 1);
                }
                for (String key : List.of("neural_step_start", "neural_step_end")) {
                    root.addProperty(key, root.get(key).getAsLong() + 5);
                }
                root.addProperty("reason", "held");
                details.addProperty("raw_ttmn", -0.25);
                var gate = details.getAsJsonObject("scores").getAsJsonObject("attack");
                gate.addProperty("raw", -0.25); gate.addProperty("score", 0.0);
                gate.addProperty("event_age", 1);
                // This context flag remains true; it must not be called a passed neural gate.
                gate.addProperty("eligible", true);
            }
            return EventDisplayFrame.parse(new com.google.gson.GsonBuilder().serializeNulls().create().toJson(root));
        } catch (Exception exception) { throw new AssertionError(exception); }
    }

    private static void assertInteractionHud(Font font) {
        for (String[] mapping : new String[][] {{"attack", "FLEE"}, {"player_approach", "ESCAPE_JUMP"},
                {"mob_approach", "BANK"}, {"food", "FEED"}, {"brush", "GROOM"}, {"light", "HOVER"},
                {"attack", "FLEE", "held"}, {null, "HOVER"}}) {
            boolean held = mapping.length == 3;
            var sample = interactionFrame(mapping[0], mapping[1], held);
            var populations = new LinkedHashMap<String, BrainDisplayFrame.Population>();
            for (String name : BrainDisplayFrame.POPULATION_NAMES) {
                populations.put(name, new BrainDisplayFrame.Population(-0.00001, 1));
            }
            var brain = new BrainDisplayPresentation(BrainDisplayState.Status.MATCHED, 500, Optional.of(
                    new BrainDisplayFrame(sample.identity(), sample.commandSequence(), sample.observationTick(),
                            sample.command().neuralStepStart(), sample.command().neuralStepEnd(),
                            sample.command().stateDigest(), "neurocraft.interactions.encoder.v2", "3".repeat(64), populations)));
            for (String lane : List.of("original", "shuffled_weight", "no_input")) {
                for (String mode : List.of("matched", "paused", "operator", "switching", "stale", "waiting")) {
                    boolean visible = mode.equals("matched");
                    var comparison = mode.equals("switching")
                            ? ComparisonLanePresentation.waiting(ComparisonLanePresentation.Status.SWITCHING,
                                    lane, sample.identity().sessionEpoch())
                            : new ComparisonLanePresentation(ComparisonLanePresentation.Status.ACTIVE, lane,
                                    sample.identity().sessionEpoch(), Optional.of(sample.identity()), lane,
                                    lane.equals("original") ? "faithful" : lane,
                                    lane.equals("original") ? null : "8".repeat(64),
                                    lane.equals("original") ? null : "7");
                    var status = mode.equals("stale") ? EventDisplayState.Status.STALE
                            : mode.equals("waiting") ? EventDisplayState.Status.WAITING_FOR_APPLIED_COMMAND
                            : EventDisplayState.Status.MATCHED;
                    var events = EventDisplayPresentation.parseOrInvalid(new EventDisplayPresentation(status, 500,
                            status == EventDisplayState.Status.MATCHED ? Optional.of(sample) : Optional.empty()).encode());
                    var channel = new BodyProgramChannel(mode.equals("operator") ? SelectionSource.OPERATOR
                            : SelectionSource.READOUT, BodyProgramId.valueOf(mapping[1]), Modulation.NEUTRAL,
                            ProgramParameters.NEUTRAL);
                    var lines = NeuroCraftHud.showcaseLines(mode.equals("paused") ? ControllerFault.STARTING
                            : ControllerFault.ACTIVE, channel, sample.command().readout(), PrimitiveAction.NEUTRAL,
                            brain, 20, comparison, events);
                    String text = String.join(" ", lines.stream().map(NeuroCraftHud.Line::text).toList());
                    for (String fact : List.of("95.66x", "403.70x", "OFFLINE ESCAPE-ARC-01", "~300x",
                            "35.1x", "4 neural steps", "NOT LIVE REACTION TIMING", "HAND-CHOSEN", "SCRIPTED",
                            "128-NEURON FIXTURE", "SYNTHETIC IMITATION", lane)) {
                        if (!text.contains(fact)) { throw new AssertionError("V2 lost scope fact " + fact); }
                    }
                    if (text.contains("APPLIED " + mapping[1]) != visible
                            || text.contains("NOW ") != (visible && mapping[0] != null)
                            || text.contains("GATE") != (visible && mapping[0] == null)
                            || (text.contains("ISSUED #") != (visible && mapping[0] != null))) {
                        throw new AssertionError("V2 freshness/authority gate failed: " + mode + " " + text);
                    }
                    if (visible) {
                        var replaySample = new EventReplayHistory.Sample(sample, lane, lane.equals("original") ? null : "7");
                        String replay = String.join(" ", EventReplayScreen.sampleLines(replaySample)
                                .stream().map(NeuroCraftHud.Line::text).toList());
                        String replaySource = replaySample.inspectedSource();
                        String expectedReadout = replaySource == null ? "No eligible event readout" : switch (replaySource) {
                            case "brush" -> "Antennal motor";
                            case "food" -> "MN9";
                            case "light" -> "DNa01";
                            default -> "TTMn";
                        };
                        for (String required : List.of(expectedReadout, "HISTORICAL DISPLAY SAMPLE",
                                "Applied match #", "Numeric threshold and executed sensory vector: unavailable")) {
                            if (!replay.contains(required)) { throw new AssertionError("replay lost " + required); }
                        }
                        if ("brush".equals(mapping[0]) && (replay.contains("LC4") || replay.contains("DNp01")
                                || replay.contains("TTMn"))) {
                            throw new AssertionError("brush replay acquired escape-pathway activity");
                        }
                        String context = mapping[0] == null ? "SOURCE none · fallback HOVER" : switch (mapping[0]) {
                            case "attack" -> "ATTACK ATTEMPT · tactile proxy";
                            case "player_approach" -> "PLAYER approach · Minecraft category";
                            case "mob_approach" -> "MOB approach · Minecraft category";
                            case "food" -> "MN9 L";
                            case "brush" -> "Ant. motor L";
                            default -> "DNa01 L";
                        };
                        if (!text.contains(context)) { throw new AssertionError("missing interaction context " + context); }
                        if ("attack".equals(mapping[0]) && (!text.contains(held
                                ? "TTMn raw -2.5e-01" : "TTMn raw -5.0e-01")
                                || !text.contains("abs delta 2.5e-01"))) {
                            throw new AssertionError("attack lost signed activity or absolute-change disclosure");
                        }
                        if (held && (!text.contains("NOW abs delta 0.0e+00 · held")
                                || !text.contains("ISSUED #26 FLEE abs delta 2.5e-01")
                                || text.contains("· onset") || text.contains("GATE abs delta"))) {
                            throw new AssertionError("held context eligibility became a fresh neural gate: " + text);
                        }
                        if ("light".equals(mapping[0]) && (text.contains("fallback HOVER")
                                || !text.contains("NOW abs delta") || !text.contains("· onset"))) {
                            throw new AssertionError("light readout HOVER became fallback");
                        }
                    }
                    for (int width : new int[] {320, 347, 427}) {
                        var wrapped = NeuroCraftHud.wrapLines(font, lines, width);
                        int right = NeuroCraftHud.panelRight(font, wrapped, width);
                        for (var line : wrapped) {
                            if (NeuroCraftHud.TEXT_LEFT + font.width(line.text()) >= right) {
                                throw new AssertionError("V2 HUD clipped " + line.text());
                            }
                        }
                        if (NeuroCraftHud.panelBottom(wrapped.size()) > 240) {
                            throw new AssertionError("V2 HUD hides scope facts " + width + " " + text);
                        }
                    }
                }
            }
        }
    }
}
