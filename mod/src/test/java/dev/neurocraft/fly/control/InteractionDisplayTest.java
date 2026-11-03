package dev.neurocraft.fly.control;

import static org.junit.jupiter.api.Assertions.*;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/** Synthetic boundary cases supplement the production Python V2 golden. */
final class InteractionDisplayTest {
    @Test void readsRealSmallEnginePythonGoldenIncludingIssuedLightOrigin() throws Exception {
        String golden = Files.readString(Path.of("..", "packages", "protocol", "golden", "interaction_display_v2.json"));
        assertEquals(golden, Files.readString(Path.of("src", "gametest", "resources", "neurocraft",
                "interaction-display-v2.json")));
        var sample = EventDisplayFrame.parse(golden);
        assertEquals("light", sample.interaction().source());
        assertEquals("HOVER", sample.command().program().name());
        assertEquals("absolute_change", sample.trigger().formula());
        assertEquals(sample.trigger().command().commandSequence(), sample.commandSequence());
        assertEquals(sample, EventDisplayTest.parse(sample.toMap()));
        var wrapped = new EventDisplayPresentation(EventDisplayState.Status.MATCHED, 500,
                java.util.Optional.of(sample));
        assertEquals(wrapped, EventDisplayPresentation.parse(wrapped.encode()));
    }

    private static Map<String, Object> score(double raw, Double baseline, String readout, String formula) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("raw", raw); result.put("baseline", baseline);
        result.put("score", baseline == null ? 0 : switch (formula) {
            case "absolute_change" -> Math.abs(raw - baseline);
            case "absolute_readout" -> raw;
            default -> Math.max(0, raw - baseline);
        });
        result.put("eligible", true); result.put("event_age", baseline == null ? null : 0);
        result.put("baseline_ready", true); result.put("readout", readout); result.put("formula", formula);
        return result;
    }

    static Map<String, Object> values() throws Exception {
        var root = EventDisplayTest.values();
        root.put("schema", EventDisplayFrame.INTERACTION_SCHEMA);
        var pulse = EventDisplayTest.object(root, "requested_pulse");
        pulse.put("attack", false); pulse.put("attack_age", null);
        var scores = new LinkedHashMap<String, Object>();
        scores.put("attack", score(-0.5, -0.25, "ttmn_raw", "absolute_change"));
        scores.put("player_approach", score(0.1, 0.0, "ttmn_jump", "absolute_readout"));
        scores.put("mob_approach", score(0.1, 0.0, "ttmn_jump", "absolute_readout"));
        scores.put("food", score(0.0005, 0.0, "mn9", "positive_change"));
        scores.put("brush", score(0.0006, null, "antenna", "positive_change"));
        scores.put("light", score(-0.125, -0.25, "dna01_bilateral", "absolute_change"));
        var details = new LinkedHashMap<String, Object>();
        details.put("source", "food"); details.put("scores", scores); details.put("raw_ttmn", -0.5);
        details.put("dna01_left", -0.5); details.put("dna01_right", 0.25);
        details.put("ray_hit_kinds", List.of(2, 2, 2, 0, 1, 0));
        details.put("ray_actor_kinds", List.of(1, 2, 3, 0, 0, 0)); details.put("local_light", 15);
        root.put("interaction", details);
        var trigger = EventDisplayTest.object(root, "issued_trigger");
        trigger.put("source", "food"); trigger.put("formula", "positive_change");
        return root;
    }

    static void select(Map<String, Object> root, String source, String program) throws Exception {
        var details = EventDisplayTest.object(root, "interaction");
        details.put("source", source); root.put("program", program);
        var gate = EventDisplayTest.object(EventDisplayTest.object(details, "scores"), source);
        var trigger = EventDisplayTest.object(root, "issued_trigger");
        trigger.put("source", source); trigger.put("formula", gate.get("formula"));
        trigger.put("readout", gate.get("readout")); trigger.put("issued_program", program);
        EventDisplayTest.object(trigger, "issued_command").put("program", program);
        var plain = new LinkedHashMap<>(gate); plain.remove("readout"); plain.remove("formula");
        trigger.put("score", plain);
    }

    @Test void acceptsNegativeTtmnAbsoluteChangeAndPreservesEarlierTriggerDuringHold() throws Exception {
        var raw = values(); select(raw, "attack", "FLEE");
        var sample = EventDisplayTest.parse(raw);
        assertEquals(-0.5, sample.interaction().rawTtmn());
        assertEquals(0.25, sample.interaction().scores().get("attack").value().score());
        assertEquals("absolute_change", sample.trigger().formula());
        assertEquals(-0.5, sample.trigger().score().raw());
        // A held program does not imply that its current score is still above the issuing threshold.
        var attack = EventDisplayTest.object(EventDisplayTest.object(
                EventDisplayTest.object(raw, "interaction"), "scores"), "attack");
        attack.put("raw", -0.25); attack.put("score", 0.0); attack.put("eligible", false);
        EventDisplayTest.object(raw, "interaction").put("raw_ttmn", -0.25);
        var held = EventDisplayTest.parse(raw);
        assertEquals(0, held.interaction().scores().get("attack").value().score());
        assertEquals(0.25, held.trigger().score().score());
        var state = new EventDisplayState(held.identity()); state.recordApplied(held.trigger().command());
        assertEquals(EventDisplayState.Status.WAITING_FOR_APPLIED_COMMAND,
                state.view(EventDisplayTest.snapshot(held, 100), 101, 7).status());
        state.recordApplied(held.command());
        assertEquals(EventDisplayState.Status.MATCHED,
                state.view(EventDisplayTest.snapshot(held, 100), 101, 7).status());
    }

    @Test void allSourceProgramsAndLandToFeedOriginDecodeAndCollectionsAreImmutable() throws Exception {
        for (String[] mapping : new String[][] {{"attack", "FLEE"}, {"player_approach", "ESCAPE_JUMP"},
                {"mob_approach", "BANK"}, {"food", "LAND"}, {"food", "FEED"},
                {"brush", "GROOM"}, {"light", "HOVER"}}) {
            var raw = values(); select(raw, mapping[0], mapping[1]);
            assertEquals(mapping[0], EventDisplayTest.parse(raw).interaction().source());
        }
        var sample = EventDisplayTest.parse(values());
        assertEquals("LAND", sample.trigger().command().program().name());
        assertEquals("FEED", sample.command().program().name());
        assertEquals(sample, EventDisplayTest.parse(sample.toMap()));
        assertThrows(UnsupportedOperationException.class, () -> sample.interaction().scores().clear());
        assertThrows(UnsupportedOperationException.class, () -> sample.interaction().rayActorKinds().clear());
        assertThrows(UnsupportedOperationException.class,
                () -> ((List<?>) EventDisplayTest.object(sample.toMap(), "interaction").get("ray_hit_kinds")).clear());
        var fallback = values(); EventDisplayTest.object(fallback, "interaction").put("source", null);
        fallback.put("issued_trigger", null); fallback.put("program", "HOVER");
        assertNull(EventDisplayTest.parse(fallback).interaction().source());
    }

    @Test void refusesMalformedScoreFormulaRawBindingAndActorMetadata() throws Exception {
        for (String key : List.of("formula", "readout", "raw", "baseline", "score", "event_age", "baseline_ready")) {
            var root = values(); var score = EventDisplayTest.object(EventDisplayTest.object(
                    EventDisplayTest.object(root, "interaction"), "scores"), "attack");
            score.put(key, switch (key) {
                case "formula" -> "positive_change";
                case "readout" -> "ttmn_jump";
                case "raw" -> 0.5;
                case "baseline" -> null;
                case "event_age" -> null;
                case "baseline_ready" -> false;
                default -> 0;
            });
            assertThrows(ProtocolException.class, () -> EventDisplayTest.parse(root), key);
        }
        for (Object actors : List.of(List.of(0, 2, 3, 0, 0, 0), List.of(1, 2, 3, 1, 0, 0),
                List.of(1, 2, 4, 0, 0, 0), List.of(1, 2, -1, 0, 0, 0),
                List.of(1, 2, true, 0, 0, 0), List.of(1, 2, 0.5, 0, 0, 0), List.of(1, 2, 3))) {
            var root = values(); EventDisplayTest.object(root, "interaction").put("ray_actor_kinds", actors);
            assertThrows(ProtocolException.class, () -> EventDisplayTest.parse(root));
        }
        var light = values(); EventDisplayTest.object(light, "interaction").put("local_light", 16);
        assertThrows(ProtocolException.class, () -> EventDisplayTest.parse(light));
        var ray = values(); EventDisplayTest.object(ray, "interaction").put("ray_hit_kinds", List.of(3, 2, 2, 0, 1, 0));
        assertThrows(ProtocolException.class, () -> EventDisplayTest.parse(ray));
        var pulse = values(); EventDisplayTest.object(pulse, "requested_pulse").put("attack", true);
        assertThrows(ProtocolException.class, () -> EventDisplayTest.parse(pulse));
    }

    @Test void refusesSourceMismatchTriggerSpoofAndNonzeroAbsoluteReadoutBaseline() throws Exception {
        for (String source : new String[] {null, "attack", "injury", "mob_approach"}) {
            var root = values(); EventDisplayTest.object(root, "interaction").put("source", source);
            assertThrows(ProtocolException.class, () -> EventDisplayTest.parse(root));
        }
        for (String field : List.of("source", "formula", "readout", "issued_program")) {
            var root = values(); EventDisplayTest.object(root, "issued_trigger").put(field, "attack");
            assertThrows(ProtocolException.class, () -> EventDisplayTest.parse(root));
        }
        var root = values(); var approach = EventDisplayTest.object(EventDisplayTest.object(
                EventDisplayTest.object(root, "interaction"), "scores"), "player_approach");
        approach.put("baseline", 0.1);
        assertThrows(ProtocolException.class, () -> EventDisplayTest.parse(root));
        var noTrigger = values(); noTrigger.put("issued_trigger", null);
        assertThrows(ProtocolException.class, () -> EventDisplayTest.parse(noTrigger));
        var future = values(); EventDisplayTest.object(future, "issued_trigger").put("command_seq", 6);
        assertThrows(ProtocolException.class, () -> EventDisplayTest.parse(future));
        var backwards = values(); select(backwards, "food", "FEED"); backwards.put("program", "LAND");
        assertThrows(ProtocolException.class, () -> EventDisplayTest.parse(backwards));
    }

    @Test void selfConsistentScoresStillMustBindTheActualNamedReadouts() throws Exception {
        for (String source : List.of("attack", "player_approach", "mob_approach", "food", "brush", "light")) {
            var root = values(); var scores = EventDisplayTest.object(EventDisplayTest.object(root, "interaction"), "scores");
            var old = EventDisplayTest.object(scores, source);
            // This score is arithmetically valid; it must fail solely because it names another raw value.
            scores.put(source, score(0.25, 0.0, (String) old.get("readout"), (String) old.get("formula")));
            assertThrows(ProtocolException.class, () -> EventDisplayTest.parse(root), source);
        }
        var root = values(); select(root, "player_approach", "ESCAPE_JUMP");
        var trigger = EventDisplayTest.object(root, "issued_trigger");
        EventDisplayTest.object(EventDisplayTest.object(trigger, "issued_command"), "readout").put("ttmn_jump", 0.25);
        assertThrows(ProtocolException.class, () -> EventDisplayTest.parse(root));
    }

    @Test void closedV2ShapeDoesNotAdmitPartialOrInventedSourceFields() throws Exception {
        var missing = values(); EventDisplayTest.object(EventDisplayTest.object(missing, "interaction"), "scores").remove("light");
        assertThrows(ProtocolException.class, () -> EventDisplayTest.parse(missing));
        var extra = values(); EventDisplayTest.object(extra, "interaction").put("injury", true);
        assertThrows(ProtocolException.class, () -> EventDisplayTest.parse(extra));
        var pulse = values(); EventDisplayTest.object(pulse, "requested_pulse").remove("attack_age");
        assertThrows(ProtocolException.class, () -> EventDisplayTest.parse(pulse));
        var extraScore = values(); EventDisplayTest.object(EventDisplayTest.object(
                EventDisplayTest.object(extraScore, "interaction"), "scores"), "light").put("left_count", 1);
        assertThrows(ProtocolException.class, () -> EventDisplayTest.parse(extraScore));
    }

    @Test void sameSequenceTriggerMustBeTheExactCurrentCommand() throws Exception {
        String golden = Files.readString(Path.of("..", "packages", "protocol", "golden", "interaction_display_v2.json"));
        var valid = EventDisplayFrame.parse(golden);
        assertEquals(valid.command(), valid.trigger().command());
        for (String key : List.of("modulation", "parameters", "readout", "state_digest", "observation_seq")) {
            var root = StrictJson.parseObject(golden, EventDisplayFrame.MAX_BYTES);
            var trigger = EventDisplayTest.object(root, "issued_trigger");
            var issued = EventDisplayTest.object(trigger, "issued_command");
            switch (key) {
                case "modulation" -> EventDisplayTest.object(issued, key).put("amplitude", 0.5);
                case "parameters" -> EventDisplayTest.object(issued, key).put("turn_bias", 0.5);
                case "readout" -> EventDisplayTest.object(issued, key).put("dna01_turn", 0.5);
                case "state_digest" -> { issued.put(key, "9".repeat(64)); trigger.put(key, "9".repeat(64)); }
                case "observation_seq" -> { issued.put(key, 0); trigger.put(key, 0); }
                default -> throw new AssertionError(key);
            }
            // The duplicate trigger envelope is self-consistent and its sequence is current.
            // It still cannot substitute another command for the actual current application.
            assertThrows(ProtocolException.class, () -> EventDisplayTest.parse(root), key);
        }
        var held = values();
        EventDisplayTest.object(EventDisplayTest.object(EventDisplayTest.object(held, "issued_trigger"),
                "issued_command"), "modulation").put("amplitude", 0.5);
        var sample = EventDisplayTest.parse(held);
        assertEquals("LAND", sample.trigger().command().program().name());
        assertEquals("FEED", sample.command().program().name());
        assertNotEquals(sample.command().modulation(), sample.trigger().command().modulation());
    }
}
