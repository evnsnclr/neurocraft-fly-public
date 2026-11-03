package dev.neurocraft.fly.control;

import static dev.neurocraft.fly.control.BrainDisplayFrame.*;

import java.math.BigDecimal;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Immutable named event diagnostic. Its command is compared in full; it never grants authority. */
public record EventDisplayFrame(
        ProgramActionFrame command, String encoderHash, String bindingHash,
        Score mn9, Score antenna, String reason, Trigger trigger, Map<String, Object> payload,
        InteractionDetails interaction) {
    public static final String SCHEMA = "neurocraft.event_selection_display.v1";
    public static final String INTERACTION_SCHEMA = "neurocraft.interaction_selection_display.v2";
    public static final int MAX_BYTES = 16 * 1024;
    private static final Set<String> COMMAND_KEYS = Set.of("identity", "command_seq", "observation_seq",
            "observation_tick", "target_tick", "valid_through_tick", "neural_step_start",
            "neural_step_end", "state_digest", "policy_hash", "program", "modulation", "parameters", "readout");
    private static final Set<String> KEYS = Set.of("schema", "presentation_only", "selection_status",
            "identity", "command_seq", "observation_seq", "observation_tick", "target_tick",
            "valid_through_tick", "neural_step_start", "neural_step_end", "state_digest", "policy_hash",
            "encoder_hash", "binding_hash", "program", "modulation", "parameters", "readout",
            "command_sha256", "reason", "issued_trigger", "events", "requested_pulse", "pulse_scope",
            "mn9", "antenna", "quantity", "mapping");
    private static final Set<String> SCORE_KEYS = Set.of("raw", "baseline", "score", "eligible",
            "event_age", "baseline_ready");
    private static final Set<String> SOURCES = Set.of("attack", "player_approach", "mob_approach",
            "food", "brush", "light");

    public record Score(double raw, Double baseline, double score, boolean eligible,
            Long eventAge, boolean baselineReady, double left, double right) {}
    /** This was ISSUED at an earlier threshold crossing; no applied claim is attached to it. */
    public record Trigger(ProgramActionFrame command, String readout, Score score,
            String source, String formula) {
        public Trigger(ProgramActionFrame command, String readout, Score score) {
            this(command, readout, score, null, "positive_change");
        }
    }
    public record InteractionScore(Score value, String formula, String readout) {}
    public record InteractionDetails(String source, Map<String, InteractionScore> scores,
            double rawTtmn, double dna01Left, double dna01Right,
            List<Integer> rayHitKinds, List<Integer> rayActorKinds, int localLight) {
        public InteractionDetails {
            scores = Map.copyOf(scores);
            rayHitKinds = List.copyOf(rayHitKinds);
            rayActorKinds = List.copyOf(rayActorKinds);
        }
    }

    public EventDisplayFrame(ProgramActionFrame command, String encoderHash, String bindingHash,
            Score mn9, Score antenna, String reason, Trigger trigger, Map<String, Object> payload) {
        this(command, encoderHash, bindingHash, mn9, antenna, reason, trigger, payload, null);
    }

    public EventDisplayFrame {
        payload = immutable(payload);
    }

    public ControllerIdentity identity() { return command.identity(); }
    public long commandSequence() { return command.commandSequence(); }
    public long observationTick() { return command.observationTick(); }
    public boolean matches(CommandFrame applied) { return command.equals(applied); }
    Map<String, Object> toMap() { return payload; }

    public static EventDisplayFrame parse(String json) throws ProtocolException {
        Map<String, Object> root = StrictJson.parseObject(json, MAX_BYTES);
        boolean interactionVersion = INTERACTION_SCHEMA.equals(root.get("schema"));
        var keys = new java.util.HashSet<>(KEYS);
        if (interactionVersion) { keys.add("interaction"); }
        exact(root, keys);
        if ((!SCHEMA.equals(root.get("schema")) && !interactionVersion)
                || !Boolean.TRUE.equals(root.get("presentation_only"))
                || !"issued; requires authoritative applied-command match".equals(root.get("selection_status"))
                || !"requested before final control input gate; no_input delivers zero".equals(root.get("pulse_scope"))
                || !"signed dimensionless model activity; not spike rates".equals(root.get("quantity"))
                || !"engineered eligibility and hand-chosen readouts select scripted programs".equals(root.get("mapping"))) {
            throw new ProtocolException("unsupported event diagnostic declaration");
        }
        ProgramActionFrame command = command(root);
        hash(root, "command_sha256"); // Retained Python canonical digest, not rehashed with Java float formatting.
        String encoder = hash(root, "encoder_hash"), binding = hash(root, "binding_hash");
        String reason = text(root, "reason");
        if (!reason.matches("[a-z][a-z0-9_]{0,63}")) {
            throw new ProtocolException("invalid event reason");
        }
        events(object(root, "events"), command.observationTick());
        pulse(object(root, "requested_pulse"), interactionVersion);
        Score mn9 = score(object(root, "mn9"), 1, 1);
        Score antenna = score(object(root, "antenna"), 6, 7);
        Trigger trigger = root.get("issued_trigger") == null ? null
                : trigger(object(root, "issued_trigger"), command, interactionVersion);
        InteractionDetails interaction = interactionVersion
                ? interaction(object(root, "interaction"), command, mn9, antenna, trigger) : null;
        return new EventDisplayFrame(command, encoder, binding, mn9, antenna, reason, trigger, root,
                interaction);
    }

    private static ProgramActionFrame command(Map<String, Object> root) throws ProtocolException {
        Map<String, Object> values = new LinkedHashMap<>();
        values.put("type", "program_action");
        values.put("protocol", ControlProtocol.PROTOCOL_VERSION);
        for (String key : COMMAND_KEYS) { values.put(key, root.get(key)); }
        return (ProgramActionFrame) ControlProtocol.decodeInbound(StrictJson.encode(values, MAX_BYTES));
    }

    private static Trigger trigger(Map<String, Object> root, ProgramActionFrame current, boolean interactionVersion)
            throws ProtocolException {
        var keys = new java.util.HashSet<>(Set.of("identity", "command_seq", "observation_seq", "observation_tick",
                "target_tick", "valid_through_tick", "neural_step_start", "neural_step_end",
                "state_digest", "policy_hash", "issued_program", "issued_command", "command_sha256",
                "readout", "score", "events"));
        if (interactionVersion) { keys.addAll(Set.of("source", "formula")); }
        exact(root, keys);
        var decoded = ControlProtocol.decodeInbound(StrictJson.encode(object(root, "issued_command"), MAX_BYTES));
        if (!(decoded instanceof ProgramActionFrame issued)) {
            throw new ProtocolException("event trigger requires a program command");
        }
        Map<String, Object> duplicate = new LinkedHashMap<>(object(root, "issued_command"));
        for (String key : COMMAND_KEYS) {
            if (!Set.of("program", "modulation", "parameters", "readout").contains(key)) {
                duplicate.put(key, root.get(key));
            }
        }
        duplicate.put("program", root.get("issued_program"));
        if (!issued.equals(command(duplicate)) || !issued.identity().equals(current.identity())
                || !issued.policyHash().equals(current.policyHash())
                || issued.commandSequence() > current.commandSequence()
                || issued.observationTick() > current.observationTick()
                || issued.neuralStepEnd() > current.neuralStepEnd()) {
            throw new ProtocolException("event trigger envelope does not bind its issued command");
        }
        if (interactionVersion && issued.commandSequence() == current.commandSequence()
                && !issued.equals(current)) {
            throw new ProtocolException("current interaction trigger differs from its exact command");
        }
        hash(root, "command_sha256");
        String readout = text(root, "readout");
        String source = interactionVersion ? text(root, "source") : null;
        String formula = interactionVersion ? text(root, "formula") : "positive_change";
        if (interactionVersion) {
            sourceMapping(source, readout, formula);
            sourceProgram(source, issued.program().name());
        } else if (!Set.of("mn9", "antenna").contains(readout)) {
            throw new ProtocolException("unknown event trigger readout");
        }
        events(object(root, "events"), issued.observationTick());
        Score score = score(object(root, "score"), 0, 0, formula);
        if (interactionVersion && "absolute_readout".equals(formula)
                && Double.compare(score.raw(), issued.readout().ttmnJump()) != 0) {
            throw new ProtocolException("trigger readout differs from issued command");
        }
        return new Trigger(issued, readout, score, source, formula);
    }

    private static Score score(Map<String, Object> values, int leftCount, int rightCount)
            throws ProtocolException {
        return score(values, leftCount, rightCount, "positive_change");
    }

    private static Score score(Map<String, Object> values, int leftCount, int rightCount, String formula)
            throws ProtocolException {
        var keys = new java.util.HashSet<>(SCORE_KEYS);
        if (leftCount != 0) { keys.addAll(Set.of("left", "right", "left_count", "right_count")); }
        exact(values, keys);
        double raw = number(values, "raw", -1, 1);
        Double baseline = values.get("baseline") == null ? null : number(values, "baseline", -1, 1);
        double score = number(values, "score", 0, 2);
        Long age = values.get("event_age") == null ? null : integer(values, "event_age");
        boolean ready = bool(values, "baseline_ready"), eligible = bool(values, "eligible");
        double expected = baseline == null ? 0.0 : switch (formula) {
            case "positive_change" -> Math.max(0.0, raw - baseline);
            case "absolute_change" -> Math.abs(raw - baseline);
            case "absolute_readout" -> raw;
            default -> throw new ProtocolException("unsupported interaction formula");
        };
        if ((baseline == null) != (age == null) || (baseline != null && !ready)
                || ("absolute_readout".equals(formula) && (raw < 0 || (baseline != null && baseline != 0)))
                || Double.compare(score, expected) != 0) {
            throw new ProtocolException("inconsistent declared event score");
        }
        double left = raw, right = raw;
        if (leftCount != 0) {
            if (integer(values, "left_count") != leftCount || integer(values, "right_count") != rightCount) {
                throw new ProtocolException("wrong named population counts");
            }
            left = number(values, "left", -1, 1); right = number(values, "right", -1, 1);
        }
        return new Score(raw, baseline, score, eligible, age, ready, left, right);
    }

    private static InteractionDetails interaction(Map<String, Object> values, ProgramActionFrame command,
            Score mn9, Score antenna, Trigger trigger) throws ProtocolException {
        exact(values, Set.of("source", "scores", "raw_ttmn", "dna01_left", "dna01_right",
                "ray_hit_kinds", "ray_actor_kinds", "local_light"));
        String source = values.get("source") == null ? null : text(values, "source");
        if (source == null) {
            if (trigger != null) { throw new ProtocolException("unselected interaction has a trigger"); }
        } else {
            sourceProgram(source, command.program().name());
            if (trigger == null || !source.equals(trigger.source())) {
                throw new ProtocolException("selected interaction differs from issued trigger");
            }
            if (trigger.command().program() != command.program()
                    && !(trigger.command().program().name().equals("LAND")
                            && command.program().name().equals("FEED"))) {
                throw new ProtocolException("interaction hold reversed its issued program");
            }
        }
        double ttmn = number(values, "raw_ttmn", -1, 1);
        double left = number(values, "dna01_left", -1, 1);
        double right = number(values, "dna01_right", -1, 1);
        long localLight = integer(values, "local_light");
        if (localLight > 15) { throw new ProtocolException("local light outside bounds"); }
        List<Integer> hits = kinds(values, "ray_hit_kinds", 2);
        List<Integer> actors = kinds(values, "ray_actor_kinds", 3);
        for (int index = 0; index < 6; index++) {
            if ((hits.get(index) != 2) != (actors.get(index) == 0)) {
                throw new ProtocolException("actor category does not match winning ray hit");
            }
        }
        Map<String, Object> scores = object(values, "scores");
        exact(scores, SOURCES);
        Map<String, InteractionScore> parsed = new LinkedHashMap<>();
        for (String name : SOURCES) {
            Map<String, Object> declared = object(scores, name);
            var keys = new java.util.HashSet<>(SCORE_KEYS);
            keys.addAll(Set.of("readout", "formula"));
            exact(declared, keys);
            String readout = text(declared, "readout"), formula = text(declared, "formula");
            sourceMapping(name, readout, formula);
            Map<String, Object> plain = new LinkedHashMap<>(declared);
            plain.remove("readout"); plain.remove("formula");
            Score parsedScore = score(plain, 0, 0, formula);
            double raw = switch (name) {
                case "attack" -> ttmn;
                case "player_approach", "mob_approach" -> command.readout().ttmnJump();
                case "food" -> mn9.raw();
                case "brush" -> antenna.raw();
                case "light" -> (left + right) * 0.5;
                default -> throw new ProtocolException("unknown interaction source");
            };
            if (Double.compare(parsedScore.raw(), raw) != 0) {
                throw new ProtocolException("interaction raw differs from named readout");
            }
            parsed.put(name, new InteractionScore(parsedScore, formula, readout));
        }
        return new InteractionDetails(source, parsed, ttmn, left, right, hits, actors, (int) localLight);
    }

    private static void sourceMapping(String source, String readout, String formula) throws ProtocolException {
        if (!SOURCES.contains(source)) { throw new ProtocolException("unknown interaction source"); }
        String expectedReadout = switch (source) {
            case "attack" -> "ttmn_raw";
            case "player_approach", "mob_approach" -> "ttmn_jump";
            case "food" -> "mn9";
            case "brush" -> "antenna";
            default -> "dna01_bilateral";
        };
        String expectedFormula = switch (source) {
            case "attack", "light" -> "absolute_change";
            case "player_approach", "mob_approach" -> "absolute_readout";
            default -> "positive_change";
        };
        if (!expectedReadout.equals(readout) || !expectedFormula.equals(formula)) {
            throw new ProtocolException("interaction source/readout/formula disagree");
        }
    }

    private static void sourceProgram(String source, String program) throws ProtocolException {
        boolean valid = switch (source) {
            case "attack" -> program.equals("FLEE");
            case "player_approach" -> program.equals("ESCAPE_JUMP");
            case "mob_approach" -> program.equals("BANK");
            case "food" -> Set.of("LAND", "FEED").contains(program);
            case "brush" -> program.equals("GROOM");
            case "light" -> program.equals("HOVER");
            default -> false;
        };
        if (!valid) { throw new ProtocolException("interaction source/program disagree"); }
    }

    private static List<Integer> kinds(Map<String, Object> values, String key, int maximum)
            throws ProtocolException {
        if (!(values.get(key) instanceof List<?> items) || items.size() != 6) {
            throw new ProtocolException("six ray kinds required");
        }
        var result = new java.util.ArrayList<Integer>();
        for (Object value : items) {
            long kind = integer(Map.of("kind", value == null ? "null" : value), "kind");
            if (kind > maximum) { throw new ProtocolException("ray kind outside bounds"); }
            result.add((int) kind);
        }
        return List.copyOf(result);
    }

    private static void events(Map<String, Object> values, long tick) throws ProtocolException {
        exact(values, Set.of("food_contact", "food_near", "food", "brush", "attack_attempt"));
        boolean contact = bool(values, "food_contact"), near = bool(values, "food_near");
        if (contact && !near) { throw new ProtocolException("food contact requires near"); }
        for (String name : new String[] {"food", "brush", "attack_attempt"}) {
            Map<String, Object> counter = object(values, name);
            exact(counter, Set.of("count", "last_game_tick"));
            long count = integer(counter, "count");
            Long last = counter.get("last_game_tick") == null ? null : integer(counter, "last_game_tick");
            if ((count == 0) != (last == null) || (last != null && last > tick)) {
                throw new ProtocolException("invalid event counter timestamp");
            }
        }
    }

    private static void pulse(Map<String, Object> values, boolean interactionVersion) throws ProtocolException {
        var keys = new java.util.HashSet<>(Set.of("food", "groom", "food_age", "groom_age", "food_delta", "brush_delta",
                "attack_delta", "baseline_absorbed"));
        if (interactionVersion) { keys.addAll(Set.of("attack", "attack_age")); }
        exact(values, keys);
        boolean absorbed = bool(values, "baseline_absorbed");
        for (String name : interactionVersion ? new String[] {"food", "groom", "attack"}
                : new String[] {"food", "groom"}) {
            boolean requested = bool(values, name);
            Long age = values.get(name + "_age") == null ? null : integer(values, name + "_age");
            if (requested != (age != null) || (age != null && age > 8) || (absorbed && requested)) {
                throw new ProtocolException("invalid requested event pulse");
            }
        }
        for (String name : new String[] {"food_delta", "brush_delta", "attack_delta"}) { integer(values, name); }
    }

    private static boolean bool(Map<String, Object> values, String key) throws ProtocolException {
        if (!(values.get(key) instanceof Boolean result)) { throw new ProtocolException(key + " must be boolean"); }
        return result;
    }
    private static double number(Map<String, Object> values, String key, double min, double max)
            throws ProtocolException {
        if (!(values.get(key) instanceof BigDecimal value) || !Double.isFinite(value.doubleValue())
                || value.doubleValue() < min || value.doubleValue() > max) {
            throw new ProtocolException(key + " outside finite bounds");
        }
        return value.doubleValue();
    }
    private static String hash(Map<String, Object> values, String key) throws ProtocolException {
        String value = text(values, key);
        if (!value.matches("[0-9a-f]{64}")) { throw new ProtocolException("invalid " + key); }
        return value;
    }
    private static Map<String, Object> immutable(Map<String, Object> values) {
        Map<String, Object> copy = new LinkedHashMap<>();
        values.forEach((key, value) -> {
            if (value instanceof Map<?, ?> nested) {
                @SuppressWarnings("unchecked") var typed = (Map<String, Object>) nested;
                copy.put(key, immutable(typed));
            } else if (value instanceof List<?> nested) {
                copy.put(key, nested.stream().map(EventDisplayFrame::immutableValue).toList());
            } else { copy.put(key, value); }
        });
        return Collections.unmodifiableMap(copy);
    }
    private static Object immutableValue(Object value) {
        if (value instanceof Map<?, ?> nested) {
            @SuppressWarnings("unchecked") var typed = (Map<String, Object>) nested;
            return immutable(typed);
        }
        if (value instanceof List<?> nested) { return nested.stream().map(EventDisplayFrame::immutableValue).toList(); }
        return value;
    }
}
