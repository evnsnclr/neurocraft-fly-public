package dev.neurocraft.fly.trial;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.List;

/** Fixed engineering experiment, not a controller or a biological stimulus claim. */
public final class MatchedTrialPlan {
    public static final int BASELINE_TICKS = 20;
    public static final int APPROACH_TICKS = 60;
    public static final int RECOVERY_TICKS = 80;
    public static final int TOTAL_TICKS = BASELINE_TICKS + APPROACH_TICKS + RECOVERY_TICKS;
    public static final int ANCHOR_WAIT_TICKS = 200;
    public static final double INITIAL_FLOOR_CLEARANCE_BLOCKS = 2.0;
    public static final double START_X = 0.5, START_Y = 67.0, START_Z = 0.5;
    public static final List<Long> SEEDS = List.of(0L, 7919L, 15838L, 23757L, 31676L,
            39595L, 47514L, 55433L, 63352L, 71271L);
    public static final List<String> LANES = List.of("original", "shuffled_weight", "no_input");

    public enum Stimulus { ENTITY, BLOCK }
    public record Trial(int index, int pair, long bodySeed, Stimulus stimulus, String lane) {}
    private MatchedTrialPlan() {}

    public static List<Trial> trials() {
        List<Trial> trials = new ArrayList<>();
        for (int pair = 0; pair < SEEDS.size(); pair++) {
            for (Stimulus stimulus : Stimulus.values()) {
                for (String lane : LANES) {
                    trials.add(new Trial(trials.size(), pair, SEEDS.get(pair), stimulus, lane));
                }
            }
        }
        return List.copyOf(trials);
    }

    /** Six integer front planes, each present for ten authoritative ticks. */
    public static int frontPlane(long anchoredTick) {
        long approach = anchoredTick - BASELINE_TICKS;
        return approach < 0 || approach >= APPROACH_TICKS ? -1 : 6 - (int) (approach / 10);
    }

    public static String sha256(String serializedDocument) {
        try {
            return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                    .digest(serializedDocument.getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException exception) { throw new AssertionError(exception); }
    }

    public static JsonObject document() {
        JsonObject result = new JsonObject();
        result.addProperty("schema", "neurocraft.matched_trial_plan.v1");
        result.addProperty("diagnostic_only", true);
        result.addProperty("source", "declared scripted world stimulus; neural commands retain authority");
        result.addProperty("anchor", "applied program command 0, observation 0, neural steps 0..5");
        result.addProperty("baseline_ticks", BASELINE_TICKS);
        result.addProperty("approach_ticks", APPROACH_TICKS);
        result.addProperty("recovery_ticks", RECOVERY_TICKS);
        result.addProperty("anchor_wait_ticks", ANCHOR_WAIT_TICKS);
        result.addProperty("front_planes", "z=6,5,4,3,2,1; ten ticks each; x centre=.5; block bottom y=67");
        result.addProperty("entity", "minecraft:cow; no AI/gravity; actual minZ aligned to the same front plane");
        result.addProperty("geometry_limit", "front planes match; native cow and block shapes differ");
        result.addProperty("reset", "fresh authenticated epoch; fixture pose/velocity/events and independent body seed reset before bridge start");
        result.addProperty("initial_pose", "feet=(.5,67,.5); yaw=0; pitch=0; velocity=0; dry and airborne; full sturdy floor exactly 2 blocks below feet");
        result.addProperty("missing_anchor", "invalid trial, no retry; continue only while bridge is viable");
        JsonArray cells = new JsonArray();
        for (Trial trial : trials()) {
            JsonObject cell = new JsonObject();
            cell.addProperty("index", trial.index());
            cell.addProperty("pair", trial.pair());
            cell.addProperty("body_seed", trial.bodySeed());
            cell.addProperty("stimulus", trial.stimulus().name());
            cell.addProperty("lane", trial.lane());
            cells.add(cell);
        }
        result.add("trials", cells);
        return result;
    }
}
