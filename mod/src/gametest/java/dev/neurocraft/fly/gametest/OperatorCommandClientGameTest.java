package dev.neurocraft.fly.gametest;

import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.body.BodyProgramRuntime;
import dev.neurocraft.fly.entity.FlyEntity;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;
import net.fabricmc.fabric.api.client.gametest.v1.FabricClientGameTest;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.fabricmc.fabric.api.client.gametest.v1.context.TestSingleplayerContext;
import net.minecraft.commands.CommandSource;
import net.minecraft.commands.CommandSourceStack;
import net.minecraft.network.chat.Component;
import net.minecraft.world.entity.EntitySpawnReason;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The operator commands' chat text, executed through the real dispatcher against a world that
 * holds exactly one fly.
 *
 * <p>A client GameTest rather than a server one on purpose: the {@code status} family reports the
 * single loaded fly, and server GameTests in a batch tick together in one shared level where other
 * classes spawn flies of their own, so "exactly one" is not a premise a server-side pin can
 * arrange. {@code worldBuilder().create()} gives this test a fresh integrated server with nothing
 * in it, and client GameTests run one after another.
 *
 * <p>Pins lane L's two operator findings (2026-09-04). {@code /neurocraftfly status} sent one
 * ~1,100-character line that wrapped to some twenty chat lines and covered the HUD for ten seconds
 * each time it was typed; the compact form must carry the fields an operator glances for and none
 * of the long form's, and the long form must still be reachable in full. Bare
 * {@code /neurocraftfly program} answered with Brigadier's "Unknown or incomplete command"; it must
 * print usage naming every registered program and {@code release}.
 */
public final class OperatorCommandClientGameTest implements FabricClientGameTest {
    private static final Logger LOGGER =
            LoggerFactory.getLogger("neurocraft-fly-command-test");
    /** Characters. Roughly four chat lines at the default chat width; the long form is ~1,100. */
    private static final int COMPACT_STATUS_CEILING = 220;

    private static final String[] COMPACT_FIELDS = {
        "bridge=",
        "authority=",
        "program=",
        "source=",
        "since_tick=",
        "transitions=",
        "lateness_retired=",
        "lane=",
        "policy_hash_prefix="
    };

    private static final String[] LONG_FORM_ONLY = {
        "applied_motors=", "modulation=(", "readout_owned_ticks=", "program_receipts_sent="
    };

    @Override
    public void runTest(ClientGameTestContext context) {
        try (TestSingleplayerContext singleplayer = context.worldBuilder().create()) {
            singleplayer.getServer().runCommand("tp @p 0 80 0 0 0");
            singleplayer.getClientWorld().waitForChunksRender();
            singleplayer.getServer().runOnServer(server -> {
                FlyEntity fly =
                        NeuroCraftFly.FLY.create(server.overworld(), EntitySpawnReason.COMMAND);
                if (fly == null) {
                    throw new AssertionError("could not construct the command-probe fly");
                }
                fly.resetForFixture(0.0, 81.5, 4.0, 180.0F);
                if (!server.overworld().addFreshEntity(fly)) {
                    throw new AssertionError("could not spawn the command-probe fly");
                }
            });
            context.waitTicks(2);

            String compact = single(execute(singleplayer, "neurocraftfly status"), "status");
            String full = single(execute(singleplayer, "neurocraftfly status full"), "status full");
            String inspect = single(execute(singleplayer, "neurocraftfly inspect"), "inspect");
            String usage = single(execute(singleplayer, "neurocraftfly program"), "program");

            LOGGER.info("/neurocraftfly status -> {} chars: {}", compact.length(), compact);
            LOGGER.info("/neurocraftfly status full -> {} chars", full.length());
            LOGGER.info("/neurocraftfly program -> {}", usage);
            assertCompactStatus(compact);
            assertFullStatus(full, "status full");
            assertFullStatus(inspect, "inspect");
            if (full.length() <= compact.length()) {
                throw new AssertionError("the full status is not longer than the compact one:\n  "
                        + full + "\n  " + compact);
            }
            assertProgramUsage(usage);
        }
    }

    private static void assertCompactStatus(String line) {
        if (!line.startsWith("NeuroCraft bridge=")) {
            throw new AssertionError("compact status does not open with the bridge state: " + line);
        }
        for (String field : COMPACT_FIELDS) {
            if (!line.contains(" " + field) && !line.startsWith("NeuroCraft " + field)) {
                throw new AssertionError("compact status lacks " + field + ": " + line);
            }
        }
        for (String field : LONG_FORM_ONLY) {
            if (line.contains(field)) {
                throw new AssertionError("compact status carries the long form's " + field + ": "
                        + line);
            }
        }
        if (line.length() > COMPACT_STATUS_CEILING) {
            throw new AssertionError("compact status is " + line.length() + " characters, over the "
                    + COMPACT_STATUS_CEILING + "-character ceiling: " + line);
        }
    }

    private static void assertFullStatus(String line, String command) {
        for (String field : new String[] {
            "applied_motors=(",
            "lateness_retired=",
            "program_owned_ticks=",
            "readout_owned_ticks=",
            "policy_hash=",
            "program_receipts_sent="
        }) {
            if (!line.contains(field)) {
                throw new AssertionError(command + " no longer reports " + field + ": " + line);
            }
        }
    }

    private static void assertProgramUsage(String line) {
        if (line.contains("Unknown or incomplete command")) {
            throw new AssertionError("bare program still fails as an incomplete command: " + line);
        }
        String names = BodyProgramRuntime.registeredPrograms().stream()
                .sorted()
                .map(id -> id.name().toLowerCase(Locale.ROOT))
                .collect(Collectors.joining("|"));
        if (!line.contains("/neurocraftfly program <" + names + ">")) {
            throw new AssertionError(
                    "program usage does not name the registered programs <" + names + ">: " + line);
        }
        if (!line.contains("/neurocraftfly program release")) {
            throw new AssertionError("program usage does not name release: " + line);
        }
    }

    private static String single(List<String> messages, String command) {
        if (messages.size() != 1) {
            throw new AssertionError(command + " sent " + messages.size()
                    + " chat message(s), expected 1: " + messages);
        }
        return messages.getFirst();
    }

    /** Runs one operator command as a permission-4 source and returns what it sent to chat. */
    private static List<String> execute(TestSingleplayerContext singleplayer, String command) {
        return singleplayer.getServer().computeOnServer(server -> {
            CapturingSource capture = new CapturingSource();
            CommandSourceStack source = server.createCommandSourceStack().withSource(capture);
            server.getCommands().performPrefixedCommand(source, command);
            return List.copyOf(capture.messages);
        });
    }

    /** Collects exactly what a player would have read in chat, successes and failures alike. */
    private static final class CapturingSource implements CommandSource {
        private final List<String> messages = new ArrayList<>();

        @Override
        public void sendSystemMessage(Component component) {
            messages.add(component.getString());
        }

        @Override
        public boolean acceptsSuccess() {
            return true;
        }

        @Override
        public boolean acceptsFailure() {
            return true;
        }

        @Override
        public boolean shouldInformAdmins() {
            return false;
        }
    }
}
