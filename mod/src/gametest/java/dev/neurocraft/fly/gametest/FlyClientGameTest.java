package dev.neurocraft.fly.gametest;

import com.mojang.blaze3d.platform.NativeImage;
import dev.neurocraft.fly.NeuroCraftFly;
import dev.neurocraft.fly.client.FlyAnimation;
import dev.neurocraft.fly.entity.FlyEntity;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.HashSet;
import java.util.Locale;
import java.util.Set;
import net.fabricmc.fabric.api.client.gametest.v1.FabricClientGameTest;
import net.fabricmc.fabric.api.client.gametest.v1.context.ClientGameTestContext;
import net.minecraft.core.BlockPos;
import net.minecraft.resources.ResourceLocation;
import net.minecraft.world.entity.Entity;
import net.minecraft.world.entity.EntitySpawnReason;
import net.minecraft.world.level.block.Blocks;

/**
 * Automated resource and renderer observation; this is not a manual gameplay demonstration.
 *
 * <p>The receipt written here carries the ADR 0016 presentation-gain marker
 * ({@code bank_is_display_scaled} and {@code presentation_visual_gain}) so that the caveat travels
 * with the artifact: the screenshot beside it is drawn through a declared display scaling of the
 * yaw channel, and a rendered bank must never be reported as the fly steering hard.
 */
public final class FlyClientGameTest implements FabricClientGameTest {
    private static final ResourceLocation TEXTURE = ResourceLocation.fromNamespaceAndPath(
            NeuroCraftFly.MOD_ID, "textures/entity/fly.png");

    @Override
    public void runTest(ClientGameTestContext context) {
        context.runOnClient(client -> {
            var resource = client.getResourceManager()
                    .getResource(TEXTURE)
                    .orElseThrow(() -> new AssertionError("original fly texture is missing"));
            try (var stream = resource.open(); NativeImage image = NativeImage.read(stream)) {
                if (image.getWidth() != 64 || image.getHeight() != 64) {
                    throw new AssertionError("fly texture must be 64x64 RGBA");
                }
            } catch (IOException exception) {
                throw new AssertionError("fly texture could not be decoded", exception);
            }
        });

        Path screenshotOutput = requiredPath("neurocraft.fly.clientRendererScreenshot");
        Path receiptOutput = requiredPath("neurocraft.fly.clientRendererReceipt");
        try (var singleplayer = context.worldBuilder().create()) {
            singleplayer.getServer().runCommand("gamemode spectator @p");
            singleplayer.getServer().runCommand("tp @p 0 80 0 0 0");
            singleplayer.getServer().runCommand("time set noon");
            singleplayer.getServer().runCommand("weather clear");
            singleplayer.getServer().runOnServer(server -> {
                var level = server.overworld();
                for (int x = -7; x <= 7; x++) {
                    for (int y = 76; y <= 87; y++) {
                        level.setBlockAndUpdate(
                                new BlockPos(x, y, 7),
                                Blocks.WHITE_CONCRETE.defaultBlockState());
                    }
                }
            });
            singleplayer.getClientWorld().waitForChunksRender();
            context.waitTicks(20);
            Path background = context.takeScreenshot("neurocraft-fly-background");

            singleplayer.getServer().runOnServer(server -> {
                FlyEntity fly =
                        NeuroCraftFly.FLY.create(server.overworld(), EntitySpawnReason.COMMAND);
                if (fly == null) {
                    throw new AssertionError("could not construct renderer-observation fly");
                }
                fly.resetForFixture(0.0, 81.5, 4.0, 180.0F);
                if (!server.overworld().addFreshEntity(fly)) {
                    throw new AssertionError("could not spawn renderer-observation fly");
                }
            });
            context.waitFor(client -> client.level != null
                    && containsFly(client.level.entitiesForRendering()));
            context.waitTicks(10);
            try {
                Path rendered = context.takeScreenshot("neurocraft-fly-rendered");
                PixelDifference difference = compareCentralFrame(background, rendered);

                if (difference.changedPixels() < 40
                        || difference.warmChangedPixels() < 20
                        || difference.warmWidth() < 4
                        || difference.warmHeight() < 4
                        || difference.uniqueWarmColors() < 3
                        || !difference.warmOverlapsFrameCenter()) {
                    throw new AssertionError(
                            "custom fly renderer did not produce a visible warm-palette fly delta: "
                                    + difference);
                }
                Files.createDirectories(screenshotOutput.getParent());
                Files.copy(rendered, screenshotOutput, StandardCopyOption.REPLACE_EXISTING);
                Files.createDirectories(receiptOutput.getParent());
                Files.writeString(
                        receiptOutput,
                        difference.toJson(),
                        StandardCharsets.UTF_8);
            } catch (IOException exception) {
                throw new AssertionError("could not inspect or retain renderer screenshot", exception);
            }
        }
    }

    private static boolean containsFly(Iterable<Entity> entities) {
        for (Entity entity : entities) {
            if (entity instanceof FlyEntity) {
                return true;
            }
        }
        return false;
    }

    private static Path requiredPath(String property) {
        String configured = System.getProperty(property);
        if (configured == null || configured.isBlank()) {
            throw new AssertionError("missing client GameTest output property " + property);
        }
        return Path.of(configured).toAbsolutePath().normalize();
    }

    private static PixelDifference compareCentralFrame(Path background, Path rendered)
            throws IOException {
        try (NativeImage before = NativeImage.read(Files.newInputStream(background));
                NativeImage after = NativeImage.read(Files.newInputStream(rendered))) {
            if (before.getWidth() != after.getWidth() || before.getHeight() != after.getHeight()) {
                throw new AssertionError("renderer screenshots changed dimensions");
            }
            int width = after.getWidth();
            int height = after.getHeight();
            int cropMinX = width / 4;
            int cropMaxX = width * 3 / 4;
            int cropMinY = height / 5;
            int cropMaxY = height * 4 / 5;
            int minX = width;
            int minY = height;
            int maxX = -1;
            int maxY = -1;
            int warmMinX = width;
            int warmMinY = height;
            int warmMaxX = -1;
            int warmMaxY = -1;
            int changed = 0;
            int warmChanged = 0;
            Set<Integer> colors = new HashSet<>();
            Set<Integer> warmColors = new HashSet<>();
            for (int y = cropMinY; y < cropMaxY; y++) {
                for (int x = cropMinX; x < cropMaxX; x++) {
                    int beforePixel = before.getPixel(x, y);
                    int afterPixel = after.getPixel(x, y);
                    if (colorDistance(beforePixel, afterPixel) <= 24) {
                        continue;
                    }
                    changed++;
                    minX = Math.min(minX, x);
                    minY = Math.min(minY, y);
                    maxX = Math.max(maxX, x);
                    maxY = Math.max(maxY, y);
                    colors.add(afterPixel);
                    if (isWarmFlyPalette(afterPixel)) {
                        warmChanged++;
                        warmMinX = Math.min(warmMinX, x);
                        warmMinY = Math.min(warmMinY, y);
                        warmMaxX = Math.max(warmMaxX, x);
                        warmMaxY = Math.max(warmMaxY, y);
                        warmColors.add(afterPixel);
                    }
                }
            }
            return new PixelDifference(
                    width,
                    height,
                    changed,
                    minX,
                    minY,
                    maxX,
                    maxY,
                    colors.size(),
                    warmChanged,
                    warmMinX,
                    warmMinY,
                    warmMaxX,
                    warmMaxY,
                    warmColors.size());
        }
    }

    private static boolean isWarmFlyPalette(int argb) {
        int red = argb >>> 16 & 0xff;
        int green = argb >>> 8 & 0xff;
        int blue = argb & 0xff;
        return red >= 55 && red - green >= 10 && green - blue >= 5;
    }

    private static int colorDistance(int first, int second) {
        int distance = 0;
        for (int shift : new int[] {0, 8, 16}) {
            distance += Math.abs((first >>> shift & 0xff) - (second >>> shift & 0xff));
        }
        return distance;
    }

    private record PixelDifference(
            int frameWidth,
            int frameHeight,
            int changedPixels,
            int minX,
            int minY,
            int maxX,
            int maxY,
            int uniqueRenderedColors,
            int warmChangedPixels,
            int warmMinX,
            int warmMinY,
            int warmMaxX,
            int warmMaxY,
            int uniqueWarmColors) {
        int width() {
            return maxX < minX ? 0 : maxX - minX + 1;
        }

        int height() {
            return maxY < minY ? 0 : maxY - minY + 1;
        }

        int warmWidth() {
            return warmMaxX < warmMinX ? 0 : warmMaxX - warmMinX + 1;
        }

        int warmHeight() {
            return warmMaxY < warmMinY ? 0 : warmMaxY - warmMinY + 1;
        }

        boolean warmOverlapsFrameCenter() {
            return warmMaxX >= frameWidth * 2 / 5
                    && warmMinX <= frameWidth * 3 / 5
                    && warmMaxY >= frameHeight * 2 / 5
                    && warmMinY <= frameHeight * 3 / 5;
        }

        /**
         * The ADR 0016 presentation-gain marker.
         *
         * <p>Every number is read from {@link FlyAnimation}'s declared presentation constants
         * rather than restated here, so the marker cannot drift away from the mapping it
         * describes. The gain is the reciprocal of the reference amplitude because the yaw
         * channel is divided by that reference and clamped to {@code [-1, +1]} before it reaches
         * the bank; a command at the reference draws full bank and anything larger draws the
         * same bank.
         */
        private static String presentationVisualGainJson() {
            float reference = FlyAnimation.YAW_TORQUE_VISUAL_REFERENCE;
            return String.format(
                    Locale.ROOT,
                    "  \"presentation_visual_gain\": {\n"
                            + "    \"adr\": \"0016\",\n"
                            + "    \"channel\": \"yaw_torque\",\n"
                            + "    \"rendered_as\": \"body_bank_degrees\",\n"
                            + "    \"reference_amplitude\": %.6f,\n"
                            + "    \"max_bank_degrees\": %.1f,\n"
                            + "    \"gain\": %.6f,\n"
                            + "    \"caveat\": \"Render-only presentation. The drawn bank is"
                            + " the yaw torque divided by the declared reference amplitude and"
                            + " clamped, so a fully banked fly is executing at least about"
                            + " %.1f%% of its yaw authority and any larger command draws the"
                            + " same bank. A rendered bank must never be reported as the fly"
                            + " steering hard: magnitude comes from the trace, not from this"
                            + " image.\"\n"
                            + "  },\n",
                    reference,
                    FlyAnimation.MAX_BANK_DEGREES,
                    1.0F / reference,
                    reference * 100.0F);
        }

        String toJson() {
            return String.format(
                    Locale.ROOT,
                    "{\n"
                            + "  \"schema\": \"neurocraft.renderer_observation.v2\",\n"
                            + "  \"automated\": true,\n"
                            + "  \"manual_gameplay_observation\": false,\n"
                            + "  \"bank_is_display_scaled\": true,\n"
                            + "%s"
                            + "  \"frame_width\": %d,\n"
                            + "  \"frame_height\": %d,\n"
                            + "  \"changed_central_pixels\": %d,\n"
                            + "  \"difference_bounds\": [%d, %d, %d, %d],\n"
                            + "  \"difference_width\": %d,\n"
                            + "  \"difference_height\": %d,\n"
                            + "  \"unique_rendered_colors\": %d,\n"
                            + "  \"warm_fly_pixels\": %d,\n"
                            + "  \"warm_fly_bounds\": [%d, %d, %d, %d],\n"
                            + "  \"warm_fly_width\": %d,\n"
                            + "  \"warm_fly_height\": %d,\n"
                            + "  \"unique_warm_fly_colors\": %d\n"
                            + "}\n",
                    presentationVisualGainJson(),
                    frameWidth,
                    frameHeight,
                    changedPixels,
                    minX,
                    minY,
                    maxX,
                    maxY,
                    width(),
                    height(),
                    uniqueRenderedColors,
                    warmChangedPixels,
                    warmMinX,
                    warmMinY,
                    warmMaxX,
                    warmMaxY,
                    warmWidth(),
                    warmHeight(),
                    uniqueWarmColors);
        }
    }
}
