import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import javax.imageio.ImageIO;

/**
 * Deterministically rasterizes the rectangle-only editable SVG palette without browser tooling.
 * Run from mod/: java tools/GenerateFlyTexture.java
 */
public final class GenerateFlyTexture {
    private static final int SIZE = 64;

    private GenerateFlyTexture() {}

    public static void main(String[] arguments) throws IOException {
        Path output = arguments.length == 0
                ? Path.of("src/client/resources/assets/neurocraft_fly/textures/entity/fly.png")
                : Path.of(arguments[0]);
        Files.createDirectories(output.toAbsolutePath().getParent());
        if (!ImageIO.write(render(), "png", output.toFile())) {
            throw new IOException("JDK PNG writer is unavailable");
        }
    }

    static BufferedImage render() {
        BufferedImage image = new BufferedImage(SIZE, SIZE, BufferedImage.TYPE_INT_ARGB);
        fill(image, "#6d351d", 0, 0, 64, 64);

        fill(image, "#b76a22", 0, 0, 28, 16);
        fill(image, "#3a2018", 0, 3, 28, 2);
        fill(image, "#3a2018", 0, 8, 28, 2);
        fill(image, "#3a2018", 0, 13, 28, 2);
        fill(image, "#e0a43d", 2, 1, 8, 1);

        fill(image, "#b7d8d8", 0.72, 32, 0, 32, 16);
        fill(image, "#dca84b", 32, 1, 32, 1);
        fill(image, "#dca84b", 35, 3, 27, 1);
        fill(image, "#dca84b", 39, 6, 23, 1);
        fill(image, "#dca84b", 43, 9, 18, 1);
        fill(image, "#dca84b", 47, 12, 13, 1);

        fill(image, "#74401f", 0, 18, 24, 20);
        fill(image, "#8f5528", 4, 20, 14, 10);
        fill(image, "#9b5b2a", 24, 18, 20, 20);
        fill(image, "#c48237", 28, 21, 12, 10);

        fill(image, "#711e25", 48, 0, 16, 18);
        fill(image, "#a82d35", 50, 2, 12, 14);
        fill(image, "#ef7b67", 51, 3, 3, 3);
        fill(image, "#d95450", 56, 7, 2, 2);
        fill(image, "#5b1720", 59, 11, 2, 2);

        fill(image, "#3b241b", 32, 18, 16, 28);
        fill(image, "#98602c", 35, 18, 2, 28);
        fill(image, "#98602c", 40, 18, 2, 28);
        fill(image, "#98602c", 45, 18, 2, 28);
        fill(image, "#241914", 48, 18, 16, 28);
        fill(image, "#a36a32", 52, 18, 2, 24);

        fill(image, "#52301e", 0, 46, 64, 18);
        fill(image, "#c2833c", 0, 46, 64, 2);
        return image;
    }

    private static void fill(
            BufferedImage image, String color, int x, int y, int width, int height) {
        fill(image, color, 1.0, x, y, width, height);
    }

    private static void fill(
            BufferedImage image,
            String color,
            double opacity,
            int x,
            int y,
            int width,
            int height) {
        int source = Color.decode(color).getRGB();
        for (int pixelY = y; pixelY < y + height; pixelY++) {
            for (int pixelX = x; pixelX < x + width; pixelX++) {
                image.setRGB(pixelX, pixelY, blend(source, image.getRGB(pixelX, pixelY), opacity));
            }
        }
    }

    private static int blend(int source, int destination, double opacity) {
        int red = blendChannel(source >>> 16, destination >>> 16, opacity);
        int green = blendChannel(source >>> 8, destination >>> 8, opacity);
        int blue = blendChannel(source, destination, opacity);
        return 0xff000000 | red << 16 | green << 8 | blue;
    }

    private static int blendChannel(int source, int destination, double opacity) {
        return (int) Math.round((source & 0xff) * opacity + (destination & 0xff) * (1.0 - opacity));
    }
}
