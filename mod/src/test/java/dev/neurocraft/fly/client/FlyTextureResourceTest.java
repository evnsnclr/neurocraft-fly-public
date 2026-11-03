package dev.neurocraft.fly.client;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.util.HashSet;
import java.util.Set;
import javax.imageio.ImageIO;
import org.junit.jupiter.api.Test;

final class FlyTextureResourceTest {
    @Test
    void committedAtlasIsFullSizeOpaqueAndContainsEveryVisiblePaletteRegion() throws Exception {
        Path texture = Path.of(
                "src/client/resources/assets/neurocraft_fly/textures/entity/fly.png");
        BufferedImage image = ImageIO.read(texture.toFile());

        assertNotNull(image);
        assertEquals(64, image.getWidth());
        assertEquals(64, image.getHeight());
        Set<Integer> colors = new HashSet<>();
        int opaquePixels = 0;
        for (int y = 0; y < image.getHeight(); y++) {
            for (int x = 0; x < image.getWidth(); x++) {
                int argb = image.getRGB(x, y);
                colors.add(argb);
                if ((argb >>> 24) == 255) {
                    opaquePixels++;
                }
            }
        }

        assertEquals(64 * 64, opaquePixels, "reserve UVs must fail visibly, not transparently");
        assertTrue(colors.size() >= 16, "atlas collapsed to a thumbnail or placeholder palette");
        assertEquals(0xffb76a22, image.getRGB(20, 1), "abdomen field");
        assertEquals(0xffa82d35, image.getRGB(58, 5), "compound-eye field");
        assertEquals(0xff52301e, image.getRGB(10, 60), "opaque reserve field");
        assertTrue(image.getRGB(40, 5) != 0xffffffff, "wing field must not be blank white");
    }
}
