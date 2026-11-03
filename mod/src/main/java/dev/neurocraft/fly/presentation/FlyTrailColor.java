package dev.neurocraft.fly.presentation;

/** Pure trail-color derivation from a decoder hash; no engine types, no clock, no RNG. */
public final class FlyTrailColor {
    /** Fixed neutral color used whenever no bridge configuration is loaded. */
    public static final int FALLBACK_RGB = 0xB4B4B4;
    /** Per-channel floor keeping every derived profile color visible in dark arenas. */
    public static final int MIN_CHANNEL = 0x40;

    private FlyTrailColor() {}

    /** Maps the first six hex nibbles of a lowercase SHA-256 decoder hash to packed RGB. */
    public static int fromDecoderHash(String decoderHash) {
        return channel(decoderHash, 0) << 16
                | channel(decoderHash, 2) << 8
                | channel(decoderHash, 4);
    }

    private static int channel(String decoderHash, int offset) {
        int raw = Integer.parseInt(decoderHash.substring(offset, offset + 2), 16);
        return MIN_CHANNEL + raw * (0xFF - MIN_CHANNEL) / 0xFF;
    }
}
