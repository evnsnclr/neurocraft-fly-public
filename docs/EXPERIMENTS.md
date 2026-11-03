# Play and compare

Start the playground from source with the launch command in the
[installation guide](INSTALL.md) (private Companion testers start it from
[Companion](COMPANION.md) instead), then join with your normal Fabric client. Use one fly in the fresh showcase world. The compact HUD
keeps the active condition and body authority visible; H opens full model details without pausing the world. Close
details, chat and other menus before interacting. On a small window, a reduced
state card may move the remaining samples into details.

## Your first interaction

1. Wait for **Live · Original weights · Readout control**.
2. Open chat, run `/give @s minecraft:brush 1`, and close chat.
3. Hold the brush, aim at the fly within native interaction reach, and right-click
   once. Stay a living non-spectator player with a clear line of sight.
4. Look for **Groom** in the body state. A matched applied sample can name
   **Brush contact** and show antennal-motor modeled activity. If it says
   **Awaiting match**, the HUD is not supplying matching evidence at that moment.
5. Let the response finish and wait through its cooldown before another attempt.

That action is a declared touch input. The body animation is scripted; the
population values are model quantities. A missed crosshair remains a missed
input, rather than a failed neural response.

## Inspect a recent event

After an interaction, press **R** to open the event replay. Use **Previous** and
**Next** (or the arrow keys) to inspect collected HUD samples, **Play samples**
(or Space) to advance through them, and **Load latest** to refresh the snapshot.
Press R again or Escape to close it. The key can be changed in Minecraft's
Controls settings.

This is a read-only history of locally collected display samples. The world
continues running while it is open. Playback does not resend inputs, rerun the
model, restore a world state or establish biological causation. Missing or
unmatched samples remain unavailable. Close the replay before performing the
next interaction.

## Six sources, six declared responses

| Interaction | Preparation | Expected program when its gates qualify |
| --- | --- | --- |
| Walk toward the fly | Stay grounded and approach its front or either side. The same ray needs consecutive decreasing Player distances within six blocks. | Escape jump |
| Bring a Mob close | Put a cow in its front/side ray and allow relative distance to decrease; a cow merely visible to your camera is insufficient. | Bank |
| Brush | Hold a brush and right-click the fly within reach and visibility. | Groom |
| Attack attempt | Aim and left-click once within reach; leave room for the fly to move away from a visible living non-spectator player. | Flee |
| Offer sweet food | Drop sugar near its forward touchdown area. Food must initially be visible within three blocks and ±60° forward; feeding needs support and food within 0.75 blocks. | Land → Feed, or Feed if already grounded |
| Change illumination | Establish a measured bright baseline, then change the light reaching the fly. A torch in daylight may change nothing. | Hover |

Player/Mob categories come from Minecraft. Approach depends on relative motion,
including the fly's own movement. Attack is an attempt with a bristle tactile
proxy, not a health or pain model. Food does not add seeking or automatic
alignment, and the animation does not consume the item.

### Controlled light preparation

In the fresh central showcase arena, these commands establish an opaque canopy
and noncolliding bright blocks. They change the scene; record them as fixture
preparation and use the same fixture in each comparison lane:

```text
/fill -24 74 -24 24 74 24 minecraft:black_concrete
/fill -24 65 -24 24 70 24 minecraft:light[level=15] replace minecraft:air
```

Wait for preparation responses and cooldowns to finish. Inspect the measured
light in H details, then close the screen. Present one decrease:

```text
/fill -24 65 -24 24 70 24 minecraft:air replace minecraft:light
```

The light reaching the fly must actually change. Restore the bright fill only
after the response and cooldown; restoration can itself cause another event.

## Repeat without changing the question

Declare the intended lane, source and number of attempts before starting. Record
the actual outcome, including missed inputs, refused programs and unavailable
HUD matches. Holds are 24 ticks for escape, 40 for Bank/Flee/Hover and 60 for
Groom/Feed, followed by 120 ticks of cooldown. At sustained 20 TPS those are
1.2, 2, 3 and 6 seconds respectively; Land adds its own descent phase.

Wait for the completed hold, cooldown and fresh model observations. Clear old
food/cows between attempts. Avoid resetting the fly, moving actors, changing
lanes or issuing operator programs after stimulus onset through the response.
If a reset is needed, do it between trials, record it, and start a new attempt.
Time spent waiting alone does not prove readiness while the controller is stalled.

## Compare the conditions

Request **Original weights**, **Shuffled weights** and **No input** one at a time,
using the in-game commands below (private Companion testers use its condition
buttons, which issue the same requests). Complete that condition's attempts before
switching, then select **Original weights** again to restore the original condition.
Use the same declared preparation and repetitions each time.

The in-game commands:

```text
/neurocraftfly control original
/neurocraftfly control shuffled_weight
/neurocraftfly control no_input
/neurocraftfly control original
```

After each switch, confirm the condition on the HUD and wait for fresh samples
and **Readout control**. Switching starts a new neural epoch but does not clear the
world. Recreate the stimulus; finish by showing original weights active again.

A shuffle can retain responses. No-input can still execute fallback Cruise.
An undelivered stimulus cannot establish suppression. Compare qualified inputs
separately from the number of scheduled attempts, retaining both denominators.

| Attempt | Source | Lane | Input actually delivered? | Matched applied program | Complete hold? | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Brush | Original | Record observation | Record or unavailable | Record or unknown | Misses and interruptions belong here |

One recording is a demonstration, not a biological result. The
[fixed30 diagnostic runner](../scripts/run_interaction_trials.py) uses scripted
preparation and synthetic native input. Its measurements are distinct from a
human-operated recording. The separate fixed60 ENTITY/BLOCK supplement is not
a replacement for six delivered interaction sources.

## Optional operator body tour

`/neurocraftfly program hover` requests operator authority through a neural PAUSE.
Wait for acknowledgement. Operator-selected programs are useful for clear Walk,
Feed or Groom close-ups, but are not additional neural responses. The available
program names are `hover`, `land`, `walk`, `feed`, `groom`, `escape_jump`,
`cruise`, `bank`, `flee` and `approach`. Issue them one at a time with
`/neurocraftfly program <name>`. Grounded support is required for Walk and Feed; Feed also needs nearby food,
and Groom requires being outside fluid. Flee and Approach need a visible living
non-spectator Player target within twelve blocks. Conditions may refuse a
program and leave Hover: inspect `/neurocraftfly program status` for the reason.
Finish with `/neurocraftfly program release` and wait for advancing READOUT
samples before interpreting another neural interaction.

## Read and record the HUD

**Body now** is current synchronized body state. **Applied sample** is an earlier
exactly matched command; **Model sample** can have a different sequence. Their
read ages are not reaction latency. **Input** names the selected event attached
to that applied sample, not every event currently present in the world.

Record the entire game window and narration with your recorder of choice. Keep
the original continuous take, source revision, local run directory and attempt
notes. OBS frame rate and game TPS are different clocks. Label edits, operator
tour segments and scene manipulations. Introduce H details once so the engineered
mapping and historical measurement limits accompany the recording.
