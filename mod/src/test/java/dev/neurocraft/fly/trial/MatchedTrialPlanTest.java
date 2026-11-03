package dev.neurocraft.fly.trial;

import static org.junit.jupiter.api.Assertions.*;

import java.util.HashSet;
import org.junit.jupiter.api.Test;

final class MatchedTrialPlanTest {
    @Test void pairsEveryDeclaredSeedWithBothStimuliAndAllThreeLanes() {
        var cells = MatchedTrialPlan.trials();
        assertEquals(60, cells.size());
        assertEquals(10, new HashSet<>(MatchedTrialPlan.SEEDS).size());
        for (int pair = 0; pair < 10; pair++) {
            var six = cells.subList(pair * 6, pair * 6 + 6);
            assertEquals(6, six.stream().map(t -> t.stimulus() + "/" + t.lane()).distinct().count());
            for (var trial : six) {
                assertEquals(pair, trial.pair());
                assertEquals(MatchedTrialPlan.SEEDS.get(pair).longValue(), trial.bodySeed());
                assertEquals(trial.index(), cells.indexOf(trial));
            }
        }
        assertThrows(UnsupportedOperationException.class, () -> cells.clear());
        assertEquals(cells.size(), MatchedTrialPlan.document().getAsJsonArray("trials").size());
    }

    @Test void exactWorldTickBoundariesGiveTwentySixtyEightyWithoutAnExtraFrame() {
        int baseline = 0, approach = 0, recovery = 0;
        for (int tick = 0; tick < MatchedTrialPlan.TOTAL_TICKS; tick++) {
            int front = MatchedTrialPlan.frontPlane(tick);
            if (tick < 20) { assertEquals(-1, front); baseline++; }
            else if (tick < 80) { assertEquals(6 - (tick - 20) / 10, front); approach++; }
            else { assertEquals(-1, front); recovery++; }
        }
        assertEquals(20, baseline); assertEquals(60, approach); assertEquals(80, recovery);
        assertEquals(-1, MatchedTrialPlan.frontPlane(-1));
        assertEquals(-1, MatchedTrialPlan.frontPlane(160));
        assertEquals(-1, MatchedTrialPlan.frontPlane(Long.MAX_VALUE));
    }
}
