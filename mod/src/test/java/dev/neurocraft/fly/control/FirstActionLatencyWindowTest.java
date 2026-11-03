package dev.neurocraft.fly.control;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Optional;
import org.junit.jupiter.api.Test;

final class FirstActionLatencyWindowTest {
    @Test
    void arrivalIsMeasuredAgainstItsOwnObservationOfferAndSend() {
        FirstActionLatencyWindow window = new FirstActionLatencyWindow(4);
        window.recordSend(0, 1_000_000L, 3_000_000L);
        window.recordSend(1, 5_000_000L, 6_000_000L);

        FirstActionLatencyWindow.Sample second =
                window.completeArrival(1, 20_000_000L).orElseThrow();
        assertEquals(1L, second.observationSequence());
        assertEquals(15_000_000L, second.offerToActionNs());
        assertEquals(14_000_000L, second.sendToActionNs());

        FirstActionLatencyWindow.Sample first =
                window.completeArrival(0, 21_000_000L).orElseThrow();
        assertEquals(20_000_000L, first.offerToActionNs());
        assertEquals(18_000_000L, first.sendToActionNs());
        assertEquals(2, window.remaining());
    }

    @Test
    void everyObservationIsMeasuredAtMostOnce() {
        FirstActionLatencyWindow window = new FirstActionLatencyWindow(4);
        window.recordSend(7, 0L, 0L);

        assertTrue(window.completeArrival(7, 1_000_000L).isPresent());
        assertEquals(Optional.empty(), window.completeArrival(7, 2_000_000L));
        assertEquals(3, window.remaining());
    }

    @Test
    void theBudgetIsSpentNeverRefilledAndSilencesRecordingToo() {
        FirstActionLatencyWindow window = new FirstActionLatencyWindow(1);
        window.recordSend(0, 0L, 0L);
        assertTrue(window.completeArrival(0, 1_000_000L).isPresent());
        assertEquals(0, window.remaining());

        window.recordSend(1, 0L, 0L);
        assertEquals(Optional.empty(), window.completeArrival(1, 1_000_000L));
        assertEquals(0, window.remaining());
    }

    @Test
    void anObservationOlderThanTheWindowIsSkippedRatherThanMismeasured() {
        FirstActionLatencyWindow window = new FirstActionLatencyWindow(8);
        for (long sequence = 0; sequence < 5; sequence++) {
            window.recordSend(sequence, sequence, sequence);
        }

        assertEquals(Optional.empty(), window.completeArrival(0, 10L));
        assertTrue(window.completeArrival(1, 10L).isPresent());
        assertEquals(7, window.remaining());
    }

    @Test
    void clearingDropsUnmatchedOffersWithoutRestoringTheBudget() {
        FirstActionLatencyWindow window = new FirstActionLatencyWindow(2);
        window.recordSend(0, 0L, 0L);
        assertTrue(window.completeArrival(0, 1L).isPresent());
        window.recordSend(1, 0L, 0L);

        window.clear();

        assertEquals(Optional.empty(), window.completeArrival(1, 1L));
        assertEquals(1, window.remaining());
    }

    @Test
    void aNegativeSampleBudgetIsRefused() {
        assertThrows(IllegalArgumentException.class, () -> new FirstActionLatencyWindow(-1));
    }
}
