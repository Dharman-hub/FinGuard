package ru.finguard.core.decision;

import org.junit.jupiter.api.Test;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RuntimeThresholdPolicyTest {

    @Test
    void shouldCreateInitialPolicy() {
        RuntimeThresholdPolicy policy = new RuntimeThresholdPolicy(
                7,
                40,
                80,
                1.5145587f
        );

        var snapshot = policy.snapshot();

        assertEquals(7, snapshot.version());
        assertEquals(40, snapshot.reviewThreshold());
        assertEquals(80, snapshot.rejectThreshold());
        assertEquals(
                Optional.of(1.5145587f),
                snapshot.neuralReviewThresholdLogit()
        );
    }

    @Test
    void shouldActivateAllThresholdsTogether() {
        RuntimeThresholdPolicy policy = new RuntimeThresholdPolicy(
                7,
                40,
                80,
                1.5145587f
        );

        var previous = policy.snapshot();

        var updated = policy.activate(
                7,
                8,
                30,
                90,
                3.7777753f
        );

        assertEquals(8, updated.version());
        assertEquals(30, updated.reviewThreshold());
        assertEquals(90, updated.rejectThreshold());
        assertEquals(
                3.7777753f,
                updated.neuralReviewThresholdLogit().orElseThrow()
        );

        assertEquals(7, previous.version());
        assertEquals(40, previous.reviewThreshold());
        assertEquals(80, previous.rejectThreshold());
        assertEquals(
                1.5145587f,
                previous.neuralReviewThresholdLogit().orElseThrow()
        );

        assertEquals(updated, policy.snapshot());
    }

    @Test
    void shouldRejectOutdatedVersion() {
        RuntimeThresholdPolicy policy = new RuntimeThresholdPolicy(
                0,
                40,
                80,
                1.5145587f
        );

        policy.activate(0, 1, 30, 90, 2.5f);

        assertThrows(
                IllegalStateException.class,
                () -> policy.activate(0, 2, 20, 95, 3.0f)
        );

        assertEquals(1, policy.snapshot().version());
        assertEquals(30, policy.snapshot().reviewThreshold());
    }

    @Test
    void shouldRejectInvalidValuesWithoutChangingPolicy() {
        RuntimeThresholdPolicy policy = new RuntimeThresholdPolicy(
                0,
                40,
                80,
                1.5145587f
        );

        var original = policy.snapshot();

        assertThrows(
                IllegalArgumentException.class,
                () -> policy.activate(0, 1, 90, 80, 2.5f)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> policy.activate(0, 1, 30, 90, Float.NaN)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> policy.activate(0, 0, 30, 90, 2.5f)
        );

        assertEquals(original, policy.snapshot());
    }

    @Test
    void shouldSupportDisabledNeuralModel() {
        RuntimeThresholdPolicy policy = new RuntimeThresholdPolicy(
                0,
                40,
                80,
                null
        );

        policy.activate(0, 1, 30, 90, null);

        assertEquals(1, policy.snapshot().version());
        assertTrue(
                policy.snapshot().neuralReviewThresholdLogit().isEmpty()
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> policy.activate(1, 2, 30, 90, 2.5f)
        );

        assertEquals(1, policy.snapshot().version());
    }

    @Test
    void shouldAllowSynchronizationToNewerVersion() {
        RuntimeThresholdPolicy policy = new RuntimeThresholdPolicy(
                3,
                40,
                80,
                1.5145587f
        );

        policy.activate(3, 10, 20, 95, 5.048477f);

        assertEquals(10, policy.snapshot().version());
        assertEquals(20, policy.snapshot().reviewThreshold());
        assertEquals(95, policy.snapshot().rejectThreshold());
        assertEquals(
                5.048477f,
                policy.snapshot()
                        .neuralReviewThresholdLogit()
                        .orElseThrow()
        );
    }
}