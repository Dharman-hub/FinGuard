package ru.finguard.starter.config;

import org.junit.jupiter.api.Test;
import ru.finguard.core.decision.RuntimeThresholdPolicy;

import java.util.ConcurrentModificationException;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

class FinGuardSettingsPairedPolicyTest {

    @Test
    void shouldApplyPersistedPolicyAndUpdateVersion() {
        RuntimeThresholdPolicy policy = createPolicy();

        FinGuardSettingsManager manager =
                new FinGuardSettingsManager(policy, null);

        manager.restrictToPairedState();

        var before = policy.snapshot();
        var after = nextPolicy(before, 2.25f);

        manager.applyPairedPolicy(before, after);

        assertEquals(after, policy.snapshot());
        assertEquals(1L, manager.snapshotWithVersion().version());

        assertThrows(
                IllegalStateException.class,
                () -> manager.updateNeuralDecisionThresholds(
                        3.0f,
                        null,
                        1L,
                        "admin"
                )
        );

        assertEquals(after, policy.snapshot());
    }

    @Test
    void shouldRejectStalePolicyWithoutChangingAnything() {
        RuntimeThresholdPolicy policy = createPolicy();

        FinGuardSettingsManager manager =
                new FinGuardSettingsManager(policy, null);

        manager.restrictToPairedState();

        var before = policy.snapshot();
        var first = nextPolicy(before, 2.25f);

        manager.applyPairedPolicy(before, first);

        var stale = nextPolicy(before, 3.0f);

        assertThrows(
                ConcurrentModificationException.class,
                () -> manager.applyPairedPolicy(before, stale)
        );

        assertEquals(first, policy.snapshot());
        assertEquals(1L, manager.snapshotWithVersion().version());
    }

    private RuntimeThresholdPolicy createPolicy() {
        return new RuntimeThresholdPolicy(
                0L,
                40,
                80,
                1.5145587f,
                null
        );
    }

    private RuntimeThresholdPolicy.Snapshot nextPolicy(
            RuntimeThresholdPolicy.Snapshot before,
            float reviewThreshold
    ) {
        return new RuntimeThresholdPolicy.Snapshot(
                before.version() + 1,
                before.reviewThreshold(),
                before.rejectThreshold(),
                Optional.of(reviewThreshold),
                Optional.empty(),
                before.neuralDataQualityPolicy()
        );
    }
}