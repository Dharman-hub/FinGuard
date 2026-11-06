
package ru.finguard.starter.config;

import org.junit.jupiter.api.Test;
import ru.finguard.core.decision.RuntimeThresholdPolicy;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

class ThresholdPolicyChangeServiceTest {
    private final RuntimeThresholdPolicy.Snapshot current =
            new RuntimeThresholdPolicy.Snapshot(
                    7L,
                    40,
                    80,
                    Optional.of(1.2f)
            );

    @Test
    void shouldCreateValidRuleThresholdPreview() {
        var preview = ThresholdPolicyChangeService.buildPreview(
                ThresholdPolicyChangeService.ChangeType.RULE_THRESHOLDS,
                current,
                7L,
                45,
                85,
                Optional.of(1.2f)
        );

        assertEquals(7L, preview.before().version());
        assertEquals(8L, preview.after().version());
        assertEquals(45, preview.after().reviewThreshold());
        assertEquals(85, preview.after().rejectThreshold());
    }

    @Test
    void shouldRejectInconsistentAuditVersion() {
        assertThrows(IllegalStateException.class, () ->
                ThresholdPolicyChangeService.buildPreview(
                        ThresholdPolicyChangeService.ChangeType.RULE_THRESHOLDS,
                        current,
                        6L,
                        45,
                        85,
                        Optional.of(1.2f)
                )
        );
    }

    @Test
    void shouldRejectInvalidRuleThresholds() {
        assertThrows(IllegalArgumentException.class, () ->
                ThresholdPolicyChangeService.buildPreview(
                        ThresholdPolicyChangeService.ChangeType.RULE_THRESHOLDS,
                        current,
                        7L,
                        90,
                        80,
                        Optional.of(1.2f)
                )
        );
    }

    @Test
    void shouldRejectUnchangedPolicy() {
        assertThrows(IllegalArgumentException.class, () ->
                ThresholdPolicyChangeService.buildPreview(
                        ThresholdPolicyChangeService.ChangeType.RULE_THRESHOLDS,
                        current,
                        7L,
                        40,
                        80,
                        Optional.of(1.2f)
                )
        );
    }

    @Test
    void shouldRejectNeuralAvailabilityChange() {
        assertThrows(IllegalArgumentException.class, () ->
                ThresholdPolicyChangeService.buildPreview(
                        ThresholdPolicyChangeService.ChangeType.NEURAL_THRESHOLD,
                        current,
                        7L,
                        40,
                        80,
                        Optional.empty()
                )
        );
    }

    @Test
    void shouldRejectMixedThresholdChanges() {
        assertThrows(IllegalArgumentException.class, () ->
                ThresholdPolicyChangeService.buildPreview(
                        ThresholdPolicyChangeService.ChangeType.NEURAL_THRESHOLD,
                        current,
                        7L,
                        45,
                        80,
                        Optional.of(1.4f)
                )
        );
    }

    @Test
    void shouldCreateValidNeuralThresholdPreview() {
        var preview = ThresholdPolicyChangeService.buildPreview(
                ThresholdPolicyChangeService.ChangeType.NEURAL_THRESHOLD,
                current,
                7L,
                40,
                80,
                Optional.of(1.4f)
        );

        assertEquals(8L, preview.after().version());
        assertEquals(1.4f,
                preview.after().neuralReviewThresholdLogit().orElseThrow());
    }

    @Test
    void shouldActivateWithoutPersistentAuditStore() {
        RuntimeThresholdPolicy policy =
                new RuntimeThresholdPolicy(0L, 40, 80, null);

        FinGuardSettingsManager manager =
                new FinGuardSettingsManager(policy, null);

        ThresholdPolicyChangeService service =
                new ThresholdPolicyChangeService(manager, null);

        var preview = service.previewRuleThresholds(45, 85);

        service.activate(preview, "admin-1");

        assertEquals(1L, manager.snapshotWithVersion().version());
        assertEquals(45, policy.snapshot().reviewThreshold());
        assertEquals(85, policy.snapshot().rejectThreshold());
        assertEquals(1, manager.auditHistory().size());
        assertEquals("admin-1", manager.auditHistory().get(0).actorId());
    }

    @Test
    void shouldRejectActivationOfOutdatedPreview() {
        RuntimeThresholdPolicy policy =
                new RuntimeThresholdPolicy(0L, 40, 80, null);

        FinGuardSettingsManager manager =
                new FinGuardSettingsManager(policy, null);

        ThresholdPolicyChangeService service =
                new ThresholdPolicyChangeService(manager, null);

        var preview = service.previewRuleThresholds(45, 85);

        manager.updateRuleThresholds(30, 90, 0L, "admin-2");

        assertThrows(
                IllegalStateException.class,
                () -> service.activate(preview, "admin-1")
        );

        assertEquals(1L, manager.snapshotWithVersion().version());
        assertEquals(30, policy.snapshot().reviewThreshold());
        assertEquals(90, policy.snapshot().rejectThreshold());
    }
}
