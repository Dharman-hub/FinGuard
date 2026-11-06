package ru.finguard.starter.config;

import org.junit.jupiter.api.Test;
import ru.finguard.core.decision.RuleDecisionThresholds;
import ru.finguard.core.decision.RuntimeThresholdPolicy;
import ru.finguard.core.neural.NeuralReviewThreshold;

import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FinGuardSettingsManagerTest {

    @Test
    void shouldExposeInitialSettings() {
        FinGuardSettingsManager manager = createManager();

        var snapshot = manager.snapshot();

        assertEquals(40, snapshot.reviewThreshold());
        assertEquals(80, snapshot.rejectThreshold());
        assertEquals(
                Optional.of(1.5145587f),
                snapshot.neuralReviewThresholdLogit()
        );
    }

    @Test
    void shouldUpdateRuleThresholds() {
        FinGuardSettingsManager manager = createManager();

        manager.updateRuleThresholds(30, 90);

        var snapshot = manager.snapshot();

        assertEquals(30, snapshot.reviewThreshold());
        assertEquals(90, snapshot.rejectThreshold());
    }

    @Test
    void shouldSwitchBetweenNeuralModes() {
        FinGuardSettingsManager manager = createManager();

        manager.useNeuralMode(NeuralReviewMode.LOW_ALERTS);

        assertEquals(
                5.048477f,
                manager.snapshot()
                        .neuralReviewThresholdLogit()
                        .orElseThrow()
        );

        manager.useNeuralMode(NeuralReviewMode.BALANCED);

        assertEquals(
                3.7777753f,
                manager.snapshot()
                        .neuralReviewThresholdLogit()
                        .orElseThrow()
        );

        manager.useNeuralMode(NeuralReviewMode.HIGH_SENSITIVITY);

        assertEquals(
                1.5145587f,
                manager.snapshot()
                        .neuralReviewThresholdLogit()
                        .orElseThrow()
        );
    }

    @Test
    void shouldApplyCustomThreshold() {
        FinGuardSettingsManager manager = createManager();

        manager.setCustomNeuralThreshold(2.5f);

        assertEquals(
                2.5f,
                manager.snapshot()
                        .neuralReviewThresholdLogit()
                        .orElseThrow()
        );
    }

    @Test
    void shouldRejectInvalidChanges() {
        FinGuardSettingsManager manager = createManager();

        assertThrows(
                IllegalArgumentException.class,
                () -> manager.updateRuleThresholds(90, 80)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> manager.setCustomNeuralThreshold(Float.NaN)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> manager.useNeuralMode(NeuralReviewMode.CUSTOM)
        );

        var snapshot = manager.snapshot();

        assertEquals(40, snapshot.reviewThreshold());
        assertEquals(80, snapshot.rejectThreshold());
        assertEquals(
                1.5145587f,
                snapshot.neuralReviewThresholdLogit().orElseThrow()
        );
    }

    @Test
    void shouldWorkWhenNeuralIsDisabled() {
        FinGuardSettingsManager manager = new FinGuardSettingsManager(
                new RuleDecisionThresholds(40, 80),
                null
        );

        manager.updateRuleThresholds(30, 90);

        assertEquals(30, manager.snapshot().reviewThreshold());
        assertEquals(90, manager.snapshot().rejectThreshold());
        assertTrue(
                manager.snapshot().neuralReviewThresholdLogit().isEmpty()
        );

        assertThrows(
                IllegalStateException.class,
                () -> manager.useNeuralMode(NeuralReviewMode.BALANCED)
        );
    }

    @Test
    void shouldRecordVersionedRuleChange() {
        FinGuardSettingsManager manager = createManager();

        var initial = manager.snapshotWithVersion();

        assertEquals(0, initial.version());

        manager.updateRuleThresholds(
                30,
                90,
                initial.version(),
                "bank-admin-1"
        );

        var current = manager.snapshotWithVersion();

        assertEquals(1, current.version());
        assertEquals(30, current.settings().reviewThreshold());
        assertEquals(90, current.settings().rejectThreshold());

        var entry = manager.auditHistory().get(0);

        assertEquals(1, entry.version());
        assertEquals("bank-admin-1", entry.actorId());
        assertEquals(
                FinGuardSettingsManager.ChangeType.RULE_THRESHOLDS,
                entry.type()
        );
        assertEquals(initial.settings(), entry.before());
        assertEquals(current.settings(), entry.after());
    }

    @Test
    void shouldRejectStaleSettingsVersion() {
        FinGuardSettingsManager manager = createManager();

        manager.updateRuleThresholds(30, 90, 0, "admin-1");

        assertThrows(
                IllegalStateException.class,
                () -> manager.updateRuleThresholds(
                        20,
                        95,
                        0,
                        "admin-2"
                )
        );

        assertEquals(1, manager.snapshotWithVersion().version());
        assertEquals(30, manager.snapshot().reviewThreshold());
        assertEquals(90, manager.snapshot().rejectThreshold());
        assertEquals(1, manager.auditHistory().size());
    }

    @Test
    void shouldAuditNeuralChanges() {
        FinGuardSettingsManager manager = createManager();

        manager.useNeuralMode(
                NeuralReviewMode.BALANCED,
                0,
                "admin-1"
        );

        manager.setCustomNeuralThreshold(
                2.5f,
                1,
                "admin-2"
        );

        assertEquals(2, manager.snapshotWithVersion().version());
        assertEquals(2, manager.auditHistory().size());

        var latest = manager.auditHistory().get(0);

        assertEquals(
                FinGuardSettingsManager.ChangeType.NEURAL_CUSTOM,
                latest.type()
        );
        assertEquals("admin-2", latest.actorId());
        assertEquals(
                3.7777753f,
                latest.before().neuralReviewThresholdLogit().orElseThrow()
        );
        assertEquals(
                2.5f,
                latest.after().neuralReviewThresholdLogit().orElseThrow()
        );
    }

    @Test
    void shouldNotAuditInvalidChanges() {
        FinGuardSettingsManager manager = createManager();

        assertThrows(
                IllegalArgumentException.class,
                () -> manager.updateRuleThresholds(
                        90,
                        80,
                        0,
                        "admin-1"
                )
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> manager.useNeuralMode(
                        NeuralReviewMode.BALANCED,
                        0,
                        " "
                )
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> manager.setCustomNeuralThreshold(
                        Float.NaN,
                        0,
                        "admin-1"
                )
        );

        assertEquals(0, manager.snapshotWithVersion().version());
        assertTrue(manager.auditHistory().isEmpty());
    }

    @Test
    void shouldKeepOnlyLatestThousandAuditEntries() {
        FinGuardSettingsManager manager = createManager();

        for (int i = 0; i < 1005; i++) {
            manager.updateRuleThresholds(40, 80);
        }

        assertEquals(1005, manager.snapshotWithVersion().version());
        assertEquals(1000, manager.auditHistory().size());
        assertEquals(1005, manager.auditHistory().get(0).version());
        assertEquals(6, manager.auditHistory().get(999).version());
    }

    @Test
    void shouldRollbackLastRuleChange() {
        FinGuardSettingsManager manager = createManager();

        manager.updateRuleThresholds(
                30,
                90,
                0,
                "admin-1"
        );

        manager.rollbackLastChange(1, "admin-2");

        var current = manager.snapshotWithVersion();

        assertEquals(2, current.version());
        assertEquals(40, current.settings().reviewThreshold());
        assertEquals(80, current.settings().rejectThreshold());

        var rollback = manager.auditHistory().get(0);

        assertEquals(
                FinGuardSettingsManager.ChangeType.ROLLBACK,
                rollback.type()
        );
        assertEquals("admin-2", rollback.actorId());
        assertEquals(30, rollback.before().reviewThreshold());
        assertEquals(40, rollback.after().reviewThreshold());
    }

    @Test
    void shouldRollbackLastNeuralChange() {
        FinGuardSettingsManager manager = createManager();

        manager.useNeuralMode(
                NeuralReviewMode.BALANCED,
                0,
                "admin-1"
        );

        manager.rollbackLastChange(1, "admin-2");

        assertEquals(
                1.5145587f,
                manager.snapshot()
                        .neuralReviewThresholdLogit()
                        .orElseThrow()
        );

        assertEquals(2, manager.snapshotWithVersion().version());
    }

    @Test
    void shouldRejectRollbackWithoutHistory() {
        FinGuardSettingsManager manager = createManager();

        assertThrows(
                IllegalStateException.class,
                () -> manager.rollbackLastChange(0, "admin-1")
        );

        assertEquals(0, manager.snapshotWithVersion().version());
        assertTrue(manager.auditHistory().isEmpty());
    }

    @Test
    void shouldRejectRollbackWithOutdatedVersion() {
        FinGuardSettingsManager manager = createManager();

        manager.updateRuleThresholds(
                30,
                90,
                0,
                "admin-1"
        );

        assertThrows(
                IllegalStateException.class,
                () -> manager.rollbackLastChange(0, "admin-2")
        );

        assertEquals(1, manager.snapshotWithVersion().version());
        assertEquals(30, manager.snapshot().reviewThreshold());
    }

    @Test
    void shouldRollbackToEarlierVersionWithoutStorage() {
        FinGuardSettingsManager manager = createManager();

        manager.updateRuleThresholds(30, 90, 0, "admin-1");
        manager.updateRuleThresholds(20, 95, 1, "admin-1");
        manager.useNeuralMode(NeuralReviewMode.BALANCED, 2, "admin-1");

        manager.rollbackToVersion(1, 3, "admin-2");

        var current = manager.snapshotWithVersion();

        assertEquals(4, current.version());
        assertEquals(30, current.settings().reviewThreshold());
        assertEquals(90, current.settings().rejectThreshold());

        assertEquals(
                1.5145587f,
                current.settings()
                        .neuralReviewThresholdLogit()
                        .orElseThrow()
        );

        assertEquals(
                FinGuardSettingsManager.ChangeType.ROLLBACK,
                manager.auditHistory().get(0).type()
        );
    }

    @Test
    void shouldRejectUnavailableRollbackVersion() {
        FinGuardSettingsManager manager = createManager();

        manager.updateRuleThresholds(30, 90, 0, "admin-1");

        assertThrows(
                IllegalArgumentException.class,
                () -> manager.rollbackToVersion(2, 1, "admin-2")
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> manager.rollbackToVersion(-1, 1, "admin-2")
        );

        assertThrows(
                IllegalStateException.class,
                () -> manager.rollbackToVersion(0, 0, "admin-2")
        );

        assertEquals(1, manager.snapshotWithVersion().version());
        assertEquals(30, manager.snapshot().reviewThreshold());
    }

    private FinGuardSettingsManager createManager() {
        return new FinGuardSettingsManager(
                new RuleDecisionThresholds(40, 80),
                new NeuralReviewThreshold(1.5145587f)
        );
    }

    @Test
    void shouldConfigureAutomaticNeuralRejection() {
        RuntimeThresholdPolicy policy =
                new RuntimeThresholdPolicy(0L, 40, 80, 1.5145587f);

        FinGuardSettingsManager manager =
                new FinGuardSettingsManager(policy, null);

        manager.updateNeuralDecisionThresholds(
                1.5145587f, 4.0f, 0L, "admin-1"
        );

        assertEquals(1L, manager.snapshotWithVersion().version());
        assertEquals(
                4.0f,
                policy.snapshot().neuralRejectThresholdLogit().orElseThrow()
        );
        assertEquals(1, manager.auditHistory().size());
        assertEquals(
                FinGuardSettingsManager.ChangeType.NEURAL_DECISION_THRESHOLDS,
                manager.auditHistory().get(0).type()
        );
    }

    @Test
    void shouldPreserveNeuralRejectionWhenRuleThresholdsChange() {
        RuntimeThresholdPolicy policy =
                new RuntimeThresholdPolicy(0L, 40, 80, 1.5145587f);

        FinGuardSettingsManager manager =
                new FinGuardSettingsManager(policy, null);

        manager.updateNeuralDecisionThresholds(
                1.5145587f, 4.0f, 0L, "admin-1"
        );

        manager.updateRuleThresholds(
                45, 85, 1L, "admin-2"
        );

        assertEquals(2L, policy.snapshot().version());
        assertEquals(
                4.0f,
                policy.snapshot().neuralRejectThresholdLogit().orElseThrow()
        );
    }

    @Test
    void shouldRollbackAutomaticNeuralRejection() {
        RuntimeThresholdPolicy policy =
                new RuntimeThresholdPolicy(0L, 40, 80, 1.5145587f);

        FinGuardSettingsManager manager =
                new FinGuardSettingsManager(policy, null);

        manager.updateNeuralDecisionThresholds(
                1.5145587f, 4.0f, 0L, "admin-1"
        );

        manager.rollbackToVersion(0L, 1L, "admin-2");

        assertEquals(2L, policy.snapshot().version());
        assertTrue(
                policy.snapshot().neuralRejectThresholdLogit().isEmpty()
        );
        assertEquals(2, manager.auditHistory().size());
    }
}