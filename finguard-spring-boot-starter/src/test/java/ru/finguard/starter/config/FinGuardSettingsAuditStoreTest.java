package ru.finguard.starter.config;

import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import ru.finguard.core.decision.RuleDecisionThresholds;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.neural.NeuralReviewThreshold;
import ru.finguard.core.provider.TransactionDataProvider;
import ru.finguard.starter.autoconfigure.FinGuardAutoConfiguration;
import ru.finguard.core.decision.NeuralDataQualityPolicy;
import ru.finguard.core.decision.RuntimeThresholdPolicy;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertFalse;

class FinGuardSettingsAuditStoreTest {

    @Test
    void shouldPersistAuditBeforeApplyingRuleChange() {
        RuleDecisionThresholds thresholds =
                new RuleDecisionThresholds(40, 80);

        TestAuditStore store = new TestAuditStore(0);

        store.beforeAppend = () -> {
            assertEquals(
                    40,
                    thresholds.snapshot().reviewThreshold()
            );
            assertEquals(
                    80,
                    thresholds.snapshot().rejectThreshold()
            );
        };

        FinGuardSettingsManager manager =
                new FinGuardSettingsManager(
                        thresholds,
                        new NeuralReviewThreshold(1.5145587f),
                        store
                );

        manager.updateRuleThresholds(30, 90, 0, "admin-1");

        assertEquals(30, thresholds.snapshot().reviewThreshold());
        assertEquals(90, thresholds.snapshot().rejectThreshold());
        assertEquals(1, manager.snapshotWithVersion().version());
        assertEquals(1, store.lastVersion());
        assertEquals(1, store.entries.size());
        assertEquals("admin-1", store.entries.get(0).actorId());
    }

    @Test
    void shouldNotApplyChangesWhenStorageFails() {
        RuleDecisionThresholds thresholds =
                new RuleDecisionThresholds(40, 80);

        NeuralReviewThreshold neural =
                new NeuralReviewThreshold(1.5145587f);

        TestAuditStore store = new TestAuditStore(0);
        store.fail = true;

        FinGuardSettingsManager manager =
                new FinGuardSettingsManager(
                        thresholds,
                        neural,
                        store
                );

        assertThrows(
                IllegalStateException.class,
                () -> manager.updateRuleThresholds(
                        30,
                        90,
                        0,
                        "admin-1"
                )
        );

        assertThrows(
                IllegalStateException.class,
                () -> manager.setCustomNeuralThreshold(
                        2.5f,
                        0,
                        "admin-1"
                )
        );

        assertEquals(40, thresholds.snapshot().reviewThreshold());
        assertEquals(80, thresholds.snapshot().rejectThreshold());
        assertEquals(1.5145587f, neural.get());
        assertEquals(0, manager.snapshotWithVersion().version());
        assertTrue(manager.auditHistory().isEmpty());
        assertTrue(store.entries.isEmpty());
    }

    @Test
    void shouldRejectInvalidChangeBeforeWritingToStore() {
        TestAuditStore store = new TestAuditStore(0);

        FinGuardSettingsManager manager =
                new FinGuardSettingsManager(
                        new RuleDecisionThresholds(40, 80),
                        null,
                        store
                );

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
                () -> manager.updateRuleThresholds(
                        30,
                        90,
                        0,
                        " "
                )
        );

        assertEquals(0, store.entries.size());
        assertEquals(0, manager.snapshotWithVersion().version());
    }

    @Test
    void shouldRestoreVersionFromPersistentStore() {
        TestAuditStore store = new TestAuditStore(0);

        FinGuardSettingsManager original =
                new FinGuardSettingsManager(
                        new RuleDecisionThresholds(40, 80),
                        null,
                        store
                );

        for (int i = 0; i < 15; i++) {
            original.updateRuleThresholds(
                    25,
                    95,
                    i,
                    "admin-seed"
            );
        }

        FinGuardSettingsManager restored =
                new FinGuardSettingsManager(
                        new RuleDecisionThresholds(40, 80),
                        null,
                        store
                );

        assertEquals(15, restored.snapshotWithVersion().version());
        assertEquals(25, restored.snapshot().reviewThreshold());
        assertEquals(95, restored.snapshot().rejectThreshold());

        restored.updateRuleThresholds(
                30,
                90,
                15,
                "admin-2"
        );

        assertEquals(16, restored.snapshotWithVersion().version());
        assertEquals(16, store.lastVersion());
        assertEquals(16, store.entries.get(15).version());
    }

    @Test
    void shouldRejectStaleWriteFromAnotherManager() {
        TestAuditStore store = new TestAuditStore(0);

        FinGuardSettingsManager first =
                new FinGuardSettingsManager(
                        new RuleDecisionThresholds(40, 80),
                        null,
                        store
                );

        FinGuardSettingsManager second =
                new FinGuardSettingsManager(
                        new RuleDecisionThresholds(40, 80),
                        null,
                        store
                );

        first.updateRuleThresholds(30, 90, 0, "admin-1");

        assertThrows(
                IllegalStateException.class,
                () -> second.updateRuleThresholds(
                        20,
                        95,
                        0,
                        "admin-2"
                )
        );

        assertEquals(1, store.lastVersion());
        assertEquals(1, store.entries.size());
        assertEquals(40, second.snapshot().reviewThreshold());
        assertEquals(80, second.snapshot().rejectThreshold());
    }

    @Test
    void shouldConnectBankAuditStoreThroughSpring() {
        TestAuditStore store = new TestAuditStore(0);

        new ApplicationContextRunner()
                .withConfiguration(
                        AutoConfigurations.of(
                                FinGuardAutoConfiguration.class
                        )
                )
                .withBean(
                        TransactionDataProvider.class,
                        () -> transfer -> TransactionHistorySnapshot.empty(
                                transfer.timestamp()
                        )
                )
                .withBean(
                        SettingsAuditStore.class,
                        () -> store
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    FinGuardSettingsManager manager =
                            context.getBean(FinGuardSettingsManager.class);

                    manager.updateRuleThresholds(
                            30,
                            90,
                            0,
                            "spring-admin"
                    );

                    assertEquals(1, store.entries.size());
                    assertEquals(1, store.lastVersion());
                    assertEquals(
                            "spring-admin",
                            store.entries.get(0).actorId()
                    );
                });
    }

    @Test
    void shouldRestoreRulesAndNeuralThresholdAfterRestart() {
        TestAuditStore store = new TestAuditStore(0);

        FinGuardSettingsManager original =
                new FinGuardSettingsManager(
                        new RuleDecisionThresholds(40, 80),
                        new NeuralReviewThreshold(1.5145587f),
                        store
                );

        original.updateRuleThresholds(
                30,
                90,
                0,
                "admin-1"
        );

        original.setCustomNeuralThreshold(
                2.5f,
                1,
                "admin-1"
        );

        RuleDecisionThresholds restoredRules =
                new RuleDecisionThresholds(40, 80);

        NeuralReviewThreshold restoredNeural =
                new NeuralReviewThreshold(1.5145587f);

        FinGuardSettingsManager restored =
                new FinGuardSettingsManager(
                        restoredRules,
                        restoredNeural,
                        store
                );

        assertEquals(2, restored.snapshotWithVersion().version());
        assertEquals(30, restoredRules.snapshot().reviewThreshold());
        assertEquals(90, restoredRules.snapshot().rejectThreshold());
        assertEquals(2.5f, restoredNeural.get());

        assertEquals(
                original.snapshot(),
                restored.snapshot()
        );
    }

    @Test
    void shouldFailWhenPersistedVersionHasNoEntry() {
        TestAuditStore store = new TestAuditStore(15);

        assertThrows(
                IllegalStateException.class,
                () -> new FinGuardSettingsManager(
                        new RuleDecisionThresholds(40, 80),
                        null,
                        store
                )
        );
    }

    @Test
    void shouldFailWhenNeuralConfigurationChanged() {
        TestAuditStore store = new TestAuditStore(0);

        FinGuardSettingsManager original =
                new FinGuardSettingsManager(
                        new RuleDecisionThresholds(40, 80),
                        new NeuralReviewThreshold(1.5145587f),
                        store
                );

        original.setCustomNeuralThreshold(
                2.5f,
                0,
                "admin-1"
        );

        assertThrows(
                IllegalStateException.class,
                () -> new FinGuardSettingsManager(
                        new RuleDecisionThresholds(40, 80),
                        null,
                        store
                )
        );
    }

    @Test
    void shouldSynchronizeRulesAndNeuralBetweenInstances() {
        TestAuditStore store = new TestAuditStore(0);

        FinGuardSettingsManager first = new FinGuardSettingsManager(
                new RuleDecisionThresholds(40, 80),
                new NeuralReviewThreshold(1.5145587f),
                store
        );

        RuleDecisionThresholds secondRules =
                new RuleDecisionThresholds(40, 80);

        NeuralReviewThreshold secondNeural =
                new NeuralReviewThreshold(1.5145587f);

        FinGuardSettingsManager second = new FinGuardSettingsManager(
                secondRules,
                secondNeural,
                store
        );

        first.updateRuleThresholds(30, 90, 0, "admin-1");

        first.useNeuralMode(
                NeuralReviewMode.BALANCED,
                1,
                "admin-1"
        );

        assertEquals(0, second.snapshotWithVersion().version());

        assertTrue(second.refreshFromStore());

        assertEquals(2, second.snapshotWithVersion().version());
        assertEquals(30, secondRules.snapshot().reviewThreshold());
        assertEquals(90, secondRules.snapshot().rejectThreshold());
        assertEquals(3.7777753f, secondNeural.get());
        assertEquals(first.snapshot(), second.snapshot());
    }

    @Test
    void shouldNotRefreshWhenVersionIsUnchanged() {
        TestAuditStore store = new TestAuditStore(0);

        FinGuardSettingsManager manager = new FinGuardSettingsManager(
                new RuleDecisionThresholds(40, 80),
                null,
                store
        );

        assertFalse(manager.refreshFromStore());

        manager.updateRuleThresholds(30, 90, 0, "admin-1");

        assertFalse(manager.refreshFromStore());
        assertEquals(1, manager.snapshotWithVersion().version());
        assertEquals(1, manager.auditHistory().size());
    }

    @Test
    void shouldRejectOutdatedAdminUpdateAfterSynchronization() {
        TestAuditStore store = new TestAuditStore(0);

        FinGuardSettingsManager first = new FinGuardSettingsManager(
                new RuleDecisionThresholds(40, 80),
                null,
                store
        );

        FinGuardSettingsManager second = new FinGuardSettingsManager(
                new RuleDecisionThresholds(40, 80),
                null,
                store
        );

        first.updateRuleThresholds(30, 90, 0, "admin-1");

        assertTrue(second.refreshFromStore());

        assertThrows(
                IllegalStateException.class,
                () -> second.updateRuleThresholds(
                        20,
                        95,
                        0,
                        "admin-2"
                )
        );

        assertEquals(1, store.lastVersion());
        assertEquals(1, store.entries.size());
        assertEquals(30, second.snapshot().reviewThreshold());
    }

    @Test
    void shouldRejectClearedPersistentHistory() {
        TestAuditStore store = new TestAuditStore(0);

        FinGuardSettingsManager manager = new FinGuardSettingsManager(
                new RuleDecisionThresholds(40, 80),
                null,
                store
        );

        manager.updateRuleThresholds(30, 90, 0, "admin-1");

        store.entries.clear();
        store.version = 0;

        assertThrows(
                IllegalStateException.class,
                manager::refreshFromStore
        );

        assertEquals(1, manager.snapshotWithVersion().version());
        assertEquals(30, manager.snapshot().reviewThreshold());
        assertEquals(90, manager.snapshot().rejectThreshold());
    }

    @Test
    void shouldPersistRollbackAndRestoreItAfterRestart() {
        TestAuditStore store = new TestAuditStore(0);

        FinGuardSettingsManager manager = new FinGuardSettingsManager(
                new RuleDecisionThresholds(40, 80),
                null,
                store
        );

        manager.updateRuleThresholds(
                30,
                90,
                0,
                "admin-1"
        );

        manager.rollbackLastChange(1, "admin-2");

        assertEquals(2, store.lastVersion());
        assertEquals(2, store.entries.size());

        assertEquals(
                FinGuardSettingsManager.ChangeType.ROLLBACK,
                store.entries.get(1).type()
        );

        FinGuardSettingsManager restored = new FinGuardSettingsManager(
                new RuleDecisionThresholds(30, 90),
                null,
                store
        );

        assertEquals(2, restored.snapshotWithVersion().version());
        assertEquals(40, restored.snapshot().reviewThreshold());
        assertEquals(80, restored.snapshot().rejectThreshold());
    }

    @Test
    void shouldNotRollbackWhenAuditStoreFails() {
        TestAuditStore store = new TestAuditStore(0);

        FinGuardSettingsManager manager = new FinGuardSettingsManager(
                new RuleDecisionThresholds(40, 80),
                null,
                store
        );

        manager.updateRuleThresholds(
                30,
                90,
                0,
                "admin-1"
        );

        store.fail = true;

        assertThrows(
                IllegalStateException.class,
                () -> manager.rollbackLastChange(1, "admin-2")
        );

        assertEquals(1, manager.snapshotWithVersion().version());
        assertEquals(30, manager.snapshot().reviewThreshold());
        assertEquals(90, manager.snapshot().rejectThreshold());
        assertEquals(1, store.lastVersion());
        assertEquals(1, store.entries.size());
    }

    @Test
    void shouldRollbackToHistoricalVersionFromPersistentStore() {
        TestAuditStore store = new TestAuditStore(0);

        FinGuardSettingsManager manager = new FinGuardSettingsManager(
                new RuleDecisionThresholds(40, 80),
                new NeuralReviewThreshold(1.5145587f),
                store
        );

        manager.updateRuleThresholds(30, 90, 0, "admin-1");
        manager.useNeuralMode(NeuralReviewMode.BALANCED, 1, "admin-1");
        manager.updateRuleThresholds(20, 95, 2, "admin-1");

        manager.rollbackToVersion(1, 3, "admin-2");

        assertEquals(4, manager.snapshotWithVersion().version());
        assertEquals(30, manager.snapshot().reviewThreshold());
        assertEquals(90, manager.snapshot().rejectThreshold());

        assertEquals(
                1.5145587f,
                manager.snapshot()
                        .neuralReviewThresholdLogit()
                        .orElseThrow()
        );

        assertEquals(4, store.entries.size());
        assertEquals(4, store.lastVersion());

        assertEquals(
                FinGuardSettingsManager.ChangeType.ROLLBACK,
                store.entries.get(3).type()
        );
    }

    @Test
    void shouldRollbackToInitialSettings() {
        TestAuditStore store = new TestAuditStore(0);

        FinGuardSettingsManager manager = new FinGuardSettingsManager(
                new RuleDecisionThresholds(40, 80),
                new NeuralReviewThreshold(1.5145587f),
                store
        );

        manager.updateRuleThresholds(30, 90, 0, "admin-1");
        manager.setCustomNeuralThreshold(2.5f, 1, "admin-1");

        manager.rollbackToVersion(0, 2, "admin-2");

        assertEquals(3, manager.snapshotWithVersion().version());
        assertEquals(40, manager.snapshot().reviewThreshold());
        assertEquals(80, manager.snapshot().rejectThreshold());

        assertEquals(
                1.5145587f,
                manager.snapshot()
                        .neuralReviewThresholdLogit()
                        .orElseThrow()
        );
    }

    @Test
    void shouldRejectRollbackWhenHistoricalEntryIsMissing() {
        TestAuditStore store = new TestAuditStore(0);

        FinGuardSettingsManager manager = new FinGuardSettingsManager(
                new RuleDecisionThresholds(40, 80),
                null,
                store
        );

        manager.updateRuleThresholds(30, 90, 0, "admin-1");
        manager.updateRuleThresholds(20, 95, 1, "admin-1");

        store.entries.remove(0);

        assertThrows(
                IllegalStateException.class,
                () -> manager.rollbackToVersion(0, 2, "admin-2")
        );

        assertEquals(2, manager.snapshotWithVersion().version());
        assertEquals(20, manager.snapshot().reviewThreshold());
        assertEquals(95, manager.snapshot().rejectThreshold());
        assertEquals(2, store.lastVersion());
    }

    @Test
    void shouldPersistAndRestoreNeuralDataQualityPolicy() {
        TestAuditStore store = new TestAuditStore(0);

        RuntimeThresholdPolicy policy = new RuntimeThresholdPolicy(
                0L, 40, 80, 1.5145587f, 4.0f
        );

        FinGuardSettingsManager manager =
                new FinGuardSettingsManager(policy, store);

        NeuralDataQualityPolicy quality = new NeuralDataQualityPolicy(
                new NeuralDataQualityPolicy.Requirements(
                        3, 1, 1, false
                ),
                new NeuralDataQualityPolicy.Requirements(
                        10, 5, 3, false
                ),
                NeuralDataQualityPolicy.MissingDataAction.REVIEW
        );

        manager.updateNeuralDataQualityPolicy(
                quality,
                0L,
                "bank-admin"
        );

        assertEquals(1L, manager.snapshotWithVersion().version());
        assertEquals(quality, policy.snapshot().neuralDataQualityPolicy());
        assertEquals(1, store.entries.size());
        assertEquals(
                FinGuardSettingsManager.ChangeType.NEURAL_DATA_QUALITY,
                store.entries.get(0).type()
        );

        manager.updateRuleThresholds(
                45, 85, 1L, "bank-admin"
        );

        assertEquals(quality, policy.snapshot().neuralDataQualityPolicy());

        RuntimeThresholdPolicy restoredPolicy =
                new RuntimeThresholdPolicy(
                        0L, 40, 80, 1.5145587f, 4.0f
                );

        FinGuardSettingsManager restored =
                new FinGuardSettingsManager(restoredPolicy, store);

        assertEquals(2L, restored.snapshotWithVersion().version());
        assertEquals(
                quality,
                restoredPolicy.snapshot().neuralDataQualityPolicy()
        );

        restored.rollbackToVersion(
                0L, 2L, "bank-admin"
        );

        assertEquals(3L, restored.snapshotWithVersion().version());
        assertEquals(
                NeuralDataQualityPolicy.unrestricted(),
                restoredPolicy.snapshot().neuralDataQualityPolicy()
        );
        assertEquals(3, store.entries.size());
        assertEquals(
                FinGuardSettingsManager.ChangeType.ROLLBACK,
                store.entries.get(2).type()
        );
    }

    @Test
    void shouldNotApplyQualityPolicyWhenAuditStorageFails() {
        TestAuditStore store = new TestAuditStore(0);

        RuntimeThresholdPolicy policy = new RuntimeThresholdPolicy(
                0L, 40, 80, 1.5145587f, 4.0f
        );

        FinGuardSettingsManager manager =
                new FinGuardSettingsManager(policy, store);

        NeuralDataQualityPolicy quality = new NeuralDataQualityPolicy(
                new NeuralDataQualityPolicy.Requirements(
                        3, 1, 1, false
                ),
                new NeuralDataQualityPolicy.Requirements(
                        10, 5, 3, false
                ),
                NeuralDataQualityPolicy.MissingDataAction.REVIEW
        );

        store.fail = true;

        assertThrows(
                IllegalStateException.class,
                () -> manager.updateNeuralDataQualityPolicy(
                        quality,
                        0L,
                        "bank-admin"
                )
        );

        assertEquals(0L, manager.snapshotWithVersion().version());
        assertEquals(
                NeuralDataQualityPolicy.unrestricted(),
                policy.snapshot().neuralDataQualityPolicy()
        );
        assertTrue(manager.auditHistory().isEmpty());
        assertTrue(store.entries.isEmpty());
    }

    private static final class TestAuditStore
            implements SettingsAuditStore {

        private long version;
        private boolean fail;
        private Runnable beforeAppend = () -> {
        };

        private final List<FinGuardSettingsManager.AuditEntry> entries =
                new ArrayList<>();

        private TestAuditStore(long initialVersion) {
            version = initialVersion;
        }

        @Override
        public synchronized long lastVersion() {
            return version;
        }

        @Override
        public synchronized Optional<FinGuardSettingsManager.AuditEntry> latestEntry() {
            if (entries.isEmpty()) {
                return Optional.empty();
            }

            return Optional.of(entries.get(entries.size() - 1));
        }

        @Override
        public synchronized Optional<FinGuardSettingsManager.AuditEntry> findEntry(
                long version
        ) {
            return entries.stream()
                    .filter(entry -> entry.version() == version)
                    .findFirst();
        }

        @Override
        public synchronized void appendIfVersionMatches(
                FinGuardSettingsManager.AuditEntry entry,
                long expectedVersion
        ) {
            if (fail) {
                throw new IllegalStateException(
                        "Audit storage unavailable"
                );
            }

            if (version != expectedVersion) {
                throw new IllegalStateException(
                        "Persistent settings version conflict"
                );
            }

            if (entry.version() != version + 1) {
                throw new IllegalArgumentException(
                        "Invalid audit entry version"
                );
            }

            beforeAppend.run();

            entries.add(entry);
            version = entry.version();
        }
    }
}