
package ru.finguard.starter.config;

import org.junit.jupiter.api.Test;
import ru.finguard.core.decision.RuntimeThresholdPolicy;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

class ThresholdPolicyActivationIntegrationTest {

    @Test
    void shouldKeepActivePolicyWhenAuditWriteFails() {
        TestAuditStore store = new TestAuditStore();

        RuntimeThresholdPolicy policy =
                new RuntimeThresholdPolicy(0L, 40, 80, null);

        FinGuardSettingsManager manager =
                new FinGuardSettingsManager(policy, store);

        ThresholdPolicyChangeService service =
                new ThresholdPolicyChangeService(manager, store);

        var preview = service.previewRuleThresholds(45, 85);

        store.failWrites = true;

        assertThrows(
                IllegalStateException.class,
                () -> service.activate(preview, "admin-1")
        );

        assertEquals(0L, manager.snapshotWithVersion().version());
        assertEquals(40, policy.snapshot().reviewThreshold());
        assertEquals(80, policy.snapshot().rejectThreshold());
        assertEquals(0L, store.lastVersion());
        assertTrue(manager.auditHistory().isEmpty());
    }

    @Test
    void shouldRejectStalePreviewAndRefreshFromStore() {
        TestAuditStore store = new TestAuditStore();

        RuntimeThresholdPolicy firstPolicy =
                new RuntimeThresholdPolicy(0L, 40, 80, null);

        RuntimeThresholdPolicy secondPolicy =
                new RuntimeThresholdPolicy(0L, 40, 80, null);

        FinGuardSettingsManager first =
                new FinGuardSettingsManager(firstPolicy, store);

        FinGuardSettingsManager second =
                new FinGuardSettingsManager(secondPolicy, store);

        ThresholdPolicyChangeService service =
                new ThresholdPolicyChangeService(first, store);

        var preview = service.previewRuleThresholds(45, 85);

        second.updateRuleThresholds(
                30, 90, 0L, "admin-2"
        );

        assertThrows(
                IllegalStateException.class,
                () -> service.activate(preview, "admin-1")
        );

        assertEquals(0L, first.snapshotWithVersion().version());
        assertEquals(40, firstPolicy.snapshot().reviewThreshold());

        assertTrue(first.refreshFromStore());

        assertEquals(1L, first.snapshotWithVersion().version());
        assertEquals(30, firstPolicy.snapshot().reviewThreshold());
        assertEquals(90, firstPolicy.snapshot().rejectThreshold());

        var updatedPreview = service.previewRuleThresholds(45, 85);

        service.activate(updatedPreview, "admin-1");

        assertEquals(2L, first.snapshotWithVersion().version());
        assertEquals(45, firstPolicy.snapshot().reviewThreshold());
        assertEquals(85, firstPolicy.snapshot().rejectThreshold());
        assertEquals(2L, store.lastVersion());
        assertEquals(2, store.entries.size());
    }

    @Test
    void shouldRestorePolicyInNewManager() {
        TestAuditStore store = new TestAuditStore();

        RuntimeThresholdPolicy firstPolicy =
                new RuntimeThresholdPolicy(0L, 40, 80, null);

        FinGuardSettingsManager first =
                new FinGuardSettingsManager(firstPolicy, store);

        ThresholdPolicyChangeService service =
                new ThresholdPolicyChangeService(first, store);

        var preview = service.previewRuleThresholds(45, 85);

        service.activate(preview, "admin-1");

        RuntimeThresholdPolicy restoredPolicy =
                new RuntimeThresholdPolicy(0L, 40, 80, null);

        FinGuardSettingsManager restored =
                new FinGuardSettingsManager(restoredPolicy, store);

        assertEquals(1L, restored.snapshotWithVersion().version());
        assertEquals(45, restoredPolicy.snapshot().reviewThreshold());
        assertEquals(85, restoredPolicy.snapshot().rejectThreshold());
        assertEquals(1L, store.lastVersion());
    }

    private static final class TestAuditStore
            implements SettingsAuditStore {

        private long version;
        private boolean failWrites;

        private final List<FinGuardSettingsManager.AuditEntry> entries =
                new ArrayList<>();

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
                long requestedVersion
        ) {
            return entries.stream()
                    .filter(entry -> entry.version() == requestedVersion)
                    .findFirst();
        }

        @Override
        public synchronized void appendIfVersionMatches(
                FinGuardSettingsManager.AuditEntry entry,
                long expectedVersion
        ) {
            if (failWrites) {
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

            entries.add(entry);
            version = entry.version();
        }
    }
}
