package ru.finguard.starter.config;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;
import ru.finguard.core.decision.NeuralDataQualityPolicy;
import ru.finguard.core.decision.RuntimeThresholdPolicy;
import ru.finguard.core.neural.OnnxModelRegistry;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ConcurrentModificationException;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class FileModelPolicySnapshotStoreTest {
    @TempDir
    Path directory;

    @Test
    void shouldRestoreModelRegistryAuditAndBothPolicies() throws Exception {
        Path file = directory.resolve("paired-state.bin");

        var first = new FileModelPolicySnapshotStore(file);

        first.saveIfRevisionMatches(initialState(), -1L);
        first.saveIfRevisionMatches(updatedState(2.25f), 0L);

        var restarted = new FileModelPolicySnapshotStore(file);
        var restored = restarted.load().orElseThrow();

        assertEquals(updatedState(2.25f), restored);
        assertEquals("bank-v1", restored.modelState().activeVersion());
        assertEquals("v3.0", restored.modelState().previousVersion());
        assertEquals(1L, restored.modelState().auditVersion());
        assertEquals(2, restored.modelState().models().size());

        assertEquals(
                2.25f,
                restored.activePolicy()
                        .neuralReviewThresholdLogit()
                        .orElseThrow()
                        .floatValue()
        );

        assertEquals(
                1.5145587f,
                restored.previousPolicy()
                        .neuralReviewThresholdLogit()
                        .orElseThrow()
                        .floatValue()
        );
    }

    @Test
    @Timeout(10)
    void shouldAllowOnlyOneConcurrentUpdate() throws Exception {
        Path file = directory.resolve("paired-state.bin");

        var first = new FileModelPolicySnapshotStore(file);
        var second = new FileModelPolicySnapshotStore(file);

        first.saveIfRevisionMatches(initialState(), -1L);

        var executor = Executors.newFixedThreadPool(2);
        var start = new CountDownLatch(1);

        try {
            Future<Boolean> a = executor.submit(
                    () -> saveAfterLatch(
                            first,
                            updatedState(2.25f),
                            start
                    )
            );

            Future<Boolean> b = executor.submit(
                    () -> saveAfterLatch(
                            second,
                            updatedState(3.0f),
                            start
                    )
            );

            start.countDown();

            assertNotEquals(
                    a.get(5, TimeUnit.SECONDS),
                    b.get(5, TimeUnit.SECONDS)
            );

            assertEquals(
                    1L,
                    first.load().orElseThrow().revision()
            );
        } finally {
            start.countDown();
            executor.shutdownNow();
        }
    }

    @Test
    void shouldRejectDamagedFileInsteadOfReturningEmptyState()
            throws Exception {
        Path file = directory.resolve("paired-state.bin");

        var store = new FileModelPolicySnapshotStore(file);

        store.saveIfRevisionMatches(initialState(), -1L);

        Files.write(
                file,
                new byte[]{1, 2, 3},
                StandardOpenOption.APPEND
        );

        assertThrows(
                IOException.class,
                () -> new FileModelPolicySnapshotStore(file).load()
        );
    }

    @Test
    void shouldRejectStaleRevisionWithoutChangingState()
            throws Exception {
        var store = new FileModelPolicySnapshotStore(
                directory.resolve("paired-state.bin")
        );

        store.saveIfRevisionMatches(initialState(), -1L);

        assertThrows(
                ConcurrentModificationException.class,
                () -> store.saveIfRevisionMatches(
                        updatedState(2.25f),
                        -1L
                )
        );

        assertEquals(
                initialState(),
                store.load().orElseThrow()
        );
    }

    private boolean saveAfterLatch(
            FileModelPolicySnapshotStore store,
            ModelPolicySnapshotStore.State next,
            CountDownLatch start
    ) throws Exception {
        start.await();

        try {
            store.saveIfRevisionMatches(next, 0L);
            return true;
        } catch (ConcurrentModificationException exception) {
            return false;
        }
    }

    private ModelPolicySnapshotStore.State initialState() {
        return new ModelPolicySnapshotStore.State(
                new ModelStateStore.State(
                        0L,
                        0L,
                        "v3.0",
                        null,
                        List.of(model("v3.0")),
                        List.of()
                ),
                policy(0L, 1.5145587f),
                null
        );
    }

    private ModelPolicySnapshotStore.State updatedState(float review) {
        var audit = new FinGuardModelManager.ModelAuditEntry(
                1L,
                Instant.parse("2026-11-03T00:00:00Z"),
                "bank-admin",
                FinGuardModelManager.ChangeType.ACTIVATE,
                "v3.0",
                "bank-v1",
                "hash-model",
                "hash-profile"
        );

        return new ModelPolicySnapshotStore.State(
                new ModelStateStore.State(
                        1L,
                        1L,
                        "bank-v1",
                        "v3.0",
                        List.of(
                                model("v3.0"),
                                model("bank-v1")
                        ),
                        List.of(audit)
                ),
                policy(1L, review),
                policy(0L, 1.5145587f)
        );
    }

    private OnnxModelRegistry.ModelVersion model(String version) {
        return new OnnxModelRegistry.ModelVersion(
                version,
                directory.resolve(version + ".onnx"),
                directory.resolve(version + ".csv"),
                "hash-model",
                "hash-profile",
                70,
                35
        );
    }

    private RuntimeThresholdPolicy.Snapshot policy(
            long version,
            float review
    ) {
        return new RuntimeThresholdPolicy.Snapshot(
                version,
                40,
                80,
                Optional.of(review),
                Optional.of(6.0f),
                NeuralDataQualityPolicy.unrestricted()
        );
    }
}