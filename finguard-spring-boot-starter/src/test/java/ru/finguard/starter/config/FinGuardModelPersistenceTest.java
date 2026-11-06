package ru.finguard.starter.config;

import org.junit.jupiter.api.Test;
import ru.finguard.core.api.RiskEngineBuilder;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.neural.OnnxModelRegistry;
import ru.finguard.core.neural.VersionedOnnxRiskEngine;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

class FinGuardModelPersistenceTest {

    @Test
    void shouldRestoreActiveVersionAndRollbackAfterRestart() throws Exception {
        MemoryStore store = new MemoryStore();

        try (FinGuardModelManager manager = createManager(store)) {
            manager.register("v3.1", model(), normalization());
            manager.activate("v3.1", "v3.0", "admin-1");

            assertEquals("v3.1", manager.activeVersion());
        }

        try (FinGuardModelManager restored = createManager(store)) {
            assertEquals("v3.1", restored.activeVersion());
            assertEquals("v3.0", restored.previousVersion().orElseThrow());
            assertEquals(1L, restored.auditVersion());
            assertEquals(2, restored.versions().size());

            restored.rollback("v3.1", "admin-2");

            assertEquals("v3.0", restored.activeVersion());
            assertEquals(2L, restored.auditVersion());
            assertEquals(2, restored.auditHistory().size());
        }
    }

    @Test
    void shouldKeepOldModelWhenPersistenceFails() throws Exception {
        MemoryStore store = new MemoryStore();

        try (FinGuardModelManager manager = createManager(store)) {
            manager.register("v3.1", model(), normalization());

            long revision = store.load().orElseThrow().revision();
            store.fail = true;

            assertThrows(
                    IOException.class,
                    () -> manager.activate("v3.1", "v3.0", "admin")
            );

            assertEquals("v3.0", manager.activeVersion());
            assertEquals(0L, manager.auditVersion());
            assertEquals(revision, store.load().orElseThrow().revision());
        }
    }

    @Test
    void shouldRejectConflictingPersistentRevision() throws Exception {
        MemoryStore store = new MemoryStore();

        try (FinGuardModelManager first = createManager(store);
             FinGuardModelManager second = createManager(store)) {

            first.register("v3.1", model(), normalization());

            assertThrows(
                    IllegalStateException.class,
                    () -> second.register("v3.2", model(), normalization())
            );

            assertEquals(1, second.versions().size());
            assertEquals("v3.0", second.activeVersion());
        }
    }

    private static FinGuardModelManager createManager(
            MemoryStore store
    ) throws Exception {
        OnnxModelRegistry registry = new OnnxModelRegistry();
        ModelStateStore.State saved = store.load().orElse(null);

        String active;
        String previous = null;

        if (saved == null) {
            registry.register("v3.0", model(), normalization());
            active = "v3.0";
        } else {
            for (OnnxModelRegistry.ModelVersion model : saved.models()) {
                assertEquals(
                        model,
                        registry.register(
                                model.version(),
                                model.modelPath(),
                                model.normalizationPath()
                        )
                );
            }

            active = saved.activeVersion();
            previous = saved.previousVersion();
        }

        VersionedOnnxRiskEngine engine = new VersionedOnnxRiskEngine(
                registry,
                active,
                previous,
                (evaluator, sha) -> new RiskEngineBuilder(
                        transfer -> TransactionHistorySnapshot.empty(
                                transfer.timestamp()
                        )
                )
                        .neuralReview(evaluator, 0.0f)
                        .modelSha256(sha)
                        .build()
        );

        try {
            return new FinGuardModelManager(registry, engine, store);
        } catch (RuntimeException exception) {
            engine.close();
            throw exception;
        }
    }

    private static Path model() throws Exception {
        return resource("finguard-v3/finguard-model.onnx");
    }

    private static Path normalization() throws Exception {
        return resource("finguard-v3/normalization.csv");
    }

    private static Path resource(String path) throws Exception {
        return Path.of(Objects.requireNonNull(
                FinGuardModelPersistenceTest.class
                        .getClassLoader()
                        .getResource(path)
        ).toURI());
    }

    private static final class MemoryStore implements ModelStateStore {

        private State saved;
        private boolean fail;

        @Override
        public synchronized Optional<State> load() {
            return Optional.ofNullable(saved);
        }

        @Override
        public synchronized void saveIfRevisionMatches(
                State next,
                long expectedRevision
        ) throws IOException {
            if (fail) {
                throw new IOException(
                        "Model storage unavailable"
                );
            }

            long actualRevision = saved == null
                    ? -1L
                    : saved.revision();

            if (actualRevision != expectedRevision) {
                throw new IllegalStateException(
                        "Model state revision conflict"
                );
            }

            saved = next;
        }
    }
}