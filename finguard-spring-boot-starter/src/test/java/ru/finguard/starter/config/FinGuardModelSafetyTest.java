package ru.finguard.starter.config;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import ru.finguard.core.api.RiskEngineBuilder;
import ru.finguard.core.decision.RuntimeThresholdPolicy;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.neural.OnnxModelRegistry;
import ru.finguard.core.neural.VersionedOnnxRiskEngine;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.ConcurrentModificationException;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import ai.onnxruntime.OrtException;

import static org.junit.jupiter.api.Assertions.*;

class FinGuardModelSafetyTest {

    @TempDir
    Path tempDir;

    @Test
    void shouldRejectModifiedOnnxModel() throws Exception {
        MemoryStore store = new MemoryStore();

        try (FinGuardModelManager manager = createManager(store)) {
            Path modifiedModel = Files.copy(
                    model(),
                    tempDir.resolve("candidate.onnx")
            );

            manager.register("v3.1", modifiedModel, normalization());

            long revision = store.load().orElseThrow().revision();

            Files.write(
                    modifiedModel,
                    new byte[]{1, 2, 3},
                    StandardOpenOption.APPEND
            );

            assertThrows(
                    IllegalStateException.class,
                    () -> manager.activate("v3.1", "v3.0", "admin")
            );

            assertUnchanged(manager, store, revision);
        }
    }

    @Test
    void shouldRejectModifiedNormalizationProfile() throws Exception {
        MemoryStore store = new MemoryStore();

        try (FinGuardModelManager manager = createManager(store)) {
            Path profile = Files.copy(
                    normalization(),
                    tempDir.resolve("normalization.csv")
            );

            manager.register("v3.1", model(), profile);

            long revision = store.load().orElseThrow().revision();

            Files.writeString(
                    profile,
                    "\ninvalid",
                    StandardOpenOption.APPEND
            );

            assertThrows(
                    IllegalStateException.class,
                    () -> manager.activate("v3.1", "v3.0", "admin")
            );

            assertUnchanged(manager, store, revision);
        }
    }

    @Test
    void shouldRejectInvalidNormalizationDuringRegistration()
            throws Exception {
        MemoryStore store = new MemoryStore();

        try (FinGuardModelManager manager = createManager(store)) {
            Path invalid = tempDir.resolve("invalid.csv");
            Files.writeString(invalid, "invalid normalization");

            long revision = store.load().orElseThrow().revision();

            assertThrows(
                    IOException.class,
                    () -> manager.register("invalid", model(), invalid)
            );

            assertTrue(manager.find("invalid").isEmpty());
            assertUnchanged(manager, store, revision);
        }
    }

    @Test
    void shouldNotRegisterModelWhenStorageFails() throws Exception {
        MemoryStore store = new MemoryStore();

        try (FinGuardModelManager manager = createManager(store)) {
            long revision = store.load().orElseThrow().revision();
            store.failWrites = true;

            assertThrows(
                    IOException.class,
                    () -> manager.register(
                            "v3.1",
                            model(),
                            normalization()
                    )
            );

            assertTrue(manager.find("v3.1").isEmpty());
            assertUnchanged(manager, store, revision);
        }
    }

    @Test
    void shouldRejectChangedArtifactsAfterRestart() throws Exception {
        MemoryStore store = new MemoryStore();

        Path profile = Files.copy(
                normalization(),
                tempDir.resolve("saved-normalization.csv")
        );

        try (FinGuardModelManager manager = createManager(store)) {
            manager.register("v3.1", model(), profile);
            manager.activate("v3.1", "v3.0", "admin");
        }

        List<String> lines = Files.readAllLines(
                profile,
                StandardCharsets.UTF_8
        );

        String[] columns = lines.get(1).split(",", -1);
        columns[1] = Double.toString(
                Double.parseDouble(columns[1]) + 1.0
        );

        lines.set(1, String.join(",", columns));

        Files.write(profile, lines, StandardCharsets.UTF_8);

        assertThrows(
                IllegalStateException.class,
                () -> createManager(store)
        );

        assertEquals(
                "v3.1",
                store.load().orElseThrow().activeVersion()
        );
    }

    @Test
    void shouldAllowOnlyOneConcurrentActivation() throws Exception {
        MemoryStore store = new MemoryStore();

        try (FinGuardModelManager manager = createManager(store)) {
            manager.register("v3.1", model(), normalization());
            manager.register("v3.2", model(), normalization());

            CountDownLatch start = new CountDownLatch(1);
            ExecutorService executor = Executors.newFixedThreadPool(2);

            try {
                Future<Exception> first = executor.submit(
                        () -> attemptActivation(
                                start, manager, "v3.1", "admin-1"
                        )
                );

                Future<Exception> second = executor.submit(
                        () -> attemptActivation(
                                start, manager, "v3.2", "admin-2"
                        )
                );

                start.countDown();

                Exception firstError = first.get(30, TimeUnit.SECONDS);
                Exception secondError = second.get(30, TimeUnit.SECONDS);

                int successes = (firstError == null ? 1 : 0)
                        + (secondError == null ? 1 : 0);

                assertEquals(1, successes);

                Exception conflict = firstError != null
                        ? firstError
                        : secondError;

                assertInstanceOf(
                        ConcurrentModificationException.class,
                        conflict
                );

                assertEquals(1L, manager.auditVersion());
                assertEquals(1, manager.auditHistory().size());

                ModelStateStore.State saved =
                        store.load().orElseThrow();

                assertEquals(3L, saved.revision());
                assertEquals(
                        manager.activeVersion(),
                        saved.activeVersion()
                );

                assertEquals(
                        "v3.0",
                        manager.previousVersion().orElseThrow()
                );

                String activeHash = manager.find(
                        manager.activeVersion()
                ).orElseThrow().modelSha256();

                assertEquals(
                        activeHash,
                        manager.riskEngine()
                                .evaluate(transfer())
                                .trace()
                                .orElseThrow()
                                .modelSha256()
                                .orElseThrow()
                );
            } finally {
                start.countDown();
                executor.shutdownNow();
            }
        }
    }

    private static Exception attemptActivation(
            CountDownLatch start,
            FinGuardModelManager manager,
            String version,
            String actor
    ) throws InterruptedException {
        start.await();

        try {
            manager.activate(version, "v3.0", actor);
            return null;
        } catch (Exception exception) {
            return exception;
        }
    }

    private static void assertUnchanged(
            FinGuardModelManager manager,
            MemoryStore store,
            long expectedRevision
    ) throws Exception {
        assertEquals("v3.0", manager.activeVersion());
        assertTrue(manager.previousVersion().isEmpty());
        assertEquals(0L, manager.auditVersion());
        assertTrue(manager.auditHistory().isEmpty());

        ModelStateStore.State saved = store.load().orElseThrow();

        assertEquals(expectedRevision, saved.revision());
        assertEquals("v3.0", saved.activeVersion());
        assertEquals(0L, saved.auditVersion());
    }

    private static FinGuardModelManager createManager(
            MemoryStore store
    ) throws Exception {
        OnnxModelRegistry registry = new OnnxModelRegistry();
        ModelStateStore.State saved = store.load().orElse(null);

        String activeVersion;
        String previousVersion = null;

        if (saved == null) {
            registry.register("v3.0", model(), normalization());
            activeVersion = "v3.0";
        } else {
            for (OnnxModelRegistry.ModelVersion stored : saved.models()) {
                OnnxModelRegistry.ModelVersion checked =
                        registry.register(
                                stored.version(),
                                stored.modelPath(),
                                stored.normalizationPath()
                        );

                if (!stored.equals(checked)) {
                    throw new IllegalStateException(
                            "Stored model artifacts changed"
                    );
                }
            }

            activeVersion = saved.activeVersion();
            previousVersion = saved.previousVersion();
        }

        VersionedOnnxRiskEngine engine =
                new VersionedOnnxRiskEngine(
                        registry,
                        activeVersion,
                        previousVersion,
                        (evaluator, sha256) -> new RiskEngineBuilder(
                                transfer -> TransactionHistorySnapshot.empty(
                                        transfer.timestamp()
                                )
                        )
                                .neuralReview(evaluator, 0.0f)
                                .modelSha256(sha256)
                                .build()
                );

        try {
            return new FinGuardModelManager(registry, engine, store);
        } catch (RuntimeException exception) {
            engine.close();
            throw exception;
        }
    }

    private static TransferData transfer() {
        return new TransferData(
                101L,
                201L,
                301L,
                BigDecimal.valueOf(1000),
                "RUB",
                Instant.parse("2026-11-01T00:00:00Z")
        );
    }

    private static Path model() throws Exception {
        return resource("finguard-v3/finguard-model.onnx");
    }

    private static Path normalization() throws Exception {
        return resource("finguard-v3/normalization.csv");
    }

    private static Path resource(String name) throws Exception {
        return Path.of(Objects.requireNonNull(
                FinGuardModelSafetyTest.class
                        .getClassLoader()
                        .getResource(name)
        ).toURI());
    }

    private static final class MemoryStore implements ModelStateStore {

        private State state;
        private boolean failWrites;

        @Override
        public synchronized Optional<State> load() {
            return Optional.ofNullable(state);
        }

        @Override
        public synchronized void saveIfRevisionMatches(
                State next,
                long expectedRevision
        ) throws IOException {
            if (failWrites) {
                throw new IOException("Storage unavailable");
            }

            long actualRevision = state == null
                    ? -1L
                    : state.revision();

            if (actualRevision != expectedRevision) {
                throw new IllegalStateException(
                        "Persistent revision conflict"
                );
            }

            state = next;
        }
    }

    @Test
    void shouldRejectInvalidOnnxDuringRegistration() throws Exception {
        MemoryStore store = new MemoryStore();

        Path invalidModel = tempDir.resolve("invalid.onnx");
        Files.write(invalidModel, new byte[]{1, 2, 3, 4});

        try (FinGuardModelManager manager = createManager(store)) {
            long revision = store.load().orElseThrow().revision();

            assertThrows(
                    OrtException.class,
                    () -> manager.register(
                            "invalid",
                            invalidModel,
                            normalization()
                    )
            );

            assertTrue(manager.find("invalid").isEmpty());
            assertUnchanged(manager, store, revision);
        }
    }

    @Test
    void shouldPrepareModelPolicyPreviewWithoutActivation() throws Exception {
        MemoryStore store = new MemoryStore();

        try (FinGuardModelManager manager = createManager(store)) {
            manager.register("bank-v1", model(), normalization());

            RuntimeThresholdPolicy policy = new RuntimeThresholdPolicy(
                    0L, 40, 80, 1.5145587f, 4.0f
            );

            var preview = manager.previewActivationWithPolicy(
                    "bank-v1",
                    policy,
                    2.25f,
                    6.0f,
                    5
            );

            assertEquals("v3.0", preview.currentModelVersion());
            assertEquals("bank-v1", preview.target().version());
            assertEquals(0L, preview.before().version());
            assertEquals(1L, preview.after().version());

            assertEquals(
                    2.25f,
                    preview.after().neuralReviewThresholdLogit().orElseThrow()
            );

            assertEquals(
                    6.0f,
                    preview.after().neuralRejectThresholdLogit().orElseThrow()
            );

            manager.verifyActivationPreview(preview, policy);

            assertEquals("v3.0", manager.activeVersion());
            assertEquals(0L, policy.snapshot().version());
            assertEquals(1.5145587f,
                    policy.snapshot().neuralReviewThresholdLogit().orElseThrow());
        }
    }

    @Test
    void shouldRejectPreviewAfterPolicyChanged() throws Exception {
        MemoryStore store = new MemoryStore();

        try (FinGuardModelManager manager = createManager(store)) {
            manager.register("bank-v1", model(), normalization());

            RuntimeThresholdPolicy policy = new RuntimeThresholdPolicy(
                    0L, 40, 80, 1.5145587f, 4.0f
            );

            var preview = manager.previewActivationWithPolicy(
                    "bank-v1",
                    policy,
                    2.25f,
                    null,
                    5
            );

            policy.activate(
                    0L,
                    1L,
                    40,
                    80,
                    1.7f,
                    5.0f
            );

            assertThrows(
                    ConcurrentModificationException.class,
                    () -> manager.verifyActivationPreview(preview, policy)
            );

            assertEquals("v3.0", manager.activeVersion());
        }
    }

    @Test
    void shouldRejectPreviewAfterCandidateModelChanged() throws Exception {
        MemoryStore store = new MemoryStore();

        try (FinGuardModelManager manager = createManager(store)) {
            Path candidate = Files.copy(
                    model(),
                    tempDir.resolve("bank-candidate.onnx")
            );

            manager.register(
                    "bank-v1",
                    candidate,
                    normalization()
            );

            RuntimeThresholdPolicy policy = new RuntimeThresholdPolicy(
                    0L, 40, 80, 1.5145587f, 4.0f
            );

            var preview = manager.previewActivationWithPolicy(
                    "bank-v1",
                    policy,
                    2.25f,
                    null,
                    5
            );

            Files.write(
                    candidate,
                    new byte[]{1, 2, 3},
                    StandardOpenOption.APPEND
            );

            assertThrows(
                    IllegalStateException.class,
                    () -> manager.verifyActivationPreview(preview, policy)
            );

            assertEquals("v3.0", manager.activeVersion());
            assertEquals(0L, policy.snapshot().version());
        }
    }
}