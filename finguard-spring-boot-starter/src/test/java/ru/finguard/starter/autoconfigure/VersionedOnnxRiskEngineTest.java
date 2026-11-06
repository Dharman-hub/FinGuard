package ru.finguard.starter.autoconfigure;

import org.junit.jupiter.api.Test;
import ru.finguard.core.api.RiskEngineBuilder;
import ru.finguard.core.model.RiskAssessment;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.neural.OnnxModelRegistry;
import ru.finguard.core.neural.VersionedOnnxRiskEngine;
import ru.finguard.core.provider.TransactionDataProvider;

import java.math.BigDecimal;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ConcurrentModificationException;
import java.util.Objects;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import static org.junit.jupiter.api.Assertions.*;

class VersionedOnnxRiskEngineTest {

    @Test
    void shouldActivateAndRollbackWithMatchingModelTrace() throws Exception {
        OnnxModelRegistry registry = registry();
        var registered = registry.find("v3.0").orElseThrow();

        try (VersionedOnnxRiskEngine engine = new VersionedOnnxRiskEngine(
                registry,
                "v3.0",
                factory(transfer -> TransactionHistorySnapshot.empty(transfer.timestamp()))
        )) {
            RiskAssessment before = engine.evaluate(transfer(1L));
            assertEquals("v3.0", engine.activeVersion());
            assertEquals(registered.modelSha256(),
                    before.trace().orElseThrow().modelSha256().orElseThrow());
            assertTrue(engine.previousVersion().isEmpty());

            engine.activate("v3.1", "v3.0");

            assertEquals("v3.1", engine.activeVersion());
            assertEquals("v3.0", engine.previousVersion().orElseThrow());
            assertEquals(registry.find("v3.1").orElseThrow().modelSha256(),
                    engine.evaluate(transfer(2L)).trace().orElseThrow()
                            .modelSha256().orElseThrow());

            engine.rollback("v3.1");

            assertEquals("v3.0", engine.activeVersion());
            assertEquals("v3.1", engine.previousVersion().orElseThrow());
            assertEquals(registered.modelSha256(),
                    engine.evaluate(transfer(3L)).trace().orElseThrow()
                            .modelSha256().orElseThrow());
        }
    }

    @Test
    void shouldRejectInvalidVersionAndVersionConflict() throws Exception {
        OnnxModelRegistry registry = registry();

        try (VersionedOnnxRiskEngine engine = new VersionedOnnxRiskEngine(
                registry,
                "v3.0",
                factory(transfer -> TransactionHistorySnapshot.empty(transfer.timestamp()))
        )) {
            assertThrows(IllegalStateException.class,
                    () -> engine.rollback("v3.0"));

            assertThrows(IllegalArgumentException.class,
                    () -> engine.activate("unknown", "v3.0"));

            assertThrows(ConcurrentModificationException.class,
                    () -> engine.activate("v3.1", "wrong-version"));

            assertEquals("v3.0", engine.activeVersion());
            assertTrue(engine.previousVersion().isEmpty());
            assertTrue(engine.evaluate(transfer(4L)).trace().isPresent());
        }
    }

    @Test
    void shouldWaitForRunningEvaluationBeforeSwitching() throws Exception {
        OnnxModelRegistry registry = registry();
        CountDownLatch entered = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);

        TransactionDataProvider provider = transfer -> {
            if (transfer.transferId() == 99L) {
                entered.countDown();
                try {
                    if (!release.await(20, TimeUnit.SECONDS)) {
                        throw new IllegalStateException("Timeout waiting for test release");
                    }
                } catch (InterruptedException exception) {
                    Thread.currentThread().interrupt();
                    throw new IllegalStateException(exception);
                }
            }
            return TransactionHistorySnapshot.empty(transfer.timestamp());
        };

        ExecutorService executor = Executors.newFixedThreadPool(2);
        try (VersionedOnnxRiskEngine engine = new VersionedOnnxRiskEngine(
                registry, "v3.0", factory(provider)
        )) {
            Future<RiskAssessment> evaluation = executor.submit(
                    () -> engine.evaluate(transfer(99L))
            );

            assertTrue(entered.await(10, TimeUnit.SECONDS));
            Future<?> activation = executor.submit(() -> {
                engine.activate("v3.1", "v3.0");
                return null;
            });

            assertThrows(TimeoutException.class,
                    () -> activation.get(150, TimeUnit.MILLISECONDS));

            release.countDown();
            RiskAssessment oldResult = evaluation.get(10, TimeUnit.SECONDS);
            activation.get(10, TimeUnit.SECONDS);

            assertEquals(registry.find("v3.0").orElseThrow().modelSha256(),
                    oldResult.trace().orElseThrow().modelSha256().orElseThrow());
            assertEquals("v3.1", engine.activeVersion());
            assertTrue(engine.evaluate(transfer(100L)).trace().isPresent());
        } finally {
            release.countDown();
            executor.shutdownNow();
        }
    }

    @Test
    void shouldRejectEvaluationsAfterClose() throws Exception {
        OnnxModelRegistry registry = registry();
        VersionedOnnxRiskEngine engine = new VersionedOnnxRiskEngine(
                registry,
                "v3.0",
                factory(transfer -> TransactionHistorySnapshot.empty(transfer.timestamp()))
        );

        engine.close();
        engine.close();

        assertThrows(IllegalStateException.class,
                () -> engine.evaluate(transfer(5L)));
        assertThrows(IllegalStateException.class, engine::activeVersion);
    }

    private static VersionedOnnxRiskEngine.EngineFactory factory(
            TransactionDataProvider provider
    ) {
        return (evaluator, modelSha256) -> new RiskEngineBuilder(provider)
                .neuralReview(evaluator, 0.0f)
                .modelSha256(modelSha256)
                .build();
    }

    private static TransferData transfer(long id) {
        return new TransferData(
                id, 1L, 2L, BigDecimal.valueOf(100), "RUB",
                Instant.parse("2026-11-01T00:00:00Z")
        );
    }

    private static OnnxModelRegistry registry() throws Exception {
        Path model = resource("finguard-v3/finguard-model.onnx");
        Path normalization = resource("finguard-v3/normalization.csv");

        OnnxModelRegistry registry = new OnnxModelRegistry();
        registry.register("v3.0", model, normalization);
        registry.register("v3.1", model, normalization);
        return registry;
    }

    private static Path resource(String name) throws Exception {
        return Path.of(Objects.requireNonNull(
                VersionedOnnxRiskEngineTest.class.getClassLoader().getResource(name),
                "Missing resource: " + name
        ).toURI());
    }
}