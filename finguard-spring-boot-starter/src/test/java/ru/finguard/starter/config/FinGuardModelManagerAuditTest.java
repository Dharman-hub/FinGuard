package ru.finguard.starter.config;

import org.junit.jupiter.api.Test;
import ru.finguard.core.api.RiskEngineBuilder;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.neural.OnnxModelRegistry;
import ru.finguard.core.neural.VersionedOnnxRiskEngine;

import java.nio.file.Path;
import java.util.ConcurrentModificationException;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.*;

class FinGuardModelManagerAuditTest {

    @Test
    void shouldAuditActivationAndRollback() throws Exception {
        try (FinGuardModelManager manager = createManager()) {
            manager.activate("v3.1", "v3.0", "bank-admin-1");
            assertEquals("v3.1", manager.activeVersion());
            assertEquals(1L, manager.auditVersion());
            assertEquals(1, manager.auditHistory().size());

            var activation = manager.auditHistory().get(0);
            assertEquals(FinGuardModelManager.ChangeType.ACTIVATE, activation.type());
            assertEquals("bank-admin-1", activation.actorId());
            assertEquals("v3.0", activation.fromVersion());
            assertEquals("v3.1", activation.toVersion());
            assertTrue(activation.modelSha256().matches("[0-9a-f]{64}"));
            assertTrue(activation.normalizationSha256().matches("[0-9a-f]{64}"));

            manager.rollback("v3.1", "bank-admin-2");
            assertEquals("v3.0", manager.activeVersion());
            assertEquals(2L, manager.auditVersion());
            assertEquals(2, manager.auditHistory().size());

            var rollback = manager.auditHistory().get(0);
            assertEquals(FinGuardModelManager.ChangeType.ROLLBACK, rollback.type());
            assertEquals("bank-admin-2", rollback.actorId());
            assertEquals("v3.1", rollback.fromVersion());
            assertEquals("v3.0", rollback.toVersion());
        }
    }

    @Test
    void shouldNotAuditFailedActivation() throws Exception {
        try (FinGuardModelManager manager = createManager()) {
            assertThrows(ConcurrentModificationException.class,
                    () -> manager.activate("v3.1", "outdated", "admin"));
            assertThrows(IllegalArgumentException.class,
                    () -> manager.activate("v3.1", "v3.0", " "));

            assertEquals("v3.0", manager.activeVersion());
            assertEquals(0L, manager.auditVersion());
            assertTrue(manager.auditHistory().isEmpty());
        }
    }

    private static FinGuardModelManager createManager() throws Exception {
        Path model = resource("finguard-v3/finguard-model.onnx");
        Path normalization = resource("finguard-v3/normalization.csv");

        OnnxModelRegistry registry = new OnnxModelRegistry();
        registry.register("v3.0", model, normalization);
        registry.register("v3.1", model, normalization);

        VersionedOnnxRiskEngine engine = new VersionedOnnxRiskEngine(
                registry,
                "v3.0",
                (evaluator, sha256) -> new RiskEngineBuilder(
                        transfer -> TransactionHistorySnapshot.empty(transfer.timestamp())
                ).neuralReview(evaluator, 0.0f).modelSha256(sha256).build()
        );

        return new FinGuardModelManager(registry, engine);
    }

    private static Path resource(String path) throws Exception {
        return Path.of(Objects.requireNonNull(
                FinGuardModelManagerAuditTest.class.getClassLoader().getResource(path)
        ).toURI());
    }
}