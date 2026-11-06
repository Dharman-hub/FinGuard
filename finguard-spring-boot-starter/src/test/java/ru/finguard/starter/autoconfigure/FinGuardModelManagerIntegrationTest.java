package ru.finguard.starter.autoconfigure;

import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import ru.finguard.core.api.RiskEngine;
import ru.finguard.core.model.RiskAssessment;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.monitoring.RiskMetrics;
import ru.finguard.core.provider.TransactionDataProvider;
import ru.finguard.starter.config.FinGuardModelManager;
import ru.finguard.starter.config.FinGuardSettingsManager;

import java.math.BigDecimal;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.*;

class FinGuardModelManagerIntegrationTest {

    private final ApplicationContextRunner contextRunner =
            new ApplicationContextRunner()
                    .withConfiguration(AutoConfigurations.of(
                            FinGuardAutoConfiguration.class
                    ));

    @Test
    void shouldSwitchAndRollbackModelThroughSpringStarter()
            throws Exception {
        Path model = resource("finguard-v3/finguard-model.onnx");
        Path normalization = resource("finguard-v3/normalization.csv");

        contextRunner
                .withBean(
                        TransactionDataProvider.class,
                        () -> transfer -> TransactionHistorySnapshot.empty(
                                transfer.timestamp()
                        )
                )
                .withPropertyValues(
                        "finguard.neural.enabled=true",
                        "finguard.neural.versioning-enabled=true",
                        "finguard.neural.initial-version=v3.0",
                        "finguard.neural.model-path=" + model,
                        "finguard.neural.normalization-path=" + normalization
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    FinGuardModelManager manager =
                            context.getBean(FinGuardModelManager.class);

                    RiskEngine engine = context.getBean(RiskEngine.class);
                    RiskMetrics metrics = context.getBean(RiskMetrics.class);

                    assertEquals(
                            1,
                            context.getBeansOfType(RiskEngine.class).size()
                    );

                    assertEquals("v3.0", manager.activeVersion());

                    RiskAssessment before = engine.evaluate(transfer());

                    String originalHash = before.trace()
                            .orElseThrow()
                            .modelSha256()
                            .orElseThrow();

                    assertDoesNotThrow(() -> manager.register(
                            "v3.1",
                            model,
                            normalization
                    ));

                    assertDoesNotThrow(() ->
                            manager.activate("v3.1", "v3.0")
                    );

                    assertEquals("v3.1", manager.activeVersion());
                    assertEquals(
                            "v3.0",
                            manager.previousVersion().orElseThrow()
                    );

                    RiskAssessment updated = engine.evaluate(transfer());

                    assertEquals(
                            originalHash,
                            updated.trace().orElseThrow()
                                    .modelSha256().orElseThrow()
                    );

                    assertDoesNotThrow(() -> manager.rollback("v3.1"));

                    assertEquals("v3.0", manager.activeVersion());

                    RiskAssessment restored = engine.evaluate(transfer());

                    assertEquals(
                            originalHash,
                            restored.trace().orElseThrow()
                                    .modelSha256().orElseThrow()
                    );

                    assertEquals(3L, metrics.snapshot().total());

                    FinGuardSettingsManager settings =
                            context.getBean(FinGuardSettingsManager.class);

                    settings.updateRuleThresholds(45, 85);

                    RiskAssessment afterSettingsChange =
                            engine.evaluate(transfer());

                    assertEquals(
                            1L,
                            afterSettingsChange.trace()
                                    .orElseThrow().policyVersion()
                    );

                    assertEquals("v3.0", manager.activeVersion());
                    assertEquals(4L, metrics.snapshot().total());
                });
    }

    @Test
    void shouldRejectVersioningWithoutNeuralEvaluation() {
        contextRunner
                .withBean(
                        TransactionDataProvider.class,
                        () -> transfer -> TransactionHistorySnapshot.empty(
                                transfer.timestamp()
                        )
                )
                .withPropertyValues(
                        "finguard.neural.enabled=false",
                        "finguard.neural.versioning-enabled=true"
                )
                .run(context -> {
                    assertNotNull(context.getStartupFailure());
                });
    }

    @Test
    void shouldKeepLegacyModeByDefault() {
        contextRunner
                .withBean(
                        TransactionDataProvider.class,
                        () -> transfer -> TransactionHistorySnapshot.empty(
                                transfer.timestamp()
                        )
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    assertTrue(
                            context.getBeansOfType(
                                    FinGuardModelManager.class
                            ).isEmpty()
                    );

                    assertNotNull(context.getBean(RiskEngine.class));
                });
    }

    @Test
    void shouldActivateAndRollbackModelWithItsThresholds() throws Exception {
        Path model = resource("finguard-v3/finguard-model.onnx");
        Path normalization = resource("finguard-v3/normalization.csv");

        contextRunner
                .withBean(
                        TransactionDataProvider.class,
                        () -> transfer -> TransactionHistorySnapshot.empty(
                                transfer.timestamp()
                        )
                )
                .withPropertyValues(
                        "finguard.neural.enabled=true",
                        "finguard.neural.versioning-enabled=true",
                        "finguard.neural.initial-version=v3.0",
                        "finguard.neural.model-path=" + model,
                        "finguard.neural.normalization-path=" + normalization
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    FinGuardModelManager manager =
                            context.getBean(FinGuardModelManager.class);

                    FinGuardSettingsManager settings =
                            context.getBean(FinGuardSettingsManager.class);

                    RiskEngine riskEngine = context.getBean(RiskEngine.class);

                    var original = riskEngine.evaluate(transfer())
                            .trace().orElseThrow();

                    assertDoesNotThrow(
                            () -> manager.register(
                                    "bank-v1",
                                    model,
                                    normalization
                            )
                    );

                    var preview = assertDoesNotThrow(
                            () -> manager.previewActivationWithPolicy(
                                    "bank-v1",
                                    settings.runtimePolicy(),
                                    2.25f,
                                    null,
                                    5
                            )
                    );

                    assertDoesNotThrow(
                            () -> manager.activateWithPolicy(
                                    preview,
                                    "bank-admin"
                            )
                    );

                    var activated = riskEngine.evaluate(transfer())
                            .trace().orElseThrow();

                    assertEquals("bank-v1", manager.activeVersion());
                    assertEquals(1L, activated.policyVersion());
                    assertEquals(
                            2.25f,
                            activated.neuralReviewThresholdLogit()
                                    .orElseThrow()
                    );

                    assertTrue(
                            activated.neuralRejectThresholdLogit().isEmpty()
                    );

                    assertThrows(
                            IllegalStateException.class,
                            () -> manager.activate(
                                    "v3.0",
                                    "bank-v1"
                            )
                    );

                    assertDoesNotThrow(
                            () -> manager.rollbackWithPolicy(
                                    "bank-v1",
                                    "bank-admin"
                            )
                    );

                    var restored = riskEngine.evaluate(transfer())
                            .trace().orElseThrow();

                    assertEquals("v3.0", manager.activeVersion());
                    assertEquals(2L, restored.policyVersion());
                    assertEquals(
                            original.neuralReviewThresholdLogit(),
                            restored.neuralReviewThresholdLogit()
                    );
                    assertEquals(
                            original.neuralRejectThresholdLogit(),
                            restored.neuralRejectThresholdLogit()
                    );
                    assertEquals(2L, manager.auditVersion());
                    assertEquals(
                            2L,
                            settings.snapshotWithVersion().version()
                    );
                });
    }

    @Test
    void shouldRejectStalePreviewWithoutSwitchingModel() throws Exception {
        Path model = resource("finguard-v3/finguard-model.onnx");
        Path normalization = resource("finguard-v3/normalization.csv");

        contextRunner
                .withBean(
                        TransactionDataProvider.class,
                        () -> transfer -> TransactionHistorySnapshot.empty(
                                transfer.timestamp()
                        )
                )
                .withPropertyValues(
                        "finguard.neural.enabled=true",
                        "finguard.neural.versioning-enabled=true",
                        "finguard.neural.initial-version=v3.0",
                        "finguard.neural.model-path=" + model,
                        "finguard.neural.normalization-path=" + normalization
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    FinGuardModelManager manager =
                            context.getBean(FinGuardModelManager.class);

                    FinGuardSettingsManager settings =
                            context.getBean(FinGuardSettingsManager.class);

                    assertDoesNotThrow(
                            () -> manager.register(
                                    "bank-v1",
                                    model,
                                    normalization
                            )
                    );

                    var preview = assertDoesNotThrow(
                            () -> manager.previewActivationWithPolicy(
                                    "bank-v1",
                                    settings.runtimePolicy(),
                                    2.25f,
                                    null,
                                    5
                            )
                    );

                    settings.updateNeuralDecisionThresholds(
                            1.7f,
                            null,
                            0L,
                            "bank-admin"
                    );

                    assertThrows(
                            java.util.ConcurrentModificationException.class,
                            () -> manager.activateWithPolicy(
                                    preview,
                                    "bank-admin"
                            )
                    );

                    assertEquals("v3.0", manager.activeVersion());
                    assertEquals(0L, manager.auditVersion());
                    assertEquals(
                            1.7f,
                            settings.runtimePolicy()
                                    .snapshot()
                                    .neuralReviewThresholdLogit()
                                    .orElseThrow()
                    );
                });
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

    private static Path resource(String name) throws Exception {
        return Path.of(Objects.requireNonNull(
                FinGuardModelManagerIntegrationTest.class
                        .getClassLoader().getResource(name)
        ).toURI());
    }
}