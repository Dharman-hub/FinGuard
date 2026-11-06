package ru.finguard.starter.autoconfigure;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import ru.finguard.core.api.RiskEngine;
import ru.finguard.core.feature.V3InputNormalizer;
import ru.finguard.core.model.SupplierIbanContext;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.neural.NeuralPredictor;
import ru.finguard.core.neural.OnnxNeuralPredictor;
import ru.finguard.core.neural.V3NeuralRiskEvaluator;
import ru.finguard.core.provider.SupplierIbanDataProvider;
import ru.finguard.core.provider.TransactionDataProvider;
import ru.finguard.core.neural.NeuralReviewThreshold;
import ru.finguard.starter.config.FinGuardSettingsManager;
import ru.finguard.starter.config.NeuralReviewMode;
import ru.finguard.core.decision.RuntimeThresholdPolicy;
import ru.finguard.core.model.DecisionReason;
import ru.finguard.core.model.RiskDecision;

import java.math.BigDecimal;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicInteger;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.util.HexFormat;

import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FinGuardNeuralAutoConfigurationTest {

    @TempDir
    Path temporaryDirectory;

    private final ApplicationContextRunner contextRunner =
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
                    );

    @Test
    void shouldAutomaticallyLoadRealOnnxModel() throws Exception {
        configuredRunner().run(context -> {
            assertFalse(context.getStartupFailure() != null);

            assertNotNull(context.getBean(OnnxNeuralPredictor.class));
            assertNotNull(context.getBean(V3InputNormalizer.class));
            assertNotNull(context.getBean(V3NeuralRiskEvaluator.class));

            NeuralPredictor predictor =
                    context.getBean(NeuralPredictor.class);

            assertTrue(predictor instanceof OnnxNeuralPredictor);

            RiskEngine engine = context.getBean(RiskEngine.class);

            var result = engine.evaluate(createTransfer());

            assertNotNull(result);
            assertNotNull(result.decision());
            assertFalse(result.factors().contains("NEURAL_UNAVAILABLE"));
        });
    }

    @Test
    void shouldConnectSupplierIbanProvider() throws Exception {
        AtomicInteger calls = new AtomicInteger();

        configuredRunner()
                .withBean(
                        SupplierIbanDataProvider.class,
                        () -> transfer -> {
                            calls.incrementAndGet();

                            return new SupplierIbanContext(
                                    true,
                                    "DE89370400440532013000",
                                    true,
                                    false
                            );
                        }
                )
                .run(context -> {
                    assertFalse(context.getStartupFailure() != null);

                    RiskEngine engine =
                            context.getBean(RiskEngine.class);

                    var result = engine.evaluate(createTransfer());

                    assertEquals(1, calls.get());
                    assertNotNull(result.decision());

                    assertFalse(
                            result.factors().contains(
                                    "SUPPLIER_IBAN_UNAVAILABLE"
                            )
                    );

                    assertFalse(
                            result.factors().contains(
                                    "NEURAL_UNAVAILABLE"
                            )
                    );
                });
    }

    @Test
    void shouldFailStartupWhenModelIsMissing() throws Exception {
        String normalizationPath = resourcePath(
                "finguard-v3/normalization.csv"
        ).toString();

        String missingModelPath = temporaryDirectory
                .resolve("missing-model.onnx")
                .toString();

        contextRunner
                .withPropertyValues(
                        "finguard.neural.enabled=true",
                        "finguard.neural.model-path=" + missingModelPath,
                        "finguard.neural.normalization-path="
                                + normalizationPath
                )
                .run(context -> {
                    assertNotNull(context.getStartupFailure());
                });
    }

    @Test
    void shouldUpdateNeuralThresholdWithoutRecreatingEngine()
            throws Exception {
        configuredRunner().run(context -> {
            assertNull(context.getStartupFailure());

            FinGuardSettingsManager manager =
                    context.getBean(FinGuardSettingsManager.class);

            RiskEngine engine = context.getBean(RiskEngine.class);

            manager.setCustomNeuralThreshold(Float.MAX_VALUE);

            var first = engine.evaluate(createTransfer());

            assertEquals(
                    Float.MAX_VALUE,
                    first.neural().orElseThrow().reviewThresholdLogit()
            );

            assertFalse(first.neural().orElseThrow().requiresReview());
            assertEquals(1, first.trace().orElseThrow().policyVersion());

            manager.setCustomNeuralThreshold(-Float.MAX_VALUE);

            var second = engine.evaluate(createTransfer());

            assertTrue(second.neural().orElseThrow().requiresReview());
            assertEquals(2, second.trace().orElseThrow().policyVersion());

            assertSame(engine, context.getBean(RiskEngine.class));
        });
    }

    @Test
    void shouldPreserveBankProvidedThresholdBean() throws Exception {
        RuntimeThresholdPolicy customPolicy =
                new RuntimeThresholdPolicy(
                        0,
                        40,
                        80,
                        2.5f
                );

        configuredRunner()
                .withBean(
                        RuntimeThresholdPolicy.class,
                        () -> customPolicy
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    assertSame(
                            customPolicy,
                            context.getBean(RuntimeThresholdPolicy.class)
                    );

                    FinGuardSettingsManager manager =
                            context.getBean(FinGuardSettingsManager.class);

                    assertSame(customPolicy, manager.runtimePolicy());

                    RiskEngine engine = context.getBean(RiskEngine.class);

                    var result = engine.evaluate(createTransfer());

                    assertEquals(
                            2.5f,
                            result.neural().orElseThrow().reviewThresholdLogit()
                    );
                });
    }

    @Test
    void shouldNotCreateNeuralThresholdWhenNeuralDisabled() {
        contextRunner.run(context -> {
            assertNull(context.getStartupFailure());

            assertEquals(
                    0,
                    context.getBeansOfType(NeuralReviewThreshold.class).size()
            );
        });
    }

    @Test
    void shouldApplySettingsManagerToRunningNeuralEngine()
            throws Exception {
        configuredRunner().run(context -> {
            assertNull(context.getStartupFailure());

            FinGuardSettingsManager manager =
                    context.getBean(FinGuardSettingsManager.class);

            RiskEngine engine = context.getBean(RiskEngine.class);

            manager.useNeuralMode(NeuralReviewMode.BALANCED);

            var first = engine.evaluate(createTransfer());

            assertEquals(
                    3.7777753f,
                    first.neural().orElseThrow().reviewThresholdLogit()
            );

            manager.setCustomNeuralThreshold(2.5f);

            var second = engine.evaluate(createTransfer());

            assertEquals(
                    2.5f,
                    second.neural().orElseThrow().reviewThresholdLogit()
            );

            assertSame(engine, context.getBean(RiskEngine.class));
        });
    }

    @Test
    void shouldReturnModelIdentityAndAppliedThresholds()
            throws Exception {
        Path modelPath = resourcePath(
                "finguard-v3/finguard-model.onnx"
        );

        String expectedModelHash = HexFormat.of().formatHex(
                MessageDigest.getInstance("SHA-256").digest(
                        Files.readAllBytes(modelPath)
                )
        );

        configuredRunner().run(context -> {
            assertNull(context.getStartupFailure());

            RiskEngine engine = context.getBean(RiskEngine.class);

            var result = engine.evaluate(createTransfer());
            var trace = result.trace().orElseThrow();

            assertEquals(
                    expectedModelHash,
                    trace.modelSha256().orElseThrow()
            );

            assertEquals(40, trace.reviewThreshold());
            assertEquals(80, trace.rejectThreshold());

            assertEquals(
                    1.5145587f,
                    trace.neuralReviewThresholdLogit().orElseThrow()
            );

            assertEquals(
                    result.neural().orElseThrow().reviewThresholdLogit(),
                    trace.neuralReviewThresholdLogit().orElseThrow()
            );

            assertEquals(
                    64,
                    trace.thresholdsFingerprint().length()
            );
        });
    }

    @Test
    void shouldApplyAutomaticNeuralRejectionFromProperties()
            throws Exception {
        configuredRunner()
                .withBean(
                        NeuralPredictor.class,
                        () -> input -> 6.0f
                )
                .withPropertyValues(
                        "finguard.neural.reject-threshold-logit=4.0"
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    RiskEngine engine = context.getBean(RiskEngine.class);
                    var result = engine.evaluate(createTransfer());

                    assertEquals(RiskDecision.REJECTED, result.decision());
                    assertEquals(DecisionReason.NEURAL_REJECT, result.reason());
                    assertTrue(result.factors().contains("NEURAL_REJECT"));

                    var trace = result.trace().orElseThrow();

                    assertEquals(0L, trace.policyVersion());
                    assertEquals(
                            1.5145587f,
                            trace.neuralReviewThresholdLogit().orElseThrow()
                    );
                    assertEquals(
                            4.0f,
                            trace.neuralRejectThresholdLogit().orElseThrow()
                    );
                });
    }

    @Test
    void shouldApplyAndRollbackNeuralRejectionWithoutRecreatingEngine()
            throws Exception {
        configuredRunner()
                .withBean(
                        NeuralPredictor.class,
                        () -> input -> 6.0f
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    FinGuardSettingsManager manager =
                            context.getBean(FinGuardSettingsManager.class);

                    RiskEngine engine = context.getBean(RiskEngine.class);

                    var initial = engine.evaluate(createTransfer());

                    assertEquals(RiskDecision.REVIEW, initial.decision());
                    assertEquals(
                            0L,
                            initial.trace().orElseThrow().policyVersion()
                    );

                    manager.updateNeuralDecisionThresholds(
                            1.5145587f,
                            4.0f,
                            0L,
                            "bank-admin"
                    );

                    var rejected = engine.evaluate(createTransfer());

                    assertEquals(RiskDecision.REJECTED, rejected.decision());
                    assertEquals(
                            DecisionReason.NEURAL_REJECT,
                            rejected.reason()
                    );
                    assertEquals(
                            1L,
                            rejected.trace().orElseThrow().policyVersion()
                    );
                    assertEquals(
                            4.0f,
                            rejected.trace().orElseThrow()
                                    .neuralRejectThresholdLogit()
                                    .orElseThrow()
                    );

                    assertEquals(1, manager.auditHistory().size());
                    assertEquals(
                            FinGuardSettingsManager.ChangeType
                                    .NEURAL_DECISION_THRESHOLDS,
                            manager.auditHistory().get(0).type()
                    );
                    assertEquals(
                            "bank-admin",
                            manager.auditHistory().get(0).actorId()
                    );

                    manager.rollbackToVersion(
                            0L,
                            1L,
                            "bank-admin"
                    );

                    var restored = engine.evaluate(createTransfer());

                    assertEquals(RiskDecision.REVIEW, restored.decision());
                    assertEquals(
                            2L,
                            restored.trace().orElseThrow().policyVersion()
                    );
                    assertTrue(
                            restored.trace().orElseThrow()
                                    .neuralRejectThresholdLogit()
                                    .isEmpty()
                    );

                    assertEquals(2, manager.auditHistory().size());
                    assertSame(engine, context.getBean(RiskEngine.class));
                });
    }

    @Test
    void shouldRejectInvalidAutomaticNeuralRejectionConfiguration()
            throws Exception {
        configuredRunner()
                .withPropertyValues(
                        "finguard.neural.reject-threshold-logit=1.0"
                )
                .run(context ->
                        assertNotNull(context.getStartupFailure())
                );
    }

    private ApplicationContextRunner configuredRunner()
            throws URISyntaxException {
        Path model = resourcePath(
                "finguard-v3/finguard-model.onnx"
        );

        Path normalization = resourcePath(
                "finguard-v3/normalization.csv"
        );

        return contextRunner.withPropertyValues(
                "finguard.enabled=true",
                "finguard.neural.enabled=true",
                "finguard.neural.model-path=" + model,
                "finguard.neural.normalization-path=" + normalization,
                "finguard.neural.review-threshold-logit=1.5145587"
        );
    }

    private Path resourcePath(String name)
            throws URISyntaxException {
        var resource = Objects.requireNonNull(
                getClass().getClassLoader().getResource(name),
                "Test resource not found: " + name
        );

        return Path.of(resource.toURI());
    }

    private TransferData createTransfer() {
        return new TransferData(
                1001L,
                2001L,
                3001L,
                new BigDecimal("1000.00"),
                "EUR",
                Instant.parse("2026-10-30T12:00:00Z")
        );
    }
}