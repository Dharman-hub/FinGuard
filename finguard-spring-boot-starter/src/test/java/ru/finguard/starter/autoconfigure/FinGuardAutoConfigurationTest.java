package ru.finguard.starter.autoconfigure;

import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import ru.finguard.core.api.RiskEngine;
import ru.finguard.core.model.ActivityStatistics;
import ru.finguard.core.model.AmountStatistics;
import ru.finguard.core.model.RecipientStatistics;
import ru.finguard.core.model.RiskAssessment;
import ru.finguard.core.model.RiskDecision;
import ru.finguard.core.model.RiskHistoryStatistics;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.model.VelocityStatistics;
import ru.finguard.core.provider.TransactionDataProvider;
import ru.finguard.core.provider.ConfirmedFraudProvider;
import ru.finguard.starter.config.FinGuardSettingsManager;
import ru.finguard.core.monitoring.RiskMetrics;
import ru.finguard.core.monitoring.RiskMetricsPublisher;
import ru.finguard.core.monitoring.RiskMetricsSink;

import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicInteger;
import java.io.InputStream;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FinGuardAutoConfigurationTest {

    private final ApplicationContextRunner contextRunner = new ApplicationContextRunner()
            .withConfiguration(AutoConfigurations.of(FinGuardAutoConfiguration.class));

    private final Instant timestamp = Instant.parse("2026-10-24T10:00:00Z");

    @Test
    void shouldCreateRiskEngineWhenProviderExists() {
        contextRunner
                .withBean(TransactionDataProvider.class, () -> transfer -> createHistory())
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    RiskEngine engine = context.getBean(RiskEngine.class);
                    RiskAssessment result = engine.evaluate(createTransfer());

                    assertEquals(RiskDecision.APPROVED, result.decision());
                    assertEquals(25, result.riskScore());
                });
    }

    @Test
    void shouldNotCreateRiskEngineWithoutProvider() {
        contextRunner.run(context -> {
            assertNull(context.getStartupFailure());
            assertEquals(0, context.getBeansOfType(RiskEngine.class).size());
        });
    }

    @Test
    void shouldDisableFinGuardThroughProperties() {
        contextRunner
                .withBean(TransactionDataProvider.class, () -> transfer -> createHistory())
                .withPropertyValues("finguard.enabled=false")
                .run(context -> {
                    assertNull(context.getStartupFailure());
                    assertEquals(0, context.getBeansOfType(RiskEngine.class).size());
                });
    }

    @Test
    void shouldApplyCustomRuleWeight() {
        contextRunner
                .withBean(TransactionDataProvider.class, () -> transfer -> createHistory())
                .withPropertyValues("finguard.rules.unusual-amount.weight=90")
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    RiskAssessment result = context.getBean(RiskEngine.class)
                            .evaluate(createTransfer());

                    assertEquals(RiskDecision.REVIEW, result.decision());
                    assertEquals(90, result.riskScore());
                });
    }

    @Test
    void shouldApplyCustomDecisionThresholds() {
        contextRunner
                .withBean(TransactionDataProvider.class, () -> transfer -> createHistory())
                .withPropertyValues(
                        "finguard.review-threshold=20",
                        "finguard.reject-threshold=80"
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    RiskAssessment result = context.getBean(RiskEngine.class)
                            .evaluate(createTransfer());

                    assertEquals(RiskDecision.REVIEW, result.decision());
                    assertEquals(25, result.riskScore());
                });
    }

    @Test
    void shouldDisableIndividualRule() {
        contextRunner
                .withBean(TransactionDataProvider.class, () -> transfer -> createHistory())
                .withPropertyValues("finguard.rules.unusual-amount.enabled=false")
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    RiskAssessment result = context.getBean(RiskEngine.class)
                            .evaluate(createTransfer());

                    assertEquals(RiskDecision.APPROVED, result.decision());
                    assertEquals(0, result.riskScore());
                    assertFalse(result.factors().contains("UNUSUAL_AMOUNT"));
                });
    }

    @Test
    void shouldPreserveCustomRiskEngine() {
        RiskEngine customEngine = transfer -> new RiskAssessment(
                RiskDecision.REVIEW,
                77,
                List.of("CUSTOM")
        );

        contextRunner
                .withBean(TransactionDataProvider.class, () -> transfer -> createHistory())
                .withBean(RiskEngine.class, () -> customEngine)
                .run(context -> {
                    assertNull(context.getStartupFailure());
                    assertEquals(1, context.getBeansOfType(RiskEngine.class).size());
                    assertSame(customEngine, context.getBean(RiskEngine.class));
                });
    }

    @Test
    void shouldFailWhenAllRulesAreDisabled() {
        contextRunner
                .withBean(TransactionDataProvider.class, () -> transfer -> createHistory())
                .withPropertyValues(
                        "finguard.rules.unusual-amount.enabled=false",
                        "finguard.rules.high-velocity.enabled=false",
                        "finguard.rules.new-recipient.enabled=false",
                        "finguard.rules.unusual-activity.enabled=false"
                )
                .run(context -> assertNotNull(context.getStartupFailure()));
    }

    @Test
    void shouldRegisterAutoConfigurationInImportsFile() throws Exception {
        String path = "META-INF/spring/org.springframework.boot.autoconfigure.AutoConfiguration.imports";

        try (InputStream resource = getClass().getClassLoader().getResourceAsStream(path)) {
            assertNotNull(resource);

            String content = new String(resource.readAllBytes(), StandardCharsets.UTF_8);

            assertEquals(
                    FinGuardAutoConfiguration.class.getName(),
                    content.trim()
            );
        }
    }

    @Test
    void shouldRejectBankConfirmedHighRiskFraud() {
        contextRunner
                .withBean(
                        TransactionDataProvider.class,
                        () -> transfer -> createHistory()
                )
                .withBean(
                        ConfirmedFraudProvider.class,
                        () -> transfer -> true
                )
                .withPropertyValues(
                        "finguard.rules.unusual-amount.weight=90"
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    RiskAssessment result = context
                            .getBean(RiskEngine.class)
                            .evaluate(createTransfer());

                    assertEquals(RiskDecision.REJECTED, result.decision());
                    assertEquals(90, result.riskScore());
                    assertTrue(
                            result.factors().contains("BANK_CONFIRMED_FRAUD")
                    );
                });
    }

    @Test
    void shouldReviewHighRiskWithoutBankConfirmation() {
        contextRunner
                .withBean(
                        TransactionDataProvider.class,
                        () -> transfer -> createHistory()
                )
                .withBean(
                        ConfirmedFraudProvider.class,
                        () -> transfer -> false
                )
                .withPropertyValues(
                        "finguard.rules.unusual-amount.weight=90"
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    RiskAssessment result = context
                            .getBean(RiskEngine.class)
                            .evaluate(createTransfer());

                    assertEquals(RiskDecision.REVIEW, result.decision());
                    assertEquals(90, result.riskScore());
                });
    }

    @Test
    void shouldReviewWhenConfirmationProviderFails() {
        contextRunner
                .withBean(
                        TransactionDataProvider.class,
                        () -> transfer -> createHistory()
                )
                .withBean(
                        ConfirmedFraudProvider.class,
                        () -> transfer -> {
                            throw new IllegalStateException(
                                    "Confirmation unavailable"
                            );
                        }
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    RiskAssessment result = context
                            .getBean(RiskEngine.class)
                            .evaluate(createTransfer());

                    assertEquals(RiskDecision.REVIEW, result.decision());
                    assertTrue(
                            result.factors().contains(
                                    "FRAUD_CONFIRMATION_UNAVAILABLE"
                            )
                    );
                });
    }

    @Test
    void shouldUpdateReviewThresholdWithoutRecreatingRiskEngine() {
        contextRunner
                .withBean(
                        TransactionDataProvider.class,
                        () -> transfer -> createHistory()
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    FinGuardSettingsManager manager =
                            context.getBean(FinGuardSettingsManager.class);

                    RiskEngine engine = context.getBean(RiskEngine.class);

                    assertEquals(
                            RiskDecision.APPROVED,
                            engine.evaluate(createTransfer()).decision()
                    );

                    manager.updateRuleThresholds(20, 80);

                    assertEquals(
                            RiskDecision.REVIEW,
                            engine.evaluate(createTransfer()).decision()
                    );

                    assertSame(engine, context.getBean(RiskEngine.class));
                });
    }

    @Test
    void shouldUpdateRejectThresholdWithoutRecreatingRiskEngine() {
        contextRunner
                .withBean(
                        TransactionDataProvider.class,
                        () -> transfer -> createHistory()
                )
                .withBean(
                        ConfirmedFraudProvider.class,
                        () -> transfer -> true
                )
                .withPropertyValues(
                        "finguard.rules.unusual-amount.weight=90"
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    FinGuardSettingsManager manager =
                            context.getBean(FinGuardSettingsManager.class);

                    RiskEngine engine = context.getBean(RiskEngine.class);

                    assertEquals(
                            RiskDecision.REJECTED,
                            engine.evaluate(createTransfer()).decision()
                    );

                    manager.updateRuleThresholds(40, 95);

                    assertEquals(
                            RiskDecision.REVIEW,
                            engine.evaluate(createTransfer()).decision()
                    );

                    assertSame(engine, context.getBean(RiskEngine.class));
                });
    }

    @Test
    void shouldRegisterSettingsManagerWithoutNeuralModel() {
        contextRunner
                .withBean(
                        TransactionDataProvider.class,
                        () -> transfer -> createHistory()
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    FinGuardSettingsManager manager =
                            context.getBean(FinGuardSettingsManager.class);

                    assertTrue(
                            manager.snapshot()
                                    .neuralReviewThresholdLogit()
                                    .isEmpty()
                    );

                    RiskEngine engine = context.getBean(RiskEngine.class);

                    manager.updateRuleThresholds(20, 80);

                    assertEquals(
                            RiskDecision.REVIEW,
                            engine.evaluate(createTransfer()).decision()
                    );
                });
    }

    @Test
    void shouldCollectMetricsWithStandardRiskEngine() {
        contextRunner
                .withBean(TransactionDataProvider.class,
                        () -> transfer -> createHistory())
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    RiskMetrics metrics = context.getBean(RiskMetrics.class);
                    context.getBean(RiskEngine.class).evaluate(createTransfer());

                    assertEquals(1, metrics.snapshot().total());
                    assertEquals(1, metrics.snapshot().approved());
                    assertEquals(0, metrics.snapshot().failed());
                });
    }

    @Test
    void shouldAllowDisablingMonitoring() {
        contextRunner
                .withBean(TransactionDataProvider.class,
                        () -> transfer -> createHistory())
                .withPropertyValues("finguard.monitoring.enabled=false")
                .run(context -> {
                    assertNull(context.getStartupFailure());
                    assertEquals(0, context.getBeansOfType(RiskMetrics.class).size());

                    RiskAssessment result = context.getBean(RiskEngine.class)
                            .evaluate(createTransfer());

                    assertEquals(RiskDecision.APPROVED, result.decision());
                });
    }

    @Test
    void shouldPublishMetricsThroughBankSink() {
        AtomicReference<RiskMetrics.Snapshot> received =
                new AtomicReference<>();

        contextRunner
                .withBean(
                        TransactionDataProvider.class,
                        () -> transfer -> createHistory()
                )
                .withBean(
                        RiskMetricsSink.class,
                        () -> received::set
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    RiskEngine engine = context.getBean(RiskEngine.class);
                    RiskMetricsPublisher publisher =
                            context.getBean(RiskMetricsPublisher.class);

                    engine.evaluate(createTransfer());

                    assertNull(received.get());

                    RiskMetrics.Snapshot result = publisher.publish();

                    assertSame(result, received.get());
                    assertEquals(1L, result.total());
                    assertEquals(1L, result.approved());
                });
    }

    @Test
    void shouldWorkWithoutMetricsSink() {
        contextRunner
                .withBean(
                        TransactionDataProvider.class,
                        () -> transfer -> createHistory()
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    assertNotNull(context.getBean(RiskMetrics.class));
                    assertTrue(
                            context.getBeansOfType(
                                    RiskMetricsPublisher.class
                            ).isEmpty()
                    );

                    assertEquals(
                            RiskDecision.APPROVED,
                            context.getBean(RiskEngine.class)
                                    .evaluate(createTransfer()).decision()
                    );
                });
    }

    @Test
    void shouldNotCreatePublisherWhenMonitoringDisabled() {
        contextRunner
                .withBean(
                        TransactionDataProvider.class,
                        () -> transfer -> createHistory()
                )
                .withBean(
                        RiskMetricsSink.class,
                        () -> snapshot -> {
                        }
                )
                .withPropertyValues(
                        "finguard.monitoring.enabled=false"
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    assertTrue(
                            context.getBeansOfType(
                                    RiskMetricsPublisher.class
                            ).isEmpty()
                    );

                    assertTrue(
                            context.getBeansOfType(
                                    RiskMetrics.class
                            ).isEmpty()
                    );

                    assertNotNull(context.getBean(RiskEngine.class));
                });
    }

    @Test
    void shouldPreserveAssessmentWithMonitoringEnabledAndDisabled() {
        TransactionHistorySnapshot history = createHistory();
        TransferData transfer = createTransfer();
        RiskAssessment[] results = new RiskAssessment[2];

        contextRunner
                .withBean(
                        TransactionDataProvider.class,
                        () -> ignored -> history
                )
                .withPropertyValues("finguard.monitoring.enabled=false")
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    results[0] = context.getBean(RiskEngine.class)
                            .evaluate(transfer);

                    assertTrue(context.getBeansOfType(RiskMetrics.class).isEmpty());
                });

        contextRunner
                .withBean(
                        TransactionDataProvider.class,
                        () -> ignored -> history
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    results[1] = context.getBean(RiskEngine.class)
                            .evaluate(transfer);

                    assertEquals(
                            1L,
                            context.getBean(RiskMetrics.class).snapshot().total()
                    );
                });

        assertNotNull(results[0]);
        assertEquals(results[0], results[1]);
        assertEquals(results[0].trace(), results[1].trace());
    }

    @Test
    void shouldNotDependOnBankMonitoringAvailability() {
        AtomicInteger publishCalls = new AtomicInteger();
        TransactionHistorySnapshot history = createHistory();

        contextRunner
                .withBean(
                        TransactionDataProvider.class,
                        () -> ignored -> history
                )
                .withBean(
                        RiskMetricsSink.class,
                        () -> snapshot -> {
                            publishCalls.incrementAndGet();
                            throw new IllegalStateException(
                                    "Bank monitoring unavailable"
                            );
                        }
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    RiskEngine engine = context.getBean(RiskEngine.class);
                    RiskMetricsPublisher publisher =
                            context.getBean(RiskMetricsPublisher.class);

                    for (int i = 0; i < 100; i++) {
                        RiskAssessment result = engine.evaluate(createTransfer());
                        assertEquals(RiskDecision.APPROVED, result.decision());
                    }

                    assertEquals(0, publishCalls.get());
                    assertEquals(
                            100L,
                            context.getBean(RiskMetrics.class).snapshot().total()
                    );

                    assertThrows(
                            IllegalStateException.class,
                            publisher::publish
                    );

                    assertEquals(1, publishCalls.get());
                    assertEquals(
                            100L,
                            context.getBean(RiskMetrics.class).snapshot().total()
                    );

                    assertEquals(
                            RiskDecision.APPROVED,
                            engine.evaluate(createTransfer()).decision()
                    );
                });
    }

    @Test
    void shouldMeasureMonitoringOverhead() {
        TransactionHistorySnapshot history = createHistory();
        TransferData transfer = createTransfer();
        TimingResult[] results = new TimingResult[2];

        contextRunner
                .withBean(
                        TransactionDataProvider.class,
                        () -> ignored -> history
                )
                .withPropertyValues("finguard.monitoring.enabled=false")
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    results[0] = measure(
                            context.getBean(RiskEngine.class),
                            transfer
                    );
                });

        contextRunner
                .withBean(
                        TransactionDataProvider.class,
                        () -> ignored -> history
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    results[1] = measure(
                            context.getBean(RiskEngine.class),
                            transfer
                    );

                    RiskMetrics.Snapshot metrics =
                            context.getBean(RiskMetrics.class).snapshot();

                    assertEquals(12_000L, metrics.total());
                    assertEquals(12_000L, metrics.approved());
                    assertEquals(0L, metrics.failed());
                });

        assertEquals(results[0].checksum(), results[1].checksum());

        System.out.printf(
                Locale.ROOT,
                "Monitoring OFF: avg=%.4f ms, p95=%.4f ms, p99=%.4f ms%n",
                results[0].averageMillis(),
                results[0].p95Millis(),
                results[0].p99Millis()
        );

        System.out.printf(
                Locale.ROOT,
                "Monitoring ON:  avg=%.4f ms, p95=%.4f ms, p99=%.4f ms%n",
                results[1].averageMillis(),
                results[1].p95Millis(),
                results[1].p99Millis()
        );

        System.out.printf(
                Locale.ROOT,
                "Observed average difference: %+.4f ms%n",
                results[1].averageMillis() - results[0].averageMillis()
        );
    }

    private TransferData createTransfer() {
        return new TransferData(
                1042L,
                125L,
                458L,
                new BigDecimal("30000.00"),
                "RUB",
                timestamp
        );
    }

    private TransactionHistorySnapshot createHistory() {
        AmountStatistics amount = new AmountStatistics(
                Optional.of(new BigDecimal("5000.00")),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty()
        );

        VelocityStatistics velocity = new VelocityStatistics(
                OptionalLong.of(1),
                OptionalLong.of(2),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.of(120),
                Optional.empty(),
                OptionalLong.empty(),
                OptionalLong.of(0),
                OptionalLong.empty(),
                OptionalLong.empty()
        );

        RecipientStatistics recipient = new RecipientStatistics(
                OptionalLong.of(2),
                Optional.empty(),
                OptionalLong.empty(),
                OptionalDouble.empty(),
                OptionalLong.empty()
        );

        ActivityStatistics activity = new ActivityStatistics(
                OptionalLong.empty(),
                OptionalLong.of(30),
                OptionalDouble.empty(),
                OptionalDouble.empty(),
                OptionalDouble.of(0.2),
                OptionalDouble.empty(),
                OptionalLong.empty()
        );

        return new TransactionHistorySnapshot(
                amount,
                velocity,
                recipient,
                activity,
                RiskHistoryStatistics.empty(),
                timestamp
        );
    }

    private static TimingResult measure(
            RiskEngine engine,
            TransferData transfer
    ) {
        int warmup = 2_000;
        int samples = 10_000;
        long checksum = 0L;

        for (int i = 0; i < warmup; i++) {
            checksum += engine.evaluate(transfer).riskScore();
        }

        long[] durations = new long[samples];
        long totalNanos = 0L;

        for (int i = 0; i < samples; i++) {
            long started = System.nanoTime();

            RiskAssessment result = engine.evaluate(transfer);

            long duration = System.nanoTime() - started;
            durations[i] = duration;
            totalNanos += duration;
            checksum += result.riskScore();
        }

        Arrays.sort(durations);

        double average = (double) totalNanos / samples / 1_000_000.0;
        double p95 = durations[(int) Math.ceil(samples * 0.95) - 1]
                / 1_000_000.0;
        double p99 = durations[(int) Math.ceil(samples * 0.99) - 1]
                / 1_000_000.0;

        return new TimingResult(average, p95, p99, checksum);
    }

    private record TimingResult(
            double averageMillis,
            double p95Millis,
            double p99Millis,
            long checksum
    ) {
    }
}