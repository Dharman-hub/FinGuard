package ru.finguard.core.monitoring;

import org.junit.jupiter.api.Test;
import ru.finguard.core.api.RiskEngine;
import ru.finguard.core.model.DecisionReason;
import ru.finguard.core.model.RiskAssessment;
import ru.finguard.core.model.RiskDecision;
import ru.finguard.core.model.TransferData;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class RiskMetricsTest {

    @Test
    void shouldCalculateApproximateP95AndP99() {
        RiskMetrics metrics = new RiskMetrics();
        RiskAssessment approved = assessment(
                RiskDecision.APPROVED, DecisionReason.LOW_RISK
        );

        for (int i = 0; i < 90; i++) {
            metrics.recordSuccess(approved, 1_000_000L);
        }
        for (int i = 0; i < 7; i++) {
            metrics.recordSuccess(approved, 10_000_000L);
        }
        for (int i = 0; i < 3; i++) {
            metrics.recordSuccess(approved, 50_000_000L);
        }

        RiskMetrics.Snapshot result = metrics.snapshot();

        assertEquals(100L, result.total());
        assertEquals(3.1, result.averageLatencyMillis(), 0.000001);
        assertEquals(50.0, result.maxLatencyMillis());
        assertEquals(10.0, result.p95LatencyMillis());
        assertEquals(50.0, result.p99LatencyMillis());
    }

    @Test
    void shouldIncludeFailedEvaluationsInLatencyStatistics() {
        RiskMetrics metrics = new RiskMetrics();
        RiskAssessment approved = assessment(
                RiskDecision.APPROVED, DecisionReason.LOW_RISK
        );

        metrics.recordSuccess(approved, 1_000_000L);
        metrics.recordFailure(30_000_000L);

        RiskMetrics.Snapshot result = metrics.snapshot();

        assertEquals(2L, result.total());
        assertEquals(1L, result.failed());
        assertEquals(15.5, result.averageLatencyMillis());
        assertEquals(30.0, result.p95LatencyMillis());
        assertEquals(30.0, result.p99LatencyMillis());
    }

    @Test
    void shouldCountParallelEvaluationsWithoutLosses() throws Exception {
        RiskMetrics metrics = new RiskMetrics();
        AtomicInteger sequence = new AtomicInteger();

        RiskAssessment approved = assessment(
                RiskDecision.APPROVED, DecisionReason.LOW_RISK
        );
        RiskAssessment review = assessment(
                RiskDecision.REVIEW, DecisionReason.DEPENDENCY_UNAVAILABLE
        );
        RiskAssessment rejected = assessment(
                RiskDecision.REJECTED, DecisionReason.BANK_CONFIRMED_FRAUD
        );

        RiskEngine engine = new MonitoringRiskEngine(transfer -> {
            int index = sequence.getAndIncrement() % 4;

            return switch (index) {
                case 0 -> approved;
                case 1 -> review;
                case 2 -> rejected;
                default -> throw new IllegalStateException(
                        "Simulated failure"
                );
            };
        }, metrics);

        TransferData transfer = new TransferData(
                1L, 2L, 3L, BigDecimal.TEN, "RUB",
                Instant.parse("2026-10-31T12:00:00Z")
        );

        ExecutorService executor = Executors.newFixedThreadPool(8);
        CountDownLatch start = new CountDownLatch(1);
        List<Future<?>> futures = new ArrayList<>();

        try {
            for (int i = 0; i < 8; i++) {
                futures.add(executor.submit(() -> {
                    start.await();

                    for (int j = 0; j < 2000; j++) {
                        try {
                            engine.evaluate(transfer);
                        } catch (IllegalStateException ignored) {
                        }
                    }

                    return null;
                }));
            }

            start.countDown();

            for (Future<?> future : futures) {
                future.get(30, TimeUnit.SECONDS);
            }
        } finally {
            executor.shutdownNow();
        }

        RiskMetrics.Snapshot result = metrics.snapshot();

        assertEquals(16_000L, result.total());
        assertEquals(4_000L, result.approved());
        assertEquals(4_000L, result.review());
        assertEquals(4_000L, result.rejected());
        assertEquals(4_000L, result.failed());
        assertEquals(4_000L, result.dependencyUnavailable());
        assertTrue(result.p95LatencyMillis() >= 0.0);
        assertTrue(result.p99LatencyMillis() >= result.p95LatencyMillis());
    }

    private static RiskAssessment assessment(
            RiskDecision decision,
            DecisionReason reason
    ) {
        return new RiskAssessment(
                decision,
                0,
                List.of(),
                Optional.empty(),
                reason
        );
    }
}