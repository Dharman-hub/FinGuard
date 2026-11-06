package ru.finguard.core.monitoring;

import org.junit.jupiter.api.Test;
import ru.finguard.core.api.RiskEngine;
import ru.finguard.core.model.DecisionReason;
import ru.finguard.core.model.RiskAssessment;
import ru.finguard.core.model.RiskDecision;
import ru.finguard.core.model.TransferData;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

class MonitoringRiskEngineTest {

    @Test
    void shouldCountDecisionsAndFallbacks() {
        RiskMetrics metrics = new RiskMetrics();
        AtomicReference<RiskAssessment> next = new AtomicReference<>(
                assessment(RiskDecision.APPROVED, DecisionReason.LOW_RISK)
        );
        RiskEngine engine = new MonitoringRiskEngine(transfer -> next.get(), metrics);

        assertEquals(RiskDecision.APPROVED, engine.evaluate(transfer()).decision());

        next.set(assessment(RiskDecision.REVIEW, DecisionReason.DEPENDENCY_UNAVAILABLE));
        assertEquals(RiskDecision.REVIEW, engine.evaluate(transfer()).decision());

        next.set(assessment(RiskDecision.REJECTED, DecisionReason.BANK_CONFIRMED_FRAUD));
        assertEquals(RiskDecision.REJECTED, engine.evaluate(transfer()).decision());

        var result = metrics.snapshot();
        assertEquals(3, result.total());
        assertEquals(1, result.approved());
        assertEquals(1, result.review());
        assertEquals(1, result.rejected());
        assertEquals(1, result.dependencyUnavailable());
        assertEquals(0, result.failed());
        assertTrue(result.averageLatencyMillis() >= 0);
        assertTrue(result.maxLatencyMillis() >= 0);
    }

    @Test
    void shouldCountAndRethrowFailures() {
        RiskMetrics metrics = new RiskMetrics();
        IllegalStateException failure = new IllegalStateException("Provider unavailable");
        RiskEngine engine = new MonitoringRiskEngine(transfer -> {
            throw failure;
        }, metrics);

        assertSame(failure, assertThrows(IllegalStateException.class,
                () -> engine.evaluate(transfer())));
        assertEquals(1, metrics.snapshot().total());
        assertEquals(1, metrics.snapshot().failed());
        assertEquals(0, metrics.snapshot().review());
    }

    @Test
    void shouldStartWithZeroCounters() {
        var result = new RiskMetrics().snapshot();
        assertEquals(0, result.total());
        assertEquals(0, result.averageLatencyMillis());
        assertEquals(0, result.maxLatencyMillis());
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

    private static TransferData transfer() {
        return new TransferData(
                1L,
                2L,
                3L,
                BigDecimal.TEN,
                "RUB",
                Instant.parse("2026-10-31T12:00:00Z")
        );
    }
}