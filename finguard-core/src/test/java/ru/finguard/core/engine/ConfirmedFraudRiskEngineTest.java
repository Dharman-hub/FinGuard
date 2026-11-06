package ru.finguard.core.engine;

import org.junit.jupiter.api.Test;
import ru.finguard.core.api.RiskEngine;
import ru.finguard.core.api.RiskEngineBuilder;
import ru.finguard.core.model.RiskDecision;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.provider.ConfirmedFraudProvider;
import ru.finguard.core.rule.RuleResult;

import java.math.BigDecimal;
import java.time.Instant;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ConfirmedFraudRiskEngineTest {

    @Test
    void shouldRejectConfirmedHighRiskFraud() {
        RiskEngine engine = createEngine(true, transfer -> true);

        var result = engine.evaluate(transfer());

        assertEquals(RiskDecision.REJECTED, result.decision());
        assertEquals(90, result.riskScore());
        assertTrue(result.factors().contains("BANK_CONFIRMED_FRAUD"));
    }

    @Test
    void shouldReviewHighRiskWithoutConfirmation() {
        RiskEngine engine = createEngine(true, transfer -> false);

        var result = engine.evaluate(transfer());

        assertEquals(RiskDecision.REVIEW, result.decision());
        assertEquals(90, result.riskScore());
    }

    @Test
    void shouldReviewHighRiskWithoutProvider() {
        RiskEngine engine = createEngine(true, null);

        var result = engine.evaluate(transfer());

        assertEquals(RiskDecision.REVIEW, result.decision());
    }

    @Test
    void shouldReviewConfirmedFraudWithLowRuleScore() {
        RiskEngine engine = createEngine(false, transfer -> true);

        var result = engine.evaluate(transfer());

        assertEquals(RiskDecision.REVIEW, result.decision());
        assertEquals(0, result.riskScore());
        assertTrue(result.factors().contains("BANK_CONFIRMED_FRAUD"));
    }

    @Test
    void shouldReviewWhenConfirmationProviderFails() {
        RiskEngine engine = createEngine(false, transfer -> {
            throw new IllegalStateException("Bank evidence unavailable");
        });

        var result = engine.evaluate(transfer());

        assertEquals(RiskDecision.REVIEW, result.decision());
        assertTrue(
                result.factors().contains("FRAUD_CONFIRMATION_UNAVAILABLE")
        );
    }

    @Test
    void shouldApproveLowRiskWithoutFraudConfirmation() {
        RiskEngine engine = createEngine(false, transfer -> false);

        var result = engine.evaluate(transfer());

        assertEquals(RiskDecision.APPROVED, result.decision());
        assertEquals(0, result.riskScore());
    }

    private RiskEngine createEngine(
            boolean triggered,
            ConfirmedFraudProvider confirmationProvider
    ) {
        RiskEngineBuilder builder = new RiskEngineBuilder(
                transfer -> TransactionHistorySnapshot.empty(
                        transfer.timestamp()
                )
        )
                .removeRule("UNUSUAL_AMOUNT")
                .removeRule("HIGH_VELOCITY")
                .removeRule("NEW_RECIPIENT")
                .removeRule("UNUSUAL_ACTIVITY")
                .addRule(
                        "TEST_RULE",
                        (transfer, history) -> triggered
                                ? RuleResult.triggered(
                                        "TEST_RULE",
                                        "High risk detected"
                                )
                                : RuleResult.passed("TEST_RULE"),
                        90
                );

        if (confirmationProvider != null) {
            builder.confirmedFraudProvider(confirmationProvider);
        }

        return builder.build();
    }

    private TransferData transfer() {
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