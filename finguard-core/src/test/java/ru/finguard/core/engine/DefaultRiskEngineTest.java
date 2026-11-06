package ru.finguard.core.engine;

import org.junit.jupiter.api.Test;
import ru.finguard.core.decision.DecisionEngine;
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
import ru.finguard.core.rule.RuleEngine;
import ru.finguard.core.rule.impl.HighVelocityRule;
import ru.finguard.core.rule.impl.NewRecipientRule;
import ru.finguard.core.rule.impl.UnusualActivityRule;
import ru.finguard.core.rule.impl.UnusualAmountRule;
import ru.finguard.core.scoring.RiskScorer;
import ru.finguard.core.neural.NeuralReviewThreshold;
import ru.finguard.core.api.RiskCheckUnavailableException;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.List;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DefaultRiskEngineTest {

    private final Instant timestamp = Instant.parse("2026-10-24T10:00:00Z");

    @Test
    void shouldApproveLowRiskTransfer() {
        TransactionHistorySnapshot history = createHistory(
                5000, 1, 2, 2, 0, 0.2
        );

        DefaultRiskEngine engine = createEngine(transfer -> history);

        RiskAssessment result = engine.evaluate(createTransfer("10000"));

        assertEquals(RiskDecision.APPROVED, result.decision());
        assertEquals(0, result.riskScore());
        assertTrue(result.factors().isEmpty());
    }

    @Test
    void shouldReviewSuspiciousTransfer() {
        TransactionHistorySnapshot history = createHistory(
                5000, 5, 2, 2, 0, 0.2
        );

        DefaultRiskEngine engine = createEngine(transfer -> history);

        RiskAssessment result = engine.evaluate(createTransfer("20000"));

        assertEquals(RiskDecision.REVIEW, result.decision());
        assertEquals(60, result.riskScore());
        assertTrue(result.factors().contains("UNUSUAL_AMOUNT"));
        assertTrue(result.factors().contains("HIGH_VELOCITY"));
    }

    @Test
    void shouldRejectHighRiskTransfer() {
        TransactionHistorySnapshot history = createHistory(
                5000, 5, 2, 0, 3, 0.0
        );

        DefaultRiskEngine engine = createEngine(transfer -> history);

        RiskAssessment result = engine.evaluate(createTransfer("20000"));

        assertEquals(RiskDecision.REVIEW, result.decision());
        assertEquals(100, result.riskScore());
        assertEquals(4, result.factors().size());
    }

    @Test
    void shouldReviewWhenHistoryIsMissing() {
        DefaultRiskEngine engine = createEngine(
                transfer -> TransactionHistorySnapshot.empty(timestamp)
        );

        RiskAssessment result = engine.evaluate(createTransfer("10000"));

        assertEquals(RiskDecision.REVIEW, result.decision());
        assertEquals(0, result.riskScore());
        assertEquals(4, result.factors().size());
        assertTrue(result.factors().contains("INSUFFICIENT_DATA:UNUSUAL_AMOUNT"));
        assertTrue(result.factors().contains("INSUFFICIENT_DATA:HIGH_VELOCITY"));
    }

    @Test
    void shouldPropagateDataProviderException() {
        DefaultRiskEngine engine = createEngine(transfer -> {
            throw new IllegalStateException("Database unavailable");
        });

        assertThrows(
                RiskCheckUnavailableException.class,
                () -> engine.evaluate(createTransfer("10000"))
        );
    }

    @Test
    void shouldRejectNullHistory() {
        DefaultRiskEngine engine = createEngine(transfer -> null);

        assertThrows(
                RiskCheckUnavailableException.class,
                () -> engine.evaluate(createTransfer("10000"))
        );
    }

    @Test
    void shouldRejectNullTransfer() {
        DefaultRiskEngine engine = createEngine(
                transfer -> TransactionHistorySnapshot.empty(timestamp)
        );

        assertThrows(
                NullPointerException.class,
                () -> engine.evaluate(null)
        );
    }

    private DefaultRiskEngine createEngine(TransactionDataProvider provider) {
        RuleEngine ruleEngine = new RuleEngine(List.of(
                new UnusualAmountRule(),
                new HighVelocityRule(),
                new NewRecipientRule(),
                new UnusualActivityRule()
        ));

        return new DefaultRiskEngine(
                provider,
                ruleEngine,
                new RiskScorer(),
                new DecisionEngine(),
                null,
                new NeuralReviewThreshold(0.0f),
                null,
                null,
                null,
                null
        );
    }

    private TransferData createTransfer(String amount) {
        return new TransferData(
                1042L,
                125L,
                458L,
                new BigDecimal(amount),
                "RUB",
                timestamp
        );
    }

    private TransactionHistorySnapshot createHistory(
            long averageAmount,
            long transfers5m,
            long transfersHour,
            long completedToRecipient,
            long newRecipients24h,
            double hourShare
    ) {
        AmountStatistics amount = new AmountStatistics(
                Optional.of(BigDecimal.valueOf(averageAmount)),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty()
        );

        VelocityStatistics velocity = new VelocityStatistics(
                OptionalLong.of(transfers5m),
                OptionalLong.of(transfersHour),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.of(120),
                Optional.empty(),
                OptionalLong.empty(),
                OptionalLong.of(newRecipients24h),
                OptionalLong.empty(),
                OptionalLong.empty()
        );

        RecipientStatistics recipient = new RecipientStatistics(
                OptionalLong.of(completedToRecipient),
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
                OptionalDouble.of(hourShare),
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
}
