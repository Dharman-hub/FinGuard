package ru.finguard.core.rule.impl;

import org.junit.jupiter.api.Test;
import ru.finguard.core.model.ActivityStatistics;
import ru.finguard.core.model.AmountStatistics;
import ru.finguard.core.model.RecipientStatistics;
import ru.finguard.core.model.RiskHistoryStatistics;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.model.VelocityStatistics;
import ru.finguard.core.rule.RuleResult;
import ru.finguard.core.rule.RuleStatus;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class UnusualAmountRuleTest {

    private final UnusualAmountRule rule = new UnusualAmountRule();
    private final Instant timestamp = Instant.parse("2026-10-24T10:00:00Z");

    @Test
    void shouldTriggerForUnusualAmount() {
        TransferData transfer = createTransfer("30000");
        TransactionHistorySnapshot history = createHistory(Optional.of(new BigDecimal("5000")));

        RuleResult result = rule.evaluate(transfer, history);

        assertEquals(RuleStatus.TRIGGERED, result.status());
        assertEquals("UNUSUAL_AMOUNT", result.code());
    }

    @Test
    void shouldPassForNormalAmount() {
        TransferData transfer = createTransfer("7000");
        TransactionHistorySnapshot history = createHistory(Optional.of(new BigDecimal("5000")));

        RuleResult result = rule.evaluate(transfer, history);

        assertEquals(RuleStatus.PASSED, result.status());
    }

    @Test
    void shouldPassWhenAmountEqualsThreshold() {
        TransferData transfer = createTransfer("15000");
        TransactionHistorySnapshot history = createHistory(Optional.of(new BigDecimal("5000")));

        RuleResult result = rule.evaluate(transfer, history);

        assertEquals(RuleStatus.PASSED, result.status());
    }

    @Test
    void shouldReturnInsufficientDataWhenHistoryIsMissing() {
        TransferData transfer = createTransfer("30000");
        TransactionHistorySnapshot history = TransactionHistorySnapshot.empty(timestamp);

        RuleResult result = rule.evaluate(transfer, history);

        assertEquals(RuleStatus.INSUFFICIENT_DATA, result.status());
    }

    @Test
    void shouldReturnInsufficientDataWhenAverageIsZero() {
        TransferData transfer = createTransfer("30000");
        TransactionHistorySnapshot history = createHistory(Optional.of(BigDecimal.ZERO));

        RuleResult result = rule.evaluate(transfer, history);

        assertEquals(RuleStatus.INSUFFICIENT_DATA, result.status());
    }

    @Test
    void shouldSupportCustomMultiplier() {
        UnusualAmountRule customRule = new UnusualAmountRule(new BigDecimal("5"));

        TransferData transfer = createTransfer("20000");
        TransactionHistorySnapshot history = createHistory(Optional.of(new BigDecimal("5000")));

        assertEquals(RuleStatus.TRIGGERED, rule.evaluate(transfer, history).status());
        assertEquals(RuleStatus.PASSED, customRule.evaluate(transfer, history).status());
    }

    @Test
    void shouldRejectInvalidMultiplier() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new UnusualAmountRule(BigDecimal.ONE)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> new UnusualAmountRule(BigDecimal.ZERO)
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

    private TransactionHistorySnapshot createHistory(Optional<BigDecimal> average) {
        AmountStatistics amountStatistics = new AmountStatistics(
                average,
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty()
        );

        return new TransactionHistorySnapshot(
                amountStatistics,
                VelocityStatistics.empty(),
                RecipientStatistics.empty(),
                ActivityStatistics.empty(),
                RiskHistoryStatistics.empty(),
                timestamp
        );
    }
}
