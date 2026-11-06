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
import java.util.OptionalDouble;
import java.util.OptionalLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class UnusualActivityRuleTest {

    private final UnusualActivityRule rule = new UnusualActivityRule();
    private final Instant timestamp = Instant.parse("2026-10-24T10:00:00Z");

    @Test
    void shouldTriggerForUnusualHour() {
        TransactionHistorySnapshot history = createHistory(
                OptionalLong.of(120),
                OptionalLong.of(30),
                OptionalDouble.of(0.005)
        );

        RuleResult result = rule.evaluate(createTransfer(), history);

        assertEquals(RuleStatus.TRIGGERED, result.status());
        assertEquals("UNUSUAL_ACTIVITY", result.code());
    }

    @Test
    void shouldTriggerWhenUserNeverTransferredDuringCurrentHour() {
        TransactionHistorySnapshot history = createHistory(
                OptionalLong.of(120),
                OptionalLong.of(30),
                OptionalDouble.of(0.0)
        );

        RuleResult result = rule.evaluate(createTransfer(), history);

        assertEquals(RuleStatus.TRIGGERED, result.status());
    }

    @Test
    void shouldPassForTypicalHour() {
        TransactionHistorySnapshot history = createHistory(
                OptionalLong.of(120),
                OptionalLong.of(30),
                OptionalDouble.of(0.15)
        );

        RuleResult result = rule.evaluate(createTransfer(), history);

        assertEquals(RuleStatus.PASSED, result.status());
    }

    @Test
    void shouldPassWhenShareEqualsThreshold() {
        TransactionHistorySnapshot history = createHistory(
                OptionalLong.of(120),
                OptionalLong.of(30),
                OptionalDouble.of(0.01)
        );

        RuleResult result = rule.evaluate(createTransfer(), history);

        assertEquals(RuleStatus.PASSED, result.status());
    }

    @Test
    void shouldReturnInsufficientDataForSmallHistory() {
        TransactionHistorySnapshot history = createHistory(
                OptionalLong.of(15),
                OptionalLong.of(30),
                OptionalDouble.of(0.0)
        );

        RuleResult result = rule.evaluate(createTransfer(), history);

        assertEquals(RuleStatus.INSUFFICIENT_DATA, result.status());
    }

    @Test
    void shouldReturnInsufficientDataForShortHistoryPeriod() {
        TransactionHistorySnapshot history = createHistory(
                OptionalLong.of(120),
                OptionalLong.of(10),
                OptionalDouble.of(0.0)
        );

        RuleResult result = rule.evaluate(createTransfer(), history);

        assertEquals(RuleStatus.INSUFFICIENT_DATA, result.status());
    }

    @Test
    void shouldReturnInsufficientDataWhenStatisticsAreMissing() {
        TransactionHistorySnapshot history = TransactionHistorySnapshot.empty(timestamp);

        RuleResult result = rule.evaluate(createTransfer(), history);

        assertEquals(RuleStatus.INSUFFICIENT_DATA, result.status());
    }

    @Test
    void shouldSupportCustomThresholds() {
        UnusualActivityRule customRule = new UnusualActivityRule(30, 0.05);

        TransactionHistorySnapshot history = createHistory(
                OptionalLong.of(40),
                OptionalLong.of(30),
                OptionalDouble.of(0.02)
        );

        assertEquals(
                RuleStatus.INSUFFICIENT_DATA,
                rule.evaluate(createTransfer(), history).status()
        );

        assertEquals(
                RuleStatus.TRIGGERED,
                customRule.evaluate(createTransfer(), history).status()
        );
    }

    @Test
    void shouldRejectInvalidConfiguration() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new UnusualActivityRule(0, 0.01)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> new UnusualActivityRule(60, 0)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> new UnusualActivityRule(60, Double.NaN)
        );
    }

    private TransferData createTransfer() {
        return new TransferData(
                1042L,
                125L,
                458L,
                new BigDecimal("10000.00"),
                "RUB",
                timestamp
        );
    }

    private TransactionHistorySnapshot createHistory(
            OptionalLong transfers30d,
            OptionalLong historyDays,
            OptionalDouble hourShare
    ) {
        VelocityStatistics velocity = new VelocityStatistics(
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                transfers30d,
                Optional.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty()
        );

        ActivityStatistics activity = new ActivityStatistics(
                OptionalLong.empty(),
                historyDays,
                OptionalDouble.empty(),
                OptionalDouble.empty(),
                hourShare,
                OptionalDouble.empty(),
                OptionalLong.empty()
        );

        return new TransactionHistorySnapshot(
                AmountStatistics.empty(),
                velocity,
                RecipientStatistics.empty(),
                activity,
                RiskHistoryStatistics.empty(),
                timestamp
        );
    }
}
