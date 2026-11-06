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
import java.util.OptionalLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class HighVelocityRuleTest {

    private final HighVelocityRule rule = new HighVelocityRule();
    private final Instant timestamp = Instant.parse("2026-10-24T10:00:00Z");

    @Test
    void shouldTriggerForHighFiveMinuteFrequency() {
        TransactionHistorySnapshot history = createHistory(
                OptionalLong.of(7),
                OptionalLong.of(10)
        );

        RuleResult result = rule.evaluate(createTransfer(), history);

        assertEquals(RuleStatus.TRIGGERED, result.status());
        assertEquals("HIGH_VELOCITY", result.code());
    }

    @Test
    void shouldTriggerForHighHourlyFrequency() {
        TransactionHistorySnapshot history = createHistory(
                OptionalLong.of(2),
                OptionalLong.of(25)
        );

        RuleResult result = rule.evaluate(createTransfer(), history);

        assertEquals(RuleStatus.TRIGGERED, result.status());
    }

    @Test
    void shouldPassForNormalFrequency() {
        TransactionHistorySnapshot history = createHistory(
                OptionalLong.of(2),
                OptionalLong.of(8)
        );

        RuleResult result = rule.evaluate(createTransfer(), history);

        assertEquals(RuleStatus.PASSED, result.status());
    }

    @Test
    void shouldPassWhenCurrentTransferReachesLimit() {
        TransactionHistorySnapshot history = createHistory(
                OptionalLong.of(4),
                OptionalLong.of(19)
        );

        RuleResult result = rule.evaluate(createTransfer(), history);

        assertEquals(RuleStatus.PASSED, result.status());
    }

    @Test
    void shouldReturnInsufficientDataWhenHistoryIsMissing() {
        TransactionHistorySnapshot history = TransactionHistorySnapshot.empty(timestamp);

        RuleResult result = rule.evaluate(createTransfer(), history);

        assertEquals(RuleStatus.INSUFFICIENT_DATA, result.status());
    }

    @Test
    void shouldWorkWithPartialHistory() {
        TransactionHistorySnapshot history = createHistory(
                OptionalLong.of(2),
                OptionalLong.empty()
        );

        RuleResult result = rule.evaluate(createTransfer(), history);

        assertEquals(RuleStatus.INSUFFICIENT_DATA, result.status());
    }

    @Test
    void shouldSupportCustomLimits() {
        HighVelocityRule customRule = new HighVelocityRule(3, 10);

        TransactionHistorySnapshot history = createHistory(
                OptionalLong.of(4),
                OptionalLong.of(8)
        );

        assertEquals(
                RuleStatus.PASSED,
                rule.evaluate(createTransfer(), history).status()
        );

        assertEquals(
                RuleStatus.TRIGGERED,
                customRule.evaluate(createTransfer(), history).status()
        );
    }

    @Test
    void shouldRejectInvalidLimits() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new HighVelocityRule(0, 10)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> new HighVelocityRule(5, -1)
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
            OptionalLong last5Minutes,
            OptionalLong lastHour
    ) {
        VelocityStatistics velocity = new VelocityStatistics(
                last5Minutes,
                lastHour,
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                Optional.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty()
        );

        return new TransactionHistorySnapshot(
                AmountStatistics.empty(),
                velocity,
                RecipientStatistics.empty(),
                ActivityStatistics.empty(),
                RiskHistoryStatistics.empty(),
                timestamp
        );
    }
}
