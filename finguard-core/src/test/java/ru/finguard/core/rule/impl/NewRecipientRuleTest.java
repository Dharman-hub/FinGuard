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

class NewRecipientRuleTest {

    private final NewRecipientRule rule = new NewRecipientRule();
    private final Instant timestamp = Instant.parse("2026-10-24T10:00:00Z");

    @Test
    void shouldPassForKnownRecipient() {
        TransactionHistorySnapshot history = createHistory(
                OptionalLong.of(5),
                OptionalLong.of(10)
        );

        RuleResult result = rule.evaluate(createTransfer(), history);

        assertEquals(RuleStatus.PASSED, result.status());
    }

    @Test
    void shouldPassForFirstNewRecipient() {
        TransactionHistorySnapshot history = createHistory(
                OptionalLong.of(0),
                OptionalLong.of(0)
        );

        RuleResult result = rule.evaluate(createTransfer(), history);

        assertEquals(RuleStatus.PASSED, result.status());
    }

    @Test
    void shouldPassWithinRecipientLimit() {
        TransactionHistorySnapshot history = createHistory(
                OptionalLong.of(0),
                OptionalLong.of(2)
        );

        RuleResult result = rule.evaluate(createTransfer(), history);

        assertEquals(RuleStatus.PASSED, result.status());
    }

    @Test
    void shouldTriggerWhenRecipientLimitIsExceeded() {
        TransactionHistorySnapshot history = createHistory(
                OptionalLong.of(0),
                OptionalLong.of(3)
        );

        RuleResult result = rule.evaluate(createTransfer(), history);

        assertEquals(RuleStatus.TRIGGERED, result.status());
        assertEquals("NEW_RECIPIENT", result.code());
    }

    @Test
    void shouldReturnInsufficientDataWhenRecipientHistoryIsMissing() {
        TransactionHistorySnapshot history = createHistory(
                OptionalLong.empty(),
                OptionalLong.of(5)
        );

        RuleResult result = rule.evaluate(createTransfer(), history);

        assertEquals(RuleStatus.INSUFFICIENT_DATA, result.status());
    }

    @Test
    void shouldReturnInsufficientDataWhenFrequencyIsMissing() {
        TransactionHistorySnapshot history = createHistory(
                OptionalLong.of(0),
                OptionalLong.empty()
        );

        RuleResult result = rule.evaluate(createTransfer(), history);

        assertEquals(RuleStatus.INSUFFICIENT_DATA, result.status());
    }

    @Test
    void shouldPassForKnownRecipientWithoutFrequencyData() {
        TransactionHistorySnapshot history = createHistory(
                OptionalLong.of(2),
                OptionalLong.empty()
        );

        RuleResult result = rule.evaluate(createTransfer(), history);

        assertEquals(RuleStatus.PASSED, result.status());
    }

    @Test
    void shouldSupportCustomLimit() {
        NewRecipientRule customRule = new NewRecipientRule(2);

        TransactionHistorySnapshot history = createHistory(
                OptionalLong.of(0),
                OptionalLong.of(2)
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
                () -> new NewRecipientRule(0)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> new NewRecipientRule(-1)
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
            OptionalLong completedTransfers,
            OptionalLong newRecipients
    ) {
        RecipientStatistics recipient = new RecipientStatistics(
                completedTransfers,
                Optional.empty(),
                OptionalLong.empty(),
                OptionalDouble.empty(),
                OptionalLong.empty()
        );

        VelocityStatistics velocity = new VelocityStatistics(
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                Optional.empty(),
                OptionalLong.empty(),
                newRecipients,
                OptionalLong.empty(),
                OptionalLong.empty()
        );

        return new TransactionHistorySnapshot(
                AmountStatistics.empty(),
                velocity,
                recipient,
                ActivityStatistics.empty(),
                RiskHistoryStatistics.empty(),
                timestamp
        );
    }
}
