package ru.finguard.core.rule.impl;

import ru.finguard.core.model.RecipientStatistics;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.rule.RiskRule;
import ru.finguard.core.rule.RuleResult;

import java.util.Objects;
import java.util.OptionalLong;

public final class NewRecipientRule implements RiskRule {

    private static final String CODE = "NEW_RECIPIENT";
    private static final long DEFAULT_MAX_NEW_RECIPIENTS_24H = 3;

    private final long maxNewRecipients24h;

    public NewRecipientRule() {
        this(DEFAULT_MAX_NEW_RECIPIENTS_24H);
    }

    public NewRecipientRule(long maxNewRecipients24h) {
        if (maxNewRecipients24h <= 0) {
            throw new IllegalArgumentException("New recipient limit must be positive");
        }

        this.maxNewRecipients24h = maxNewRecipients24h;
    }

    @Override
    public RuleResult evaluate(TransferData transfer, TransactionHistorySnapshot history) {
        Objects.requireNonNull(transfer);
        Objects.requireNonNull(history);

        RecipientStatistics recipient = history.recipientStatistics();

        OptionalLong completedTransfers = recipient.completedTransfersToRecipient();

        if (completedTransfers.isEmpty()) {
            return RuleResult.insufficientData(
                    CODE,
                    "Recipient transfer history is unavailable"
            );
        }

        if (completedTransfers.getAsLong() > 0) {
            return RuleResult.passed(CODE);
        }

        OptionalLong newRecipients = history.velocityStatistics().newRecipientsLast24Hours();

        if (newRecipients.isEmpty()) {
            return RuleResult.insufficientData(
                    CODE,
                    "New recipient frequency statistics are unavailable"
            );
        }

        if (newRecipients.getAsLong() >= maxNewRecipients24h) {
            return RuleResult.triggered(
                    CODE,
                    "Too many transfers to new recipients within 24 hours"
            );
        }

        return RuleResult.passed(CODE);
    }
}
