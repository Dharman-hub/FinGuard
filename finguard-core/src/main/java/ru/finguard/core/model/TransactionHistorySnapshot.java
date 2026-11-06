package ru.finguard.core.model;

import java.time.Instant;
import java.util.Objects;

public record TransactionHistorySnapshot(
        AmountStatistics amountStatistics,
        VelocityStatistics velocityStatistics,
        RecipientStatistics recipientStatistics,
        ActivityStatistics activityStatistics,
        RiskHistoryStatistics riskHistoryStatistics,
        Instant calculatedAt
) {
    public TransactionHistorySnapshot {
        Objects.requireNonNull(amountStatistics);
        Objects.requireNonNull(velocityStatistics);
        Objects.requireNonNull(recipientStatistics);
        Objects.requireNonNull(activityStatistics);
        Objects.requireNonNull(riskHistoryStatistics);
        Objects.requireNonNull(calculatedAt);
    }

    public static TransactionHistorySnapshot empty(Instant calculatedAt) {
        return new TransactionHistorySnapshot(
                AmountStatistics.empty(),
                VelocityStatistics.empty(),
                RecipientStatistics.empty(),
                ActivityStatistics.empty(),
                RiskHistoryStatistics.empty(),
                calculatedAt
        );
    }
}
