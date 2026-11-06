package ru.finguard.core.model;

import java.util.Objects;
import java.util.OptionalLong;

public record RiskHistoryStatistics(
        OptionalLong declinedTransfers24h,
        OptionalLong failedTransfers24h,
        OptionalLong riskReviewTransfers30d
) {
    public RiskHistoryStatistics {
        validate(declinedTransfers24h);
        validate(failedTransfers24h);
        validate(riskReviewTransfers30d);
    }

    private static void validate(OptionalLong value) {
        Objects.requireNonNull(value);

        if (value.isPresent() && value.getAsLong() < 0) {
            throw new IllegalArgumentException("Risk history statistics cannot be negative");
        }
    }

    public static RiskHistoryStatistics empty() {
        return new RiskHistoryStatistics(
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty()
        );
    }
}
