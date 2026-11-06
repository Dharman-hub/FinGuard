package ru.finguard.core.model;

import java.math.BigDecimal;
import java.util.Objects;
import java.util.Optional;

public record AmountStatistics(
        Optional<BigDecimal> averageAmount30d,
        Optional<BigDecimal> medianAmount30d,
        Optional<BigDecimal> maxAmount30d,
        Optional<BigDecimal> percentile95Amount30d,
        Optional<BigDecimal> standardDeviation30d,
        Optional<BigDecimal> totalAmount24h,
        Optional<BigDecimal> totalAmount7d
) {
    public AmountStatistics {
        validate(averageAmount30d);
        validate(medianAmount30d);
        validate(maxAmount30d);
        validate(percentile95Amount30d);
        validate(standardDeviation30d);
        validate(totalAmount24h);
        validate(totalAmount7d);
    }

    private static void validate(Optional<BigDecimal> value) {
        Objects.requireNonNull(value);

        value.ifPresent(amount -> {
            if (amount.signum() < 0) {
                throw new IllegalArgumentException("Amount statistics cannot be negative");
            }
        });
    }

    public static AmountStatistics empty() {
        return new AmountStatistics(
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty()
        );
    }
}
