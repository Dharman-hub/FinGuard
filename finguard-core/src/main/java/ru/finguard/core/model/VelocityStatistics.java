package ru.finguard.core.model;

import java.math.BigDecimal;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalLong;

public record VelocityStatistics(
        OptionalLong transfersLast5Minutes,
        OptionalLong transfersLastHour,
        OptionalLong transfersLast24Hours,
        OptionalLong transfersLast7Days,
        OptionalLong transfersLast30Days,
        Optional<BigDecimal> totalAmountLastHour,
        OptionalLong uniqueRecipientsLast24Hours,
        OptionalLong newRecipientsLast24Hours,
        OptionalLong secondsSinceLastTransfer,
        OptionalLong repeatedAmountsLastHour
) {
    public VelocityStatistics {
        validate(transfersLast5Minutes);
        validate(transfersLastHour);
        validate(transfersLast24Hours);
        validate(transfersLast7Days);
        validate(transfersLast30Days);
        validate(uniqueRecipientsLast24Hours);
        validate(newRecipientsLast24Hours);
        validate(secondsSinceLastTransfer);
        validate(repeatedAmountsLastHour);

        Objects.requireNonNull(totalAmountLastHour);

        totalAmountLastHour.ifPresent(amount -> {
            if (amount.signum() < 0) {
                throw new IllegalArgumentException("Total amount cannot be negative");
            }
        });
    }

    private static void validate(OptionalLong value) {
        Objects.requireNonNull(value);

        if (value.isPresent() && value.getAsLong() < 0) {
            throw new IllegalArgumentException("Velocity statistics cannot be negative");
        }
    }

    public static VelocityStatistics empty() {
        return new VelocityStatistics(
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                Optional.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty()
        );
    }
}
