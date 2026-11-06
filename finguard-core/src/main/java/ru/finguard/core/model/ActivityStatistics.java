package ru.finguard.core.model;

import java.util.Objects;
import java.util.OptionalDouble;
import java.util.OptionalLong;

public record ActivityStatistics(
        OptionalLong completedTransfersTotal,
        OptionalLong historyDaysAvailable,
        OptionalDouble averageDailyTransfers30d,
        OptionalDouble nightTransfersShare30d,
        OptionalDouble currentHourTransfersShare30d,
        OptionalDouble weekendTransfersShare30d,
        OptionalLong secondsSinceLastCompletedTransfer
) {
    public ActivityStatistics {
        validate(completedTransfersTotal);
        validate(historyDaysAvailable);
        validate(secondsSinceLastCompletedTransfer);
        validateNonNegative(averageDailyTransfers30d);
        validateShare(nightTransfersShare30d);
        validateShare(currentHourTransfersShare30d);
        validateShare(weekendTransfersShare30d);
    }

    private static void validate(OptionalLong value) {
        Objects.requireNonNull(value);

        if (value.isPresent() && value.getAsLong() < 0) {
            throw new IllegalArgumentException("Activity statistics cannot be negative");
        }
    }

    private static void validateNonNegative(OptionalDouble value) {
        Objects.requireNonNull(value);

        if (value.isPresent()) {
            double number = value.getAsDouble();

            if (!Double.isFinite(number) || number < 0) {
                throw new IllegalArgumentException("Activity value cannot be negative");
            }
        }
    }

    private static void validateShare(OptionalDouble value) {
        Objects.requireNonNull(value);

        if (value.isPresent()) {
            double number = value.getAsDouble();

            if (!Double.isFinite(number) || number < 0 || number > 1) {
                throw new IllegalArgumentException("Activity share must be between 0 and 1");
            }
        }
    }

    public static ActivityStatistics empty() {
        return new ActivityStatistics(
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalDouble.empty(),
                OptionalDouble.empty(),
                OptionalDouble.empty(),
                OptionalDouble.empty(),
                OptionalLong.empty()
        );
    }
}
