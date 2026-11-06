package ru.finguard.core.model;

import java.math.BigDecimal;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalLong;

public record RecipientStatistics(
        OptionalLong completedTransfersToRecipient,
        Optional<BigDecimal> totalCompletedAmountToRecipient,
        OptionalLong secondsSinceLastTransferToRecipient,
        OptionalDouble recipientShare30d,
        OptionalLong uniqueRecipients30d
) {
    public RecipientStatistics {
        validate(completedTransfersToRecipient);
        validate(secondsSinceLastTransferToRecipient);
        validate(uniqueRecipients30d);

        Objects.requireNonNull(totalCompletedAmountToRecipient);
        totalCompletedAmountToRecipient.ifPresent(amount -> {
            if (amount.signum() < 0) {
                throw new IllegalArgumentException("Amount cannot be negative");
            }
        });

        Objects.requireNonNull(recipientShare30d);
        if (recipientShare30d.isPresent()) {
            double value = recipientShare30d.getAsDouble();

            if (!Double.isFinite(value) || value < 0 || value > 1) {
                throw new IllegalArgumentException("Recipient share must be between 0 and 1");
            }
        }
    }

    private static void validate(OptionalLong value) {
        Objects.requireNonNull(value);

        if (value.isPresent() && value.getAsLong() < 0) {
            throw new IllegalArgumentException("Recipient statistics cannot be negative");
        }
    }

    public static RecipientStatistics empty() {
        return new RecipientStatistics(
                OptionalLong.empty(),
                Optional.empty(),
                OptionalLong.empty(),
                OptionalDouble.empty(),
                OptionalLong.empty()
        );
    }
}
