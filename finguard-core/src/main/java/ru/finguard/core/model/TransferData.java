package ru.finguard.core.model;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.Objects;

public record TransferData(
        long transferId,
        long userId,
        long recipientId,
        BigDecimal amount,
        String currency,
        Instant timestamp
) {
    public TransferData {
        Objects.requireNonNull(amount);
        Objects.requireNonNull(currency);
        Objects.requireNonNull(timestamp);

        if (amount.signum() <= 0) {
            throw new IllegalArgumentException("Transfer amount must be positive");
        }
    }
}