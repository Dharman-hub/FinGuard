package ru.finguard.core.api;

public final class RiskCheckUnavailableException
        extends RuntimeException {

    public RiskCheckUnavailableException(
            String message,
            Throwable cause
    ) {
        super(message, cause);
    }
}