package ru.finguard.core.model;

public record NeuralRiskDetails(
        float logit,
        float reviewThresholdLogit
) {
    public NeuralRiskDetails {
        if (!Float.isFinite(logit)
                || !Float.isFinite(reviewThresholdLogit)) {
            throw new IllegalArgumentException(
                    "Neural risk details must be finite"
            );
        }
    }

    public boolean requiresReview() {
        return logit > reviewThresholdLogit;
    }
}