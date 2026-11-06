package ru.finguard.starter.config;

public enum NeuralReviewMode {

    LOW_ALERTS(5.048477f),
    BALANCED(3.7777753f),
    HIGH_SENSITIVITY(1.5145587f),
    CUSTOM(Float.NaN);

    private final float thresholdLogit;

    NeuralReviewMode(float thresholdLogit) {
        this.thresholdLogit = thresholdLogit;
    }

    public float thresholdLogit() {
        if (this == CUSTOM) {
            throw new IllegalStateException(
                    "CUSTOM mode requires an explicit threshold"
            );
        }

        return thresholdLogit;
    }
}