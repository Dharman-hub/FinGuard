package ru.finguard.core.neural;

import java.util.concurrent.atomic.AtomicReference;

public final class NeuralReviewThreshold {

    private final AtomicReference<Float> thresholdLogit;

    public NeuralReviewThreshold(float initialThresholdLogit) {
        validate(initialThresholdLogit);
        this.thresholdLogit = new AtomicReference<>(
                initialThresholdLogit
        );
    }

    public float get() {
        return thresholdLogit.get();
    }

    public void update(float newThresholdLogit) {
        validate(newThresholdLogit);
        thresholdLogit.set(newThresholdLogit);
    }

    private static void validate(float value) {
        if (!Float.isFinite(value)) {
            throw new IllegalArgumentException(
                    "Neural review threshold must be finite"
            );
        }
    }
}