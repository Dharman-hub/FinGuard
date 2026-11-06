package ru.finguard.training.bank;

import ru.finguard.core.feature.V3InputNormalizer;

import java.time.Instant;
import java.util.Objects;

public record BankV3TrainingSample(
        String transferId,
        long userId,
        Instant eventTime,
        Instant confirmedAt,
        int label,
        float[] modelInput
) {
    public BankV3TrainingSample {
        if (transferId == null || transferId.isBlank()) {
            throw new IllegalArgumentException("Transfer ID is required");
        }

        Objects.requireNonNull(eventTime);
        Objects.requireNonNull(confirmedAt);
        Objects.requireNonNull(modelInput);

        if (confirmedAt.isBefore(eventTime)) {
            throw new IllegalArgumentException("Confirmation predates transfer");
        }

        if (label != 0 && label != 1) {
            throw new IllegalArgumentException("Label must be 0 or 1");
        }

        if (modelInput.length != V3InputNormalizer.INPUT_SIZE) {
            throw new IllegalArgumentException("Expected 70 model inputs");
        }

        modelInput = modelInput.clone();

        for (int i = 0; i < V3InputNormalizer.FEATURE_COUNT; i++) {
            float value = modelInput[i];
            float mask = modelInput[V3InputNormalizer.FEATURE_COUNT + i];

            if (!Float.isFinite(value) || (mask != 0.0f && mask != 1.0f)) {
                throw new IllegalArgumentException("Invalid model input at index " + i);
            }

            if (mask == 0.0f && value != 0.0f) {
                throw new IllegalArgumentException("Missing feature must have zero value");
            }
        }

        float iban = modelInput[34];
        float ibanMask = modelInput[69];

        if ((iban != 0.0f && iban != 1.0f)
                || (ibanMask == 0.0f && iban != 0.0f)) {
            throw new IllegalArgumentException("Invalid IBAN feature");
        }
    }

    @Override
    public float[] modelInput() {
        return modelInput.clone();
    }
}