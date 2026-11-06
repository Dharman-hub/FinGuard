package ru.finguard.core.feature;

import java.util.Objects;

public record FeatureVector(
        float[] values,
        float[] presenceMask
) {
    public FeatureVector {
        Objects.requireNonNull(values);
        Objects.requireNonNull(presenceMask);

        int featureCount = FeatureName.values().length;

        if (values.length != featureCount || presenceMask.length != featureCount) {
            throw new IllegalArgumentException("Invalid feature vector length");
        }

        values = values.clone();
        presenceMask = presenceMask.clone();

        for (int i = 0; i < featureCount; i++) {
            if (!Float.isFinite(values[i])) {
                throw new IllegalArgumentException("Feature values must be finite");
            }

            if (presenceMask[i] != 0.0f && presenceMask[i] != 1.0f) {
                throw new IllegalArgumentException("Presence mask must contain 0 or 1");
            }
        }
    }

    @Override
    public float[] values() {
        return values.clone();
    }

    @Override
    public float[] presenceMask() {
        return presenceMask.clone();
    }

    public float[] toModelInput() {
        float[] input = new float[values.length + presenceMask.length];

        System.arraycopy(values, 0, input, 0, values.length);
        System.arraycopy(presenceMask, 0, input, values.length, presenceMask.length);

        return input;
    }
}
