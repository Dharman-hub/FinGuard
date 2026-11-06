package ru.finguard.core.feature;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

public final class V3InputNormalizer {

    public static final int FEATURE_COUNT = 35;
    public static final int INPUT_SIZE = 70;

    private static final String HEADER = "feature,mean,std,observed_train_count";
    private static final String IBAN_FEATURE = "NEW_SUPPLIER_IBAN_AFTER_HISTORY";

    private final double[] means;
    private final double[] standardDeviations;

    private V3InputNormalizer(double[] means, double[] standardDeviations) {
        this.means = means;
        this.standardDeviations = standardDeviations;
    }

    public static V3InputNormalizer load(Path path) throws IOException {
        Objects.requireNonNull(path, "Normalization path is required");

        FeatureName[] featureNames = FeatureName.values();

        if (featureNames.length != FEATURE_COUNT - 1) {
            throw new IllegalStateException("Expected 34 base features");
        }

        double[] means = new double[FEATURE_COUNT];
        double[] deviations = new double[FEATURE_COUNT];

        try (BufferedReader reader = Files.newBufferedReader(
                path, StandardCharsets.UTF_8)) {

            if (!HEADER.equals(reader.readLine())) {
                throw new IOException("Unexpected v3 normalization header");
            }

            for (int i = 0; i < FEATURE_COUNT; i++) {
                String line = reader.readLine();

                if (line == null) {
                    throw new IOException("Incomplete v3 normalization profile");
                }

                String[] columns = line.split(",", -1);

                String expected = i == FEATURE_COUNT - 1
                        ? IBAN_FEATURE
                        : featureNames[i].name();

                if (columns.length != 4 || !expected.equals(columns[0])) {
                    throw new IOException("Invalid feature at index " + i);
                }

                long observed;

                try {
                    means[i] = Double.parseDouble(columns[1]);
                    deviations[i] = Double.parseDouble(columns[2]);
                    observed = Long.parseLong(columns[3]);
                } catch (NumberFormatException exception) {
                    throw new IOException(
                            "Invalid normalization row for " + expected,
                            exception
                    );
                }

                if (!Double.isFinite(means[i])
                        || !Double.isFinite(deviations[i])
                        || deviations[i] <= 0.0
                        || observed < 0) {
                    throw new IOException(
                            "Invalid normalization parameters for " + expected
                    );
                }
            }

            if (reader.readLine() != null) {
                throw new IOException("Unexpected extra normalization rows");
            }
        }

        if (means[34] != 0.0 || deviations[34] != 1.0) {
            throw new IOException(
                    "The supplier IBAN feature must not be normalized"
            );
        }

        return new V3InputNormalizer(means, deviations);
    }

    public float[] normalize(float[] rawInput) {
        Objects.requireNonNull(rawInput, "Model input is required");

        if (rawInput.length != INPUT_SIZE) {
            throw new IllegalArgumentException(
                    "Expected 70 inputs, got " + rawInput.length
            );
        }

        float[] result = rawInput.clone();

        for (int i = 0; i < FEATURE_COUNT; i++) {
            float value = rawInput[i];
            float mask = rawInput[FEATURE_COUNT + i];

            if (!Float.isFinite(value)
                    || (mask != 0.0f && mask != 1.0f)) {
                throw new IllegalArgumentException(
                        "Invalid input at feature " + i
                );
            }

            if (i == 34 && ((value != 0.0f && value != 1.0f)
                    || (mask == 0.0f && value != 0.0f))) {
                throw new IllegalArgumentException(
                        "Invalid supplier IBAN feature"
                );
            }

            if (mask == 0.0f) {
                result[i] = 0.0f;
                continue;
            }

            if (i == 34) {
                continue;
            }

            double normalized =
                    (value - means[i]) / standardDeviations[i];

            if (!Double.isFinite(normalized)
                    || Math.abs(normalized) > Float.MAX_VALUE) {
                throw new IllegalArgumentException(
                        "Normalized input out of range at feature " + i
                );
            }

            result[i] = (float) normalized;
        }

        return result;
    }
}