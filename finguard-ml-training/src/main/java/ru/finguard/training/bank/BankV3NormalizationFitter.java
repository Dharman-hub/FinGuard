package ru.finguard.training.bank;

import ru.finguard.core.feature.FeatureName;
import ru.finguard.core.feature.V3InputNormalizer;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

public final class BankV3NormalizationFitter {

    private static final int FEATURE_COUNT = V3InputNormalizer.FEATURE_COUNT;

    private BankV3NormalizationFitter() {
    }

    public static void fit(Path trainFile, Path destination) throws IOException {
        Objects.requireNonNull(trainFile);
        Objects.requireNonNull(destination);

        long[] observed = new long[FEATURE_COUNT];
        double[] means = new double[FEATURE_COUNT];
        double[] squaredDifferences = new double[FEATURE_COUNT];
        long total = 0;

        try (BankV3DatasetBinary.Reader reader = BankV3DatasetBinary.openReader(trainFile)) {
            BankV3TrainingSample sample;
            while ((sample = reader.readNext()) != null) {
                total++;
                float[] input = sample.modelInput();

                for (int i = 0; i < FEATURE_COUNT - 1; i++) {
                    if (input[FEATURE_COUNT + i] == 0.0f) {
                        continue;
                    }

                    observed[i]++;
                    double delta = input[i] - means[i];
                    means[i] += delta / observed[i];
                    squaredDifferences[i] += delta * (input[i] - means[i]);
                }

                if (input[FEATURE_COUNT * 2 - 1] == 1.0f) {
                    observed[FEATURE_COUNT - 1]++;
                }
            }
        }

        if (total == 0) {
            throw new IllegalArgumentException("TRAIN is empty");
        }

        FeatureName[] names = FeatureName.values();

        if (names.length != FEATURE_COUNT - 1) {
            throw new IllegalStateException("Unexpected feature schema");
        }

        Path output = destination.toAbsolutePath();
        Files.createDirectories(output.getParent());

        try (BufferedWriter writer = Files.newBufferedWriter(output, StandardCharsets.UTF_8)) {
            writer.write("feature,mean,std,observed_train_count");
            writer.newLine();

            for (int i = 0; i < FEATURE_COUNT; i++) {
                double mean = observed[i] == 0 ? 0.0 : means[i];

                double variance = observed[i] == 0 ? 0.0
                        : squaredDifferences[i] / observed[i];

                double std = Math.sqrt(Math.max(0.0, variance));

                if (i == FEATURE_COUNT - 1) {
                    mean = 0.0;
                    std = 1.0;
                } else if (std < 1.0e-12) {
                    std = 1.0;
                }

                if (!Double.isFinite(mean)
                        || !Double.isFinite(std)
                        || std <= 0.0) {
                    throw new IllegalStateException(
                            "Invalid normalization for feature " + i
                    );
                }

                String name = i == FEATURE_COUNT - 1
                        ? "NEW_SUPPLIER_IBAN_AFTER_HISTORY"
                        : names[i].name();

                writer.write(
                        name + "," + mean + "," + std + "," + observed[i]
                );
                writer.newLine();
            }
        }
    }
}