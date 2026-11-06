package ru.finguard.training.dataset.synsepa;

import ru.finguard.core.feature.FeatureName;

import java.io.BufferedInputStream;
import java.io.BufferedWriter;
import java.io.DataInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

public final class SynSepV3NormalizationApplication {

    private static final int MAGIC = 0x46475633;
    private static final int VERSION = 1;
    private static final int FEATURE_COUNT = 35;
    private static final int INPUT_SIZE = 70;
    private static final int HEADER_SIZE = 20;
    private static final int RECORD_SIZE = 294;

    private static final Path TRAIN = Path.of(
            "data", "synsepa", "experiment-v3", "train.fgbin"
    );

    private static final Path OUTPUT = Path.of(
            "models", "synsepa-v3", "normalization.csv"
    );

    private SynSepV3NormalizationApplication() {
    }

    public static void main(String[] args) throws IOException {
        FeatureName[] oldFeatures = FeatureName.values();

        if (oldFeatures.length != 34) {
            throw new IllegalStateException("Expected 34 original features");
        }

        long[] counts = new long[FEATURE_COUNT];
        double[] means = new double[FEATURE_COUNT];
        double[] sumsOfSquares = new double[FEATURE_COUNT];
        long transactions;

        try (DataInputStream input = new DataInputStream(
                new BufferedInputStream(Files.newInputStream(TRAIN), 1024 * 1024)
        )) {
            int magic = input.readInt();
            int version = input.readInt();
            int inputSize = input.readInt();
            transactions = input.readLong();

            if (magic != MAGIC || version != VERSION
                    || inputSize != INPUT_SIZE || transactions < 1) {
                throw new IOException("Unexpected SynSEPA v3 file header");
            }

            long expectedSize = Math.addExact(
                    HEADER_SIZE,
                    Math.multiplyExact(transactions, RECORD_SIZE)
            );

            if (Files.size(TRAIN) != expectedSize) {
                throw new IOException("Incorrect train.fgbin file size");
            }

            float[] values = new float[FEATURE_COUNT];

            for (long row = 0; row < transactions; row++) {
                int accountId = input.readInt();
                long timestamp = input.readLong();
                int label = input.readUnsignedByte();
                int type = input.readUnsignedByte();

                if (accountId < 0 || accountId >= 10_000 || timestamp <= 0
                        || (label != 0 && label != 1)
                        || (label == 0 && type != 0)
                        || (label == 1 && (type < 1 || type > 3))) {
                    throw new IOException("Invalid transaction metadata at row " + row);
                }

                for (int i = 0; i < FEATURE_COUNT; i++) {
                    values[i] = input.readFloat();

                    if (!Float.isFinite(values[i])) {
                        throw new IOException("Invalid feature value at row " + row);
                    }
                }

                for (int i = 0; i < FEATURE_COUNT; i++) {
                    float mask = input.readFloat();

                    if (mask != 0.0f && mask != 1.0f) {
                        throw new IOException("Invalid feature mask at row " + row);
                    }

                    if (i == 34 && ((values[i] != 0.0f && values[i] != 1.0f)
                            || (mask == 0.0f && values[i] != 0.0f))) {
                        throw new IOException("Invalid IBAN feature at row " + row);
                    }

                    if (mask == 0.0f) {
                        continue;
                    }

                    counts[i]++;
                    double delta = values[i] - means[i];
                    means[i] += delta / counts[i];
                    sumsOfSquares[i] += delta * (values[i] - means[i]);
                }
            }

            if (input.read() != -1) {
                throw new IOException("Extra bytes after training records");
            }
        }

        if (transactions != 1_283_654) {
            throw new IOException("Unexpected number of TRAIN transfers: " + transactions);
        }

        if (counts[34] != 163_911) {
            throw new IOException(
                    "Unexpected available IBAN features: " + counts[34]
            );
        }

        Files.createDirectories(OUTPUT.getParent());

        try (BufferedWriter writer = Files.newBufferedWriter(
                OUTPUT, StandardCharsets.UTF_8
        )) {
            writer.write("feature,mean,std,observed_train_count");
            writer.newLine();

            for (int i = 0; i < FEATURE_COUNT; i++) {
                String name = i < 34
                        ? oldFeatures[i].name()
                        : "NEW_SUPPLIER_IBAN_AFTER_HISTORY";

                double mean = counts[i] == 0 ? 0.0 : means[i];
                double std = counts[i] == 0 ? 1.0
                        : Math.sqrt(Math.max(0.0, sumsOfSquares[i] / counts[i]));

                if (i == 34) {
                    mean = 0.0;
                    std = 1.0;
                } else if (std < 1.0e-12) {
                    std = 1.0;
                }

                if (!Double.isFinite(mean) || !Double.isFinite(std)) {
                    throw new IOException("Invalid normalization for " + name);
                }

                writer.write(name + "," + mean + "," + std + "," + counts[i]);
                writer.newLine();

                System.out.printf(
                        Locale.US,
                        "%-43s count=%-10d mean=%14.6f std=%14.6f%n",
                        name, counts[i], mean, std
                );
            }
        }

        System.out.println();
        System.out.println("========== SYNSEPA V3 NORMALIZATION ==========");
        System.out.println("Train transfers: " + transactions);
        System.out.println("Features: 35");
        System.out.println("Masks: 35 (unchanged)");
        System.out.println("Validation and Test: NOT USED");
        System.out.println("Saved: " + OUTPUT.toAbsolutePath());
        System.out.println("SYNSEPA V3 NORMALIZATION SUCCESSFUL");
    }
}