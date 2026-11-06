package ru.finguard.training.dataset.synsepa;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import ru.finguard.core.feature.FeatureName;
import ru.finguard.training.evaluation.FraudMetrics;

import java.io.BufferedReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Map;
import java.util.ArrayList;
import java.util.List;

public final class SynSepV3OnnxFinalTestApplication {

    private static final int INPUT_SIZE = 70;
    private static final int FEATURE_COUNT = 35;
    private static final int BATCH_SIZE = 512;

    private static final int[] BUDGETS = {1, 2, 5};

    private static final float[] THRESHOLDS = {
            5.048477f,
            3.7777753f,
            1.5145587f
    };

    private static final Path DATASET = Path.of(
            "data", "synsepa", "experiment-v3", "test.fgbin"
    );

    private static final Path MODEL = Path.of(
            "models", "synsepa-v3", "finguard-model.onnx"
    );

    private static final Path NORMALIZATION = Path.of(
            "models", "synsepa-v3", "normalization.csv"
    );

    private SynSepV3OnnxFinalTestApplication() {
    }

    public static void main(String[] args) throws Exception {
        System.out.println(
                "========== FINGUARD V3 ONNX FINAL TEST =========="
        );

        System.out.println("Model: synsepa-v3-70");
        System.out.println("Dataset: TEST");
        System.out.println("Thresholds: calibrated on VALIDATION");
        System.out.println("Test threshold tuning: DISABLED");

        if (!Files.isRegularFile(MODEL)
                || !Files.isRegularFile(DATASET)) {
            throw new IllegalStateException(
                    "ONNX model or TEST dataset not found"
            );
        }

        Profile profile = loadProfile();
        Statistics statistics = new Statistics();

        MissingHistoryScenario[] scenarios = {
                new MissingHistoryScenario(
                        "NO_VELOCITY_HISTORY",
                        FeatureName.TRANSFERS_LAST_5_MINUTES.ordinal(),
                        FeatureName.REPEATED_AMOUNTS_LAST_HOUR.ordinal(),
                        false
                ),
                new MissingHistoryScenario(
                        "NO_RECIPIENT_HISTORY_AND_IBAN",
                        FeatureName.COMPLETED_TRANSFERS_TO_RECIPIENT.ordinal(),
                        FeatureName.UNIQUE_RECIPIENTS_30D.ordinal(),
                        true
                ),
                new MissingHistoryScenario(
                        "CURRENT_AMOUNT_ONLY",
                        FeatureName.AMOUNT_TO_AVERAGE_RATIO.ordinal(),
                        FEATURE_COUNT - 1,
                        false
                )
        };

        OrtEnvironment environment = OrtEnvironment.getEnvironment();

        try (OrtSession.SessionOptions options =
                     new OrtSession.SessionOptions();
             OrtSession session = environment.createSession(
                     MODEL.toString(),
                     options
             )) {

            if (!session.getInputNames().contains("features")
                    || !session.getOutputNames().contains("risk_logit")) {
                throw new IllegalStateException(
                        "Unexpected ONNX model interface"
                );
            }

            var summary = SynSepV3BatchReader.forEachBatch(
                    DATASET,
                    INPUT_SIZE,
                    BATCH_SIZE,
                    batch -> {
                        evaluateBatch(
                                batch,
                                profile,
                                environment,
                                session,
                                statistics
                        );

                        for (MissingHistoryScenario scenario : scenarios) {
                            evaluateBatch(
                                    batch,
                                    profile,
                                    environment,
                                    session,
                                    scenario.statistics,
                                    scenario
                            );
                        }
                    }
            );

            if (summary.total() != 274_867
                    || summary.legitimate() != 274_444
                    || summary.invoice() != 136
                    || summary.impersonation() != 207
                    || summary.ceo() != 80
                    || statistics.total != summary.total()) {
                throw new IllegalStateException(
                        "Unexpected TEST distribution"
                );
            }
        }

        System.out.println();
        System.out.println(
                "========== INDEPENDENT TEST RESULTS =========="
        );

        System.out.println("Transfers: " + statistics.total);
        System.out.println("Legitimate: 274444");
        System.out.println("Fraud: 423");

        double averagePrecision = FraudMetrics.evaluate(
                statistics.predictions,
                0.5
        ).averagePrecision();

        System.out.printf(
                Locale.US,
                "Average Precision (AP): %.4f%%%n",
                averagePrecision * 100.0
        );

        for (int i = 0; i < BUDGETS.length; i++) {
            long detected = statistics.invoice[i]
                    + statistics.impersonation[i]
                    + statistics.ceo[i];

            long falsePositives = statistics.falsePositives[i];

            System.out.println();
            System.out.println(
                    "========== " + BUDGETS[i]
                            + " FP/1000 VALIDATION BUDGET =========="
            );

            System.out.println(
                    "Fixed ONNX threshold: "
                            + THRESHOLDS[i]
            );

            System.out.printf(
                    Locale.US,
                    "Actual TEST FP/1000: %.4f%n",
                    falsePositives * 1000.0 / 274_444
            );

            System.out.printf(
                    Locale.US,
                    "TP=%d FN=%d FP=%d%n",
                    detected,
                    423 - detected,
                    falsePositives
            );

            System.out.printf(
                    Locale.US,
                    "Recall: %.4f%%%n",
                    detected * 100.0 / 423
            );

            System.out.printf(
                    Locale.US,
                    "Precision: %.4f%%%n",
                    detected * 100.0 / (detected + falsePositives)
            );

            long falseNegatives = 423L - detected;
            long trueNegatives = 274_444L - falsePositives;

            double f1 = 2.0 * detected
                    / (2.0 * detected + falsePositives + falseNegatives);

            System.out.printf(
                    Locale.US,
                    "F1: %.4f%%%n",
                    f1 * 100.0
            );

            System.out.printf(
                    "TN=%d%n",
                    trueNegatives
            );

            System.out.printf(
                    "Invoice: %d / 136%n",
                    statistics.invoice[i]
            );

            System.out.printf(
                    "Impersonation: %d / 207%n",
                    statistics.impersonation[i]
            );

            System.out.printf(
                    "CEO fraud: %d / 80%n",
                    statistics.ceo[i]
            );
        }

        System.out.println();
        printMissingHistoryResults(statistics, scenarios);
        System.out.println(
                "FINGUARD V3 ONNX FINAL TEST COMPLETED"
        );
    }

    private static void evaluateBatch(
            SynSepV3BatchReader.Batch batch,
            Profile profile,
            OrtEnvironment environment,
            OrtSession session,
            Statistics statistics
    ) {
        evaluateBatch(
                batch,
                profile,
                environment,
                session,
                statistics,
                null
        );
    }

    private static void evaluateBatch(
            SynSepV3BatchReader.Batch batch,
            Profile profile,
            OrtEnvironment environment,
            OrtSession session,
            Statistics statistics,
            MissingHistoryScenario scenario
    ) {
        float[] inputs = normalize(batch, profile);

        if (scenario != null) {
            scenario.mask(inputs, batch.size());
        }

        ByteBuffer bytes = ByteBuffer.allocateDirect(
                inputs.length * Float.BYTES
        ).order(ByteOrder.nativeOrder());

        var buffer = bytes.asFloatBuffer();
        buffer.put(inputs);
        buffer.flip();

        try (OnnxTensor tensor = OnnxTensor.createTensor(
                environment,
                buffer,
                new long[]{batch.size(), INPUT_SIZE}
        );
             OrtSession.Result result = session.run(
                     Map.of("features", tensor)
             )) {

            float[][] predictions = (float[][]) result
                    .get(0)
                    .getValue();

            if (predictions.length != batch.size()) {
                throw new IllegalStateException(
                        "Unexpected prediction count"
                );
            }

            for (int i = 0; i < batch.size(); i++) {
                if (predictions[i].length != 1) {
                    throw new IllegalStateException(
                            "Unexpected prediction shape"
                    );
                }

                statistics.add(
                        predictions[i][0],
                        batch.labels()[i],
                        batch.fraudTypes()[i]
                );
            }

        } catch (OrtException exception) {
            throw new IllegalStateException(
                    "ONNX inference failed",
                    exception
            );
        }
    }

    private static float[] normalize(
            SynSepV3BatchReader.Batch batch,
            Profile profile
    ) {
        float[] inputs = batch.inputs().clone();

        for (int row = 0; row < batch.size(); row++) {
            int offset = row * INPUT_SIZE;

            for (int feature = 0; feature < FEATURE_COUNT; feature++) {
                int valueIndex = offset + feature;
                int maskIndex = offset + FEATURE_COUNT + feature;

                if (inputs[maskIndex] == 0.0f) {
                    inputs[valueIndex] = 0.0f;
                    continue;
                }

                if (feature == 34) {
                    continue;
                }

                double normalized =
                        (inputs[valueIndex] - profile.means[feature])
                                / profile.stds[feature];

                if (!Double.isFinite(normalized)
                        || Math.abs(normalized) > Float.MAX_VALUE) {
                    throw new IllegalStateException(
                            "Invalid normalized input"
                    );
                }

                inputs[valueIndex] = (float) normalized;
            }
        }

        return inputs;
    }

    private static void printMissingHistoryResults(
            Statistics baseline,
            MissingHistoryScenario[] scenarios
    ) {
        int index = 2;

        if (BUDGETS[index] != 5) {
            throw new IllegalStateException("Expected 5 FP/1000 budget");
        }

        long baselineDetected = baseline.invoice[index]
                + baseline.impersonation[index]
                + baseline.ceo[index];

        long baselineFp = baseline.falsePositives[index];

        System.out.println();
        System.out.println("========== MISSING HISTORY STRESS TEST ==========");
        System.out.println("Threshold: " + THRESHOLDS[index]);
        System.out.println("Threshold recalibration: DISABLED");
        System.out.println("Baseline TP=" + baselineDetected
                + ", FP=" + baselineFp);

        for (MissingHistoryScenario scenario : scenarios) {
            Statistics stats = scenario.statistics;

            if (stats.total != baseline.total) {
                throw new IllegalStateException(
                        "Incomplete stress test: " + scenario.name
                );
            }

            long tp = stats.invoice[index]
                    + stats.impersonation[index]
                    + stats.ceo[index];

            long fp = stats.falsePositives[index];
            long fn = 423 - tp;
            long tn = 274_444 - fp;

            System.out.println();
            System.out.println("Scenario: " + scenario.name);
            System.out.println(
                    "Affected transfers: " + scenario.affectedTransfers
            );
            System.out.println(
                    "Observed features hidden: "
                            + scenario.hiddenObservedFeatures
            );

            System.out.printf(
                    Locale.US,
                    "TP=%d FN=%d FP=%d TN=%d | Recall=%.4f%% | "
                            + "Precision=%.4f%% | FP/1000=%.4f%n",
                    tp,
                    fn,
                    fp,
                    tn,
                    tp * 100.0 / 423,
                    tp + fp == 0 ? 0.0 : tp * 100.0 / (tp + fp),
                    fp * 1000.0 / 274_444
            );

            System.out.printf(
                    "Compared with baseline: TP %+d, FP %+d%n",
                    tp - baselineDetected,
                    fp - baselineFp
            );

            System.out.printf(
                    "Invoice: %d/136 | Impersonation: %d/207 | CEO: %d/80%n",
                    stats.invoice[index],
                    stats.impersonation[index],
                    stats.ceo[index]
            );
        }
    }

    private static final class MissingHistoryScenario {

        private final String name;
        private final int firstFeature;
        private final int lastFeature;
        private final boolean includeIban;
        private final Statistics statistics = new Statistics();

        private long affectedTransfers;
        private long hiddenObservedFeatures;

        private MissingHistoryScenario(
                String name,
                int firstFeature,
                int lastFeature,
                boolean includeIban
        ) {
            if (firstFeature < 0
                    || lastFeature >= FEATURE_COUNT
                    || firstFeature > lastFeature) {
                throw new IllegalArgumentException(
                        "Invalid feature range"
                );
            }

            this.name = name;
            this.firstFeature = firstFeature;
            this.lastFeature = lastFeature;
            this.includeIban = includeIban;
        }

        private void mask(float[] inputs, int batchSize) {
            for (int row = 0; row < batchSize; row++) {
                int offset = row * INPUT_SIZE;
                boolean changed = false;

                for (int feature = firstFeature;
                     feature <= lastFeature;
                     feature++) {
                    changed |= hide(inputs, offset, feature);
                }

                if (includeIban && lastFeature < FEATURE_COUNT - 1) {
                    changed |= hide(
                            inputs,
                            offset,
                            FEATURE_COUNT - 1
                    );
                }

                if (changed) {
                    affectedTransfers++;
                }
            }
        }

        private boolean hide(
                float[] inputs,
                int offset,
                int feature
        ) {
            int valueIndex = offset + feature;
            int maskIndex = offset + FEATURE_COUNT + feature;

            if (inputs[maskIndex] == 0.0f) {
                return false;
            }

            inputs[valueIndex] = 0.0f;
            inputs[maskIndex] = 0.0f;
            hiddenObservedFeatures++;

            return true;
        }
    }

    private static Profile loadProfile() throws Exception {
        double[] means = new double[FEATURE_COUNT];
        double[] stds = new double[FEATURE_COUNT];

        FeatureName[] names = FeatureName.values();

        if (names.length != 34) {
            throw new IllegalStateException(
                    "Expected 34 original features"
            );
        }

        try (BufferedReader reader = Files.newBufferedReader(
                NORMALIZATION,
                StandardCharsets.UTF_8
        )) {
            if (!"feature,mean,std,observed_train_count"
                    .equals(reader.readLine())) {
                throw new IllegalStateException(
                        "Unexpected normalization header"
                );
            }

            for (int i = 0; i < FEATURE_COUNT; i++) {
                String line = reader.readLine();

                if (line == null) {
                    throw new IllegalStateException(
                            "Missing normalization entry"
                    );
                }

                String[] fields = line.split(",", -1);

                String expected = i == 34
                        ? "NEW_SUPPLIER_IBAN_AFTER_HISTORY"
                        : names[i].name();

                if (fields.length != 4
                        || !expected.equals(fields[0])) {
                    throw new IllegalStateException(
                            "Normalization feature mismatch"
                    );
                }

                means[i] = Double.parseDouble(fields[1]);
                stds[i] = Double.parseDouble(fields[2]);

                if (!Double.isFinite(means[i])
                        || !Double.isFinite(stds[i])
                        || stds[i] <= 0.0) {
                    throw new IllegalStateException(
                            "Invalid normalization statistics"
                    );
                }
            }

            if (reader.readLine() != null) {
                throw new IllegalStateException(
                        "Unexpected normalization entries"
                );
            }
        }

        if (means[34] != 0.0 || stds[34] != 1.0) {
            throw new IllegalStateException(
                    "Invalid IBAN normalization"
            );
        }

        return new Profile(means, stds);
    }

    private record Profile(double[] means, double[] stds) {
    }

    private static final class Statistics {

        private long total;

        private final List<FraudMetrics.Prediction> predictions =
                new ArrayList<>(274_867);

        private final long[] falsePositives =
                new long[BUDGETS.length];

        private final long[] invoice =
                new long[BUDGETS.length];

        private final long[] impersonation =
                new long[BUDGETS.length];

        private final long[] ceo =
                new long[BUDGETS.length];

        private void add(float logit, byte label, byte type) {
            if (!Float.isFinite(logit)) {
                throw new IllegalStateException(
                        "Non-finite ONNX prediction"
                );
            }

            if (label != 0 && label != 1) {
                throw new IllegalStateException("Invalid fraud label");
            }

            double probability;

            if (logit >= 0.0f) {
                probability = 1.0 / (1.0 + Math.exp(-(double) logit));
            } else {
                double exp = Math.exp((double) logit);
                probability = exp / (1.0 + exp);
            }

            predictions.add(new FraudMetrics.Prediction(label, probability));

            total++;

            for (int i = 0; i < BUDGETS.length; i++) {
                if (logit <= THRESHOLDS[i]) {
                    continue;
                }

                if (label == 0) {
                    falsePositives[i]++;
                } else if (label == 1) {
                    switch (type) {
                        case 1 -> invoice[i]++;
                        case 2 -> impersonation[i]++;
                        case 3 -> ceo[i]++;
                        default -> throw new IllegalStateException(
                                "Unknown fraud type"
                        );
                    }
                } else {
                    throw new IllegalStateException(
                            "Invalid fraud label"
                    );
                }
            }
        }
    }
}