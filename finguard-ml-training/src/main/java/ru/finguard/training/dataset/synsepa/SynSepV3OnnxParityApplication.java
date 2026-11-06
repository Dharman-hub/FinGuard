package ru.finguard.training.dataset.synsepa;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ru.finguard.core.feature.FeatureName;

import java.io.BufferedReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.Map;
import java.util.Arrays;

public final class SynSepV3OnnxParityApplication {

    private static final String MODEL_NAME = "synsepa-v3-70";

    private static final Path MODEL_ROOT = Path.of(
            "models", "synsepa-v3"
    );

    private static final Path MODEL_DIRECTORY = MODEL_ROOT
            .resolve("comparison")
            .resolve(MODEL_NAME);

    private static final Path ONNX_FILE = MODEL_ROOT.resolve(
            "finguard-model.onnx"
    );

    private static final Path NORMALIZATION = MODEL_ROOT.resolve(
            "normalization.csv"
    );

    private static final Path VALIDATION = Path.of(
            "data", "synsepa", "experiment-v3", "validation.fgbin"
    );

    private static final int INPUT_SIZE = 70;
    private static final int FEATURE_COUNT = 35;
    private static final int BATCH_SIZE = 512;

    private static final int[] BUDGETS = {1, 2, 5};

    private static final float[] THRESHOLDS = {
            5.04847717f,
            3.77777505f,
            1.51455879f
    };

    private SynSepV3OnnxParityApplication() {
    }

    public static void main(String[] args) throws Exception {
        System.out.println(
                "========== FINGUARD V3 ONNX PARITY TEST =========="
        );

        System.out.println("Model: " + MODEL_NAME);
        System.out.println("Inputs: " + INPUT_SIZE);
        System.out.println("Comparison: DJL vs ONNX Runtime");
        System.out.println("Split: VALIDATION");
        System.out.println("Test: NOT USED");

        if (!Files.isRegularFile(ONNX_FILE)) {
            throw new IllegalStateException(
                    "ONNX model not found: " + ONNX_FILE
            );
        }

        Profile profile = loadProfile();
        Statistics statistics = new Statistics();

        OrtEnvironment environment = OrtEnvironment.getEnvironment();

        try (Model model = Model.newInstance(
                MODEL_NAME,
                Device.cpu(),
                "PyTorch"
        );
             OrtSession.SessionOptions options =
                     new OrtSession.SessionOptions();
             OrtSession session = environment.createSession(
                     ONNX_FILE.toString(),
                     options
             )) {

            model.setBlock(
                    new SequentialBlock()
                            .add(Linear.builder().setUnits(32).build())
                            .add(Activation::relu)
                            .add(Linear.builder().setUnits(16).build())
                            .add(Activation::relu)
                            .add(Linear.builder().setUnits(1).build())
            );

            model.load(MODEL_DIRECTORY, MODEL_NAME);

            if (!session.getInputNames().contains("features")
                    || !session.getOutputNames().contains("risk_logit")) {
                throw new IllegalStateException(
                        "Unexpected ONNX input or output names"
                );
            }

            try (Predictor<NDList, NDList> predictor =
                         model.newPredictor(new NoopTranslator())) {

                var summary = SynSepV3BatchReader.forEachBatch(
                        VALIDATION,
                        INPUT_SIZE,
                        BATCH_SIZE,
                        batch -> compareBatch(
                                batch,
                                profile,
                                model,
                                predictor,
                                environment,
                                session,
                                statistics
                        )
                );

                if (summary.total() != 276_807
                        || summary.legitimate() != 276_380
                        || summary.invoice() != 128
                        || summary.impersonation() != 216
                        || summary.ceo() != 83
                        || statistics.total != summary.total()) {
                    throw new IllegalStateException(
                            "Unexpected validation distribution"
                    );
                }
            }
        }

        System.out.println();
        System.out.println(
                "========== NUMERICAL COMPARISON =========="
        );

        System.out.println(
                "Compared transfers: " + statistics.total
        );

        System.out.printf(
                Locale.US,
                "Maximum absolute difference: %.10f%n",
                statistics.maxDifference
        );

        System.out.printf(
                Locale.US,
                "Mean absolute difference: %.10f%n",
                statistics.totalDifference / statistics.total
        );

        System.out.println(
                "Differences above tolerance: "
                        + statistics.outsideTolerance
        );

        System.out.println();
        System.out.println(
                "========== DECISION COMPARISON =========="
        );

        for (int i = 0; i < BUDGETS.length; i++) {
            System.out.println();
            System.out.println(
                    BUDGETS[i] + " FP/1000 VALIDATION BUDGET"
            );

            System.out.printf(
                    Locale.US,
                    "Threshold: %.8f%n",
                    THRESHOLDS[i]
            );

            System.out.println(
                    "DJL: TP=" + statistics.djlTp[i]
                            + " FP=" + statistics.djlFp[i]
            );

            System.out.println(
                    "ONNX: TP=" + statistics.onnxTp[i]
                            + " FP=" + statistics.onnxFp[i]
            );

            System.out.println(
                    "Decision mismatches: "
                            + statistics.decisionMismatches[i]
            );
        }

        if (statistics.djlTp[2] != 425
                || statistics.djlFp[2] != 1381) {
            throw new IllegalStateException(
                    "DJL results differ from the previous Validation evaluation"
            );
        }

        if (statistics.outsideTolerance > 0) {
            throw new IllegalStateException(
                    "ONNX and DJL numerical predictions differ"
            );
        }

        long totalBoundaryMismatches = 0;

        for (long mismatches : statistics.decisionMismatches) {
            totalBoundaryMismatches += mismatches;
        }

        System.out.println(
                "Threshold-boundary mismatches: "
                        + totalBoundaryMismatches
        );

        System.out.println(
                "ONNX AND DJL NUMERICAL PARITY PASSED"
        );


        System.out.println();

        printOnnxCalibration(statistics);

        System.out.println();
        System.out.println(
                "FINGUARD V3 ONNX PARITY TEST PASSED"
        );
    }

    private static void compareBatch(
            SynSepV3BatchReader.Batch batch,
            Profile profile,
            Model model,
            Predictor<NDList, NDList> predictor,
            OrtEnvironment environment,
            OrtSession session,
            Statistics statistics
    ) {
        float[] normalized = normalize(batch, profile);

        FloatBuffer buffer = ByteBuffer.allocateDirect(
                normalized.length * Float.BYTES
        ).order(ByteOrder.nativeOrder()).asFloatBuffer();

        buffer.put(normalized);
        buffer.flip();

        try (NDManager manager = model.getNDManager().newSubManager()) {
            var input = manager.create(normalized).reshape(
                    new Shape(batch.size(), INPUT_SIZE)
            );

            try (NDList djlOutput = predictor.predict(
                    new NDList(input)
            );
                 OnnxTensor onnxInput = OnnxTensor.createTensor(
                         environment,
                         buffer,
                         new long[]{batch.size(), INPUT_SIZE}
                 );
                 OrtSession.Result onnxOutput = session.run(
                         Map.of("features", onnxInput)
                 )) {

                float[] djlValues = djlOutput
                        .singletonOrThrow()
                        .toFloatArray();

                float[][] onnxValues = (float[][]) onnxOutput
                        .get(0)
                        .getValue();

                if (djlValues.length != batch.size()
                        || onnxValues.length != batch.size()) {
                    throw new IllegalStateException(
                            "Model output size mismatch"
                    );
                }

                for (int i = 0; i < batch.size(); i++) {
                    if (onnxValues[i].length != 1) {
                        throw new IllegalStateException(
                                "Unexpected ONNX output shape"
                        );
                    }

                    float djl = djlValues[i];
                    float onnx = onnxValues[i][0];

                    if (!Float.isFinite(djl)
                            || !Float.isFinite(onnx)) {
                        throw new IllegalStateException(
                                "Non-finite prediction"
                        );
                    }

                    statistics.add(
                            djl,
                            onnx,
                            batch.labels()[i]
                    );
                }
            }
        } catch (TranslateException | OrtException exception) {
            throw new IllegalStateException(
                    "Model comparison failed",
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

                double value =
                        (inputs[valueIndex] - profile.means[feature])
                                / profile.stds[feature];

                if (!Double.isFinite(value)
                        || Math.abs(value) > Float.MAX_VALUE) {
                    throw new IllegalStateException(
                            "Invalid normalized input"
                    );
                }

                inputs[valueIndex] = (float) value;
            }
        }

        return inputs;
    }

    private static void printOnnxCalibration(
            Statistics statistics
    ) {
        if (statistics.legitimateCount != 276_380
                || statistics.fraudCount != 427) {
            throw new IllegalStateException(
                    "Unexpected validation distribution"
            );
        }

        float[] legitimateScores =
                statistics.onnxLegitimateScores.clone();

        Arrays.sort(legitimateScores);

        System.out.println();
        System.out.println(
                "========== ONNX VALIDATION CALIBRATION =========="
        );

        for (int budget : BUDGETS) {
            int allowedFp = (int) (
                    (long) legitimateScores.length * budget / 1000
            );

            int thresholdIndex =
                    legitimateScores.length - allowedFp - 1;

            float threshold = legitimateScores[thresholdIndex];

            int falsePositives = 0;
            int truePositives = 0;

            for (float score : legitimateScores) {
                if (score > threshold) {
                    falsePositives++;
                }
            }

            for (float score : statistics.onnxFraudScores) {
                if (score > threshold) {
                    truePositives++;
                }
            }

            if (falsePositives > allowedFp) {
                throw new IllegalStateException(
                        "False positive budget exceeded"
                );
            }

            System.out.println();
            System.out.println("Budget: " + budget + " FP/1000");
            System.out.println("Threshold logit: " + threshold);

            System.out.printf(
                    Locale.US,
                    "True positives: %d / %d%n",
                    truePositives,
                    statistics.fraudCount
            );

            System.out.printf(
                    Locale.US,
                    "False positives: %d / %d%n",
                    falsePositives,
                    statistics.legitimateCount
            );

            System.out.printf(
                    Locale.US,
                    "Actual FP/1000: %.4f%n",
                    falsePositives * 1000.0
                            / statistics.legitimateCount
            );
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
                            "Missing normalization row"
                    );
                }

                String[] fields = line.split(",", -1);

                String expected = i == 34
                        ? "NEW_SUPPLIER_IBAN_AFTER_HISTORY"
                        : names[i].name();

                if (fields.length != 4
                        || !expected.equals(fields[0])) {
                    throw new IllegalStateException(
                            "Normalization feature mismatch: " + expected
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
                        "Unexpected normalization rows"
                );
            }
        }

        if (means[34] != 0.0 || stds[34] != 1.0) {
            throw new IllegalStateException(
                    "New IBAN feature must remain binary"
            );
        }

        return new Profile(means, stds);
    }

    private record Profile(
            double[] means,
            double[] stds
    ) {
    }

    private static final class Statistics {

        private long total;
        private long outsideTolerance;
        private double maxDifference;
        private double totalDifference;

        private final float[] onnxLegitimateScores = new float[276_380];
        private final float[] onnxFraudScores = new float[427];

        private int legitimateCount;
        private int fraudCount;

        private final long[] djlTp = new long[BUDGETS.length];
        private final long[] djlFp = new long[BUDGETS.length];
        private final long[] onnxTp = new long[BUDGETS.length];
        private final long[] onnxFp = new long[BUDGETS.length];
        private final long[] decisionMismatches =
                new long[BUDGETS.length];

        private void add(
                float djl,
                float onnx,
                byte label
        ) {
            total++;

            double difference = Math.abs((double) djl - onnx);

            maxDifference = Math.max(maxDifference, difference);
            totalDifference += difference;

            if (label == 0) {
                if (legitimateCount >= onnxLegitimateScores.length) {
                    throw new IllegalStateException("Too many legitimate transfers");
                }

                onnxLegitimateScores[legitimateCount++] = onnx;
            } else if (label == 1) {
                if (fraudCount >= onnxFraudScores.length) {
                    throw new IllegalStateException("Too many fraudulent transfers");
                }

                onnxFraudScores[fraudCount++] = onnx;
            } else {
                throw new IllegalArgumentException("Invalid fraud label");
            }

            double tolerance = 0.001
                    + 0.0001 * Math.max(
                    Math.abs((double) djl),
                    Math.abs((double) onnx)
            );

            if (difference > tolerance) {
                outsideTolerance++;
            }

            for (int i = 0; i < BUDGETS.length; i++) {
                boolean djlFlagged = djl > THRESHOLDS[i];
                boolean onnxFlagged = onnx > THRESHOLDS[i];

                if (djlFlagged != onnxFlagged) {
                    decisionMismatches[i]++;

                    double distanceToThreshold = Math.min(
                            Math.abs((double) djl - THRESHOLDS[i]),
                            Math.abs((double) onnx - THRESHOLDS[i])
                    );

                    if (distanceToThreshold > 0.0001) {
                        throw new IllegalStateException(
                                "Unexpected decision mismatch at budget "
                                        + BUDGETS[i]
                        );
                    }
                }

                if (label == 1) {
                    if (djlFlagged) {
                        djlTp[i]++;
                    }

                    if (onnxFlagged) {
                        onnxTp[i]++;
                    }
                } else {
                    if (djlFlagged) {
                        djlFp[i]++;
                    }

                    if (onnxFlagged) {
                        onnxFp[i]++;
                    }
                }
            }
        }
    }
}