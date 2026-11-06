package ru.finguard.training.dataset.synsepa;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.tracker.Tracker;
import ru.finguard.core.feature.FeatureName;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

public final class SynSepV3TrainingApplication {

    private static final Path DATASET = Path.of(
            "data", "synsepa", "experiment-v3"
    );

    private static final Path MODEL_DIRECTORY = Path.of(
            "models", "synsepa-v3"
    );

    private static final int FEATURE_COUNT = 35;
    private static final int BATCH_SIZE = 512;
    private static final int EPOCHS = 5;

    private static final float LEARNING_RATE = 0.0002f;
    private static final float POSITIVE_WEIGHT =
            1_281_624.0f / 2_030.0f;

    private SynSepV3TrainingApplication() {
    }

    public static void main(String[] args) throws IOException {
        Path normalization = MODEL_DIRECTORY.resolve(
                "normalization.csv"
        );

        Profile profile = loadProfile(normalization);

        System.out.println(
                "========== SYNSEPA V3 TRAINING =========="
        );
        System.out.println("Dataset: original SynSEPA");
        System.out.println("Fraud: Invoice, Impersonation, CEO");
        System.out.println("Romance: EXCLUDED");
        System.out.println("Epochs: " + EPOCHS);
        System.out.println("Batch size: " + BATCH_SIZE);
        System.out.println("Learning rate: " + LEARNING_RATE);
        System.out.printf(
                Locale.US,
                "Positive weight: %.6f%n",
                POSITIVE_WEIGHT
        );
        System.out.println("Test: NOT USED");

        trainModel(70, profile);

        System.out.println();
        System.out.println(
                "SYNSEPA V3 TRAINING COMPLETED"
        );
    }

    private static void trainModel(
            int inputSize,
            Profile profile
    ) throws IOException {
        String modelName = "synsepa-v3-70";

        Path output = MODEL_DIRECTORY
                .resolve("comparison")
                .resolve(modelName);

        Files.createDirectories(output);

        var optimizer = Adam.builder()
                .optLearningRateTracker(
                        Tracker.fixed(LEARNING_RATE)
                )
                .optClipGrad(1.0f)
                .build();

        var config = new DefaultTrainingConfig(
                new WeightedBinaryLoss(POSITIVE_WEIGHT)
        )
                .optOptimizer(optimizer)
                .optDevices(new Device[]{Device.cpu()});

        System.out.println();
        System.out.println(
                "========== MODEL " + inputSize + " =========="
        );

        try (Model model = Model.newInstance(
                modelName,
                Device.cpu(),
                "PyTorch"
        )) {
            model.setBlock(
                    new SequentialBlock()
                            .add(
                                    Linear.builder()
                                            .setUnits(32)
                                            .build()
                            )
                            .add(Activation::relu)
                            .add(
                                    Linear.builder()
                                            .setUnits(16)
                                            .build()
                            )
                            .add(Activation::relu)
                            .add(
                                    Linear.builder()
                                            .setUnits(1)
                                            .build()
                            )
            );

            model.setProperty(
                    "inputSize",
                    Integer.toString(inputSize)
            );

            model.setProperty(
                    "featureCount",
                    Integer.toString(inputSize / 2)
            );

            model.setProperty(
                    "dataset",
                    "synsepa-experiment-v3"
            );

            double bestValidationLoss =
                    Double.POSITIVE_INFINITY;

            try (Trainer trainer = model.newTrainer(config)) {
                trainer.initialize(new Shape(1, inputSize));

                for (int epoch = 1; epoch <= EPOCHS; epoch++) {
                    System.out.println();
                    System.out.println(
                            "Epoch " + epoch + "/" + EPOCHS
                    );

                    EpochResult train = runTrainEpoch(
                            trainer,
                            inputSize,
                            profile
                    );

                    EpochResult validation = runValidation(
                            trainer,
                            inputSize,
                            profile
                    );

                    System.out.printf(
                            Locale.US,
                            "Train: samples=%d loss=%.8f%n",
                            train.samples(),
                            train.loss()
                    );

                    System.out.printf(
                            Locale.US,
                            "Validation: samples=%d loss=%.8f%n",
                            validation.samples(),
                            validation.loss()
                    );

                    if (validation.loss() < bestValidationLoss) {
                        bestValidationLoss = validation.loss();

                        model.save(output, modelName);

                        System.out.println(
                                "Best checkpoint saved"
                        );
                    }
                }
            }
        }

        System.out.println(
                "Model saved: " + output.toAbsolutePath()
        );
    }

    private static EpochResult runTrainEpoch(
            Trainer trainer,
            int inputSize,
            Profile profile
    ) throws IOException {
        long[] samples = {0};
        long[] batches = {0};
        double[] totalLoss = {0.0};

        var summary = SynSepV3BatchReader.forEachBatch(
                DATASET.resolve("train.fgbin"),
                inputSize,
                BATCH_SIZE,
                batch -> {
                    try (NDManager manager =
                                 trainer.getManager().newSubManager()) {

                        NDArray inputs = createInputs(
                                manager,
                                batch,
                                profile
                        );

                        NDArray labels = createLabels(
                                manager,
                                batch
                        );

                        NDArray loss;

                        try (GradientCollector collector =
                                     trainer.newGradientCollector()) {

                            NDList predictions = trainer.forward(
                                    new NDList(inputs)
                            );

                            loss = trainer.getLoss().evaluate(
                                    new NDList(labels),
                                    predictions
                            );

                            collector.backward(loss);
                        }

                        double value = loss.getFloat();

                        if (!Double.isFinite(value)) {
                            throw new IllegalStateException(
                                    "Non-finite training loss"
                            );
                        }

                        trainer.step();

                        samples[0] += batch.size();
                        totalLoss[0] += value * batch.size();
                        batches[0]++;

                        if (batches[0] % 500 == 0) {
                            System.out.println(
                                    "Training batches: " + batches[0]
                            );
                        }
                    }
                }
        );

        if (summary.total() != 1_283_654
                || samples[0] != summary.total()) {
            throw new IllegalStateException(
                    "Incomplete training epoch"
            );
        }

        return new EpochResult(
                samples[0],
                totalLoss[0] / samples[0]
        );
    }

    private static EpochResult runValidation(
            Trainer trainer,
            int inputSize,
            Profile profile
    ) throws IOException {
        long[] samples = {0};
        double[] totalLoss = {0.0};

        var summary = SynSepV3BatchReader.forEachBatch(
                DATASET.resolve("validation.fgbin"),
                inputSize,
                BATCH_SIZE,
                batch -> {
                    try (NDManager manager =
                                 trainer.getManager().newSubManager()) {

                        NDArray inputs = createInputs(
                                manager,
                                batch,
                                profile
                        );

                        NDArray output = trainer.evaluate(
                                new NDList(inputs)
                        ).singletonOrThrow();

                        float[] logits = output.toFloatArray();
                        byte[] labels = batch.labels();

                        if (logits.length != labels.length) {
                            throw new IllegalStateException(
                                    "Validation output size mismatch"
                            );
                        }

                        for (int i = 0; i < labels.length; i++) {
                            double logit = logits[i];
                            double target = labels[i];

                            double loss = Math.max(logit, 0.0)
                                    - target * logit
                                    + Math.log1p(
                                    Math.exp(-Math.abs(logit))
                            );

                            if (labels[i] == 1) {
                                loss *= POSITIVE_WEIGHT;
                            }

                            if (!Double.isFinite(loss)) {
                                throw new IllegalStateException(
                                        "Non-finite validation loss"
                                );
                            }

                            totalLoss[0] += loss;
                        }

                        samples[0] += batch.size();
                    }
                }
        );

        if (summary.total() != 276_807
                || samples[0] != summary.total()) {
            throw new IllegalStateException(
                    "Incomplete validation epoch"
            );
        }

        return new EpochResult(
                samples[0],
                totalLoss[0] / samples[0]
        );
    }

    private static NDArray createInputs(
            NDManager manager,
            SynSepV3BatchReader.Batch batch,
            Profile profile
    ) {
        int inputSize = batch.inputSize();
        int featureCount = inputSize / 2;

        float[] normalized = batch.inputs().clone();

        for (int row = 0; row < batch.size(); row++) {
            int offset = row * inputSize;

            for (int feature = 0;
                 feature < featureCount;
                 feature++) {

                int valueIndex = offset + feature;
                int maskIndex =
                        offset + featureCount + feature;

                float value = normalized[valueIndex];
                float mask = normalized[maskIndex];

                if (mask == 0.0f) {
                    normalized[valueIndex] = 0.0f;
                    continue;
                }

                if (feature == 34) {
                    continue;
                }

                double result =
                        (value - profile.means()[feature])
                                / profile.stds()[feature];

                if (!Double.isFinite(result)
                        || Math.abs(result) > Float.MAX_VALUE) {
                    throw new IllegalStateException(
                            "Invalid normalized input"
                    );
                }

                normalized[valueIndex] = (float) result;
            }
        }

        return manager.create(normalized).reshape(
                new Shape(batch.size(), inputSize)
        );
    }

    private static NDArray createLabels(
            NDManager manager,
            SynSepV3BatchReader.Batch batch
    ) {
        float[] labels = new float[batch.size()];

        for (int i = 0; i < batch.size(); i++) {
            labels[i] = batch.labels()[i];
        }

        return manager.create(labels).reshape(
                new Shape(batch.size(), 1)
        );
    }

    private static Profile loadProfile(
            Path file
    ) throws IOException {
        double[] means = new double[FEATURE_COUNT];
        double[] stds = new double[FEATURE_COUNT];

        FeatureName[] original = FeatureName.values();

        if (original.length != 34) {
            throw new IOException(
                    "Expected 34 original FinGuard features"
            );
        }

        try (BufferedReader reader = Files.newBufferedReader(
                file,
                StandardCharsets.UTF_8
        )) {
            if (!"feature,mean,std,observed_train_count"
                    .equals(reader.readLine())) {
                throw new IOException(
                        "Invalid normalization header"
                );
            }

            for (int i = 0; i < FEATURE_COUNT; i++) {
                String line = reader.readLine();

                if (line == null) {
                    throw new IOException(
                            "Missing normalization row " + i
                    );
                }

                String[] fields = line.split(",", -1);

                String expectedName = i == 34
                        ? "NEW_SUPPLIER_IBAN_AFTER_HISTORY"
                        : original[i].name();

                if (fields.length != 4
                        || !expectedName.equals(fields[0])) {
                    throw new IOException(
                            "Normalization feature mismatch: "
                                    + expectedName
                    );
                }

                try {
                    means[i] = Double.parseDouble(fields[1]);
                    stds[i] = Double.parseDouble(fields[2]);
                } catch (NumberFormatException exception) {
                    throw new IOException(
                            "Invalid normalization value",
                            exception
                    );
                }

                if (!Double.isFinite(means[i])
                        || !Double.isFinite(stds[i])
                        || stds[i] <= 0.0) {
                    throw new IOException(
                            "Invalid normalization statistics"
                    );
                }
            }

            if (reader.readLine() != null) {
                throw new IOException(
                        "Unexpected normalization rows"
                );
            }
        }

        if (means[34] != 0.0 || stds[34] != 1.0) {
            throw new IOException(
                    "IBAN feature must remain binary"
            );
        }

        return new Profile(means, stds);
    }

    private static final class WeightedBinaryLoss extends Loss {

        private final float positiveWeight;

        private WeightedBinaryLoss(float positiveWeight) {
            super("WeightedBinaryCrossEntropy");
            this.positiveWeight = positiveWeight;
        }

        @Override
        public NDArray evaluate(
                NDList labels,
                NDList predictions
        ) {
            NDArray logits = predictions.singletonOrThrow();

            NDArray targets = labels.singletonOrThrow()
                    .reshape(logits.getShape());

            NDArray losses = Activation.relu(logits)
                    .sub(logits.mul(targets))
                    .add(
                            Activation.softPlus(
                                    logits.abs().neg()
                            )
                    );

            NDArray weights = targets
                    .mul(positiveWeight - 1.0f)
                    .add(1.0f);

            return losses.mul(weights).mean();
        }
    }

    private record Profile(
            double[] means,
            double[] stds
    ) {
    }

    private record EpochResult(
            long samples,
            double loss
    ) {
    }
}