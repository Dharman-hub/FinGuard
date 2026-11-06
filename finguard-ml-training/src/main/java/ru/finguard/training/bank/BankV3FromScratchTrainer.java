package ru.finguard.training.bank;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.Trainer;
import ai.djl.training.optimizer.Adam;
import ai.djl.training.tracker.Tracker;
import ai.djl.MalformedModelException;
import ai.djl.nn.SequentialBlock;

import ru.finguard.core.feature.V3InputNormalizer;
import ru.finguard.training.loss.PositiveWeightedBceLoss;
import ru.finguard.training.model.FraudMlp;


import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.stream.Stream;
import java.util.Arrays;

public final class BankV3FromScratchTrainer {

    private static final int INPUT_SIZE = V3InputNormalizer.INPUT_SIZE;

    private BankV3FromScratchTrainer() {
    }

    public static Report train(
            Path datasetDirectory,
            Path destination,
            Settings settings
    ) throws IOException {
        return runTraining(datasetDirectory, destination, settings, null);
    }

    public static Report fineTune(
            Path datasetDirectory,
            Path destination,
            Settings settings,
            Path checkpointDirectory,
            String checkpointName,
            Path sourceNormalization
    ) throws IOException {
        return runTraining(
                datasetDirectory,
                destination,
                settings,
                new StartingCheckpoint(
                        checkpointDirectory,
                        checkpointName,
                        sourceNormalization
                )
        );
    }

    private static Report runTraining(
            Path datasetDirectory,
            Path destination,
            Settings settings,
            StartingCheckpoint checkpoint
    ) throws IOException {
        Objects.requireNonNull(datasetDirectory);
        Objects.requireNonNull(destination);
        Objects.requireNonNull(settings);

        Path trainFile = datasetDirectory.resolve("train.fgbin");
        Path validationFile = datasetDirectory.resolve("validation.fgbin");
        Path testFile = datasetDirectory.resolve("test.fgbin");

        if (!Files.isRegularFile(trainFile)
                || !Files.isRegularFile(validationFile)
                || !Files.isRegularFile(testFile)) {
            throw new IllegalArgumentException(
                    "TRAIN, VALIDATION and TEST files are required"
            );
        }

        Counts train = count(trainFile);
        Counts validation = count(validationFile);

        requireBothClasses(train, "TRAIN");
        requireBothClasses(validation, "VALIDATION");

        float positiveWeight = (float) Math.max(
                1.0,
                Math.min(
                        settings.maxPositiveWeight(),
                        train.legitimate() / (double) train.fraud()
                )
        );

        Path output = destination.toAbsolutePath();

        if (Files.exists(output)) {
            throw new IOException(
                    "Model directory already exists: " + output
            );
        }

        Path parent = output.getParent();
        Files.createDirectories(parent);

        Path stage = Files.createTempDirectory(
                parent,
                "bank-v3-training-"
        );

        boolean published = false;

        try {
            Path normalizationPath =
                    stage.resolve("normalization.csv");

            if (checkpoint == null) {
                BankV3NormalizationFitter.fit(
                        trainFile,
                        normalizationPath
                );
            } else {
                if (!Files.isDirectory(checkpoint.directory())
                        || !Files.isRegularFile(checkpoint.normalization())) {
                    throw new IOException(
                            "Source checkpoint or normalization is missing"
                    );
                }

                Files.copy(
                        checkpoint.normalization(),
                        normalizationPath
                );
            }

            V3InputNormalizer normalizer =
                    V3InputNormalizer.load(normalizationPath);

            Path checkpoints = stage.resolve("checkpoints");
            Files.createDirectories(checkpoints);

            List<EpochResult> history = new ArrayList<>();
            double bestValidationLoss = Double.POSITIVE_INFINITY;
            int bestEpoch = 0;

            var optimizer = Adam.builder()
                    .optLearningRateTracker(
                            Tracker.fixed(settings.learningRate())
                    )
                    .optClipGrad(1.0f)
                    .build();

            var config = new DefaultTrainingConfig(
                    new PositiveWeightedBceLoss(positiveWeight)
            )
                    .optOptimizer(optimizer)
                    .optDevices(new Device[]{Device.cpu()});

            try (Model model = FraudMlp.create()) {

                if (checkpoint != null) {
                    try {
                        model.load(
                                checkpoint.directory(),
                                checkpoint.modelName()
                        );
                    } catch (MalformedModelException exception) {
                        throw new IOException(
                                "Unable to load pretrained FinGuard model",
                                exception
                        );
                    }
                }

                model.setProperty(
                        "inputSize",
                        Integer.toString(INPUT_SIZE)
                );
                model.setProperty(
                        "featureCount",
                        Integer.toString(INPUT_SIZE / 2)
                );
                model.setProperty(
                        "trainingMode",
                        checkpoint == null ? "FROM_SCRATCH" : "FINE_TUNE"
                );

                float[] pretrainedWeights = checkpoint == null
                        ? null
                        : firstLayerWeights(model);

                try (Trainer trainer = model.newTrainer(config)) {
                    trainer.initialize(new Shape(1, INPUT_SIZE));

                    if (pretrainedWeights != null
                            && !Arrays.equals(
                            pretrainedWeights,
                            firstLayerWeights(model)
                    )) {
                        throw new IllegalStateException(
                                "Trainer initialization changed pretrained weights"
                        );
                    }

                    for (int epoch = 1;
                         epoch <= settings.epochs();
                         epoch++) {

                        double trainLoss = trainEpoch(
                                trainer,
                                trainFile,
                                normalizer,
                                settings,
                                epoch
                        );

                        double validationLoss = validationEpoch(
                                trainer,
                                validationFile,
                                normalizer,
                                settings.batchSize(),
                                positiveWeight
                        );

                        if (!Double.isFinite(trainLoss)
                                || !Double.isFinite(validationLoss)) {
                            throw new IllegalStateException(
                                    "Non-finite training loss"
                            );
                        }

                        boolean best =
                                validationLoss < bestValidationLoss;

                        if (best) {
                            bestValidationLoss = validationLoss;
                            bestEpoch = epoch;

                            model.setProperty("Epoch", "1");
                            model.save(checkpoints, "best");
                        }

                        history.add(new EpochResult(
                                epoch,
                                trainLoss,
                                validationLoss,
                                best
                        ));
                    }
                }
            }

            if (bestEpoch == 0) {
                throw new IllegalStateException(
                        "No model checkpoint was produced"
                );
            }

            Files.writeString(
                    stage.resolve("training.properties"),
                    "format=bank-v3-1\n"
                            + "inputSize=70\n"
                            + "trainingMode="
                            + (checkpoint == null ? "FROM_SCRATCH" : "FINE_TUNE")
                            + "\n"
                            + "sourceModel="
                            + (checkpoint == null ? "NONE" : checkpoint.modelName())
                            + "\n"
                            + "trainTotal=" + train.total() + "\n"
                            + "trainFraud=" + train.fraud() + "\n"
                            + "validationTotal=" + validation.total() + "\n"
                            + "validationFraud=" + validation.fraud() + "\n"
                            + "positiveWeight=" + positiveWeight + "\n"
                            + "bestEpoch=" + bestEpoch + "\n"
                            + "bestValidationLoss=" + bestValidationLoss + "\n",
                    StandardCharsets.UTF_8
            );

            try {
                Files.move(
                        stage,
                        output,
                        StandardCopyOption.ATOMIC_MOVE
                );
            } catch (AtomicMoveNotSupportedException exception) {
                Files.move(stage, output);
            }

            published = true;

            return new Report(
                    output,
                    train,
                    validation,
                    positiveWeight,
                    bestEpoch,
                    bestValidationLoss,
                    history
            );
        } finally {
            if (!published) {
                deleteTree(stage);
            }
        }
    }

    private static double trainEpoch(
            Trainer trainer,
            Path file,
            V3InputNormalizer normalizer,
            Settings settings,
            int epoch
    ) throws IOException {
        long processed = 0;
        double lossSum = 0.0;
        Random random = new Random(42L + epoch);

        try (BankV3DatasetBinary.Reader reader =
                     BankV3DatasetBinary.openReader(file)) {

            List<BankV3TrainingSample> block =
                    new ArrayList<>(settings.shuffleBlockSize());

            while (true) {
                block.clear();

                for (int i = 0;
                     i < settings.shuffleBlockSize();
                     i++) {

                    BankV3TrainingSample sample = reader.readNext();

                    if (sample == null) {
                        break;
                    }

                    block.add(sample);
                }

                if (block.isEmpty()) {
                    break;
                }

                Collections.shuffle(block, random);

                for (int start = 0;
                     start < block.size();
                     start += settings.batchSize()) {

                    int end = Math.min(
                            start + settings.batchSize(),
                            block.size()
                    );

                    int size = end - start;

                    double loss = trainBatch(
                            trainer,
                            normalizer,
                            block.subList(start, end)
                    );

                    lossSum += loss * size;
                    processed += size;
                }
            }
        }

        if (processed == 0) {
            throw new IllegalStateException("Empty TRAIN dataset");
        }

        return lossSum / processed;
    }

    private static double validationEpoch(
            Trainer trainer,
            Path file,
            V3InputNormalizer normalizer,
            int batchSize,
            float positiveWeight
    ) throws IOException {
        long processed = 0;
        double lossSum = 0.0;

        try (BankV3DatasetBinary.Reader reader =
                     BankV3DatasetBinary.openReader(file)) {

            List<BankV3TrainingSample> batch =
                    new ArrayList<>(batchSize);

            while (true) {
                batch.clear();

                for (int i = 0; i < batchSize; i++) {
                    BankV3TrainingSample sample = reader.readNext();

                    if (sample == null) {
                        break;
                    }

                    batch.add(sample);
                }

                if (batch.isEmpty()) {
                    break;
                }

                lossSum += validationBatch(
                        trainer,
                        normalizer,
                        batch,
                        positiveWeight
                );

                processed += batch.size();
            }
        }

        if (processed == 0) {
            throw new IllegalStateException(
                    "Empty VALIDATION dataset"
            );
        }

        return lossSum / processed;
    }

    private static double trainBatch(
            Trainer trainer,
            V3InputNormalizer normalizer,
            List<BankV3TrainingSample> batch
    ) {
        float[] values = normalize(batch, normalizer);
        float[] targets = labels(batch);

        try (NDManager manager =
                     trainer.getManager().newSubManager()) {

            NDArray input = manager.create(values).reshape(
                    new Shape(batch.size(), INPUT_SIZE)
            );

            NDArray labels = manager.create(targets).reshape(
                    new Shape(batch.size(), 1)
            );

            NDArray loss;

            try (GradientCollector collector =
                         trainer.newGradientCollector()) {

                NDList prediction = trainer.forward(
                        new NDList(input)
                );

                loss = trainer.getLoss().evaluate(
                        new NDList(labels),
                        prediction
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

            return value;
        }
    }

    private static double validationBatch(
            Trainer trainer,
            V3InputNormalizer normalizer,
            List<BankV3TrainingSample> batch,
            float positiveWeight
    ) {
        float[] values = normalize(batch, normalizer);

        try (NDManager manager =
                     trainer.getManager().newSubManager()) {

            NDArray input = manager.create(values).reshape(
                    new Shape(batch.size(), INPUT_SIZE)
            );

            float[] logits = trainer.evaluate(
                    new NDList(input)
            ).singletonOrThrow().toFloatArray();

            if (logits.length != batch.size()) {
                throw new IllegalStateException(
                        "Unexpected validation output size"
                );
            }

            double lossSum = 0.0;

            for (int i = 0; i < logits.length; i++) {
                double logit = logits[i];
                int label = batch.get(i).label();

                double loss = Math.max(logit, 0.0)
                        - label * logit
                        + Math.log1p(
                        Math.exp(-Math.abs(logit))
                );

                lossSum += label == 1
                        ? positiveWeight * loss
                        : loss;
            }

            if (!Double.isFinite(lossSum)) {
                throw new IllegalStateException(
                        "Non-finite validation loss"
                );
            }

            return lossSum;
        }
    }

    private static float[] normalize(
            List<BankV3TrainingSample> batch,
            V3InputNormalizer normalizer
    ) {
        float[] values = new float[
                batch.size() * INPUT_SIZE
                ];

        for (int i = 0; i < batch.size(); i++) {
            float[] normalized = normalizer.normalize(
                    batch.get(i).modelInput()
            );

            System.arraycopy(
                    normalized,
                    0,
                    values,
                    i * INPUT_SIZE,
                    INPUT_SIZE
            );
        }

        return values;
    }

    private static float[] labels(
            List<BankV3TrainingSample> batch
    ) {
        float[] values = new float[batch.size()];

        for (int i = 0; i < batch.size(); i++) {
            values[i] = batch.get(i).label();
        }

        return values;
    }

    private static Counts count(Path file) throws IOException {
        long legitimate = 0;
        long fraud = 0;

        try (BankV3DatasetBinary.Reader reader =
                     BankV3DatasetBinary.openReader(file)) {

            BankV3TrainingSample sample;

            while ((sample = reader.readNext()) != null) {
                if (sample.label() == 1) {
                    fraud++;
                } else {
                    legitimate++;
                }
            }
        }

        return new Counts(legitimate, fraud);
    }

    private static void requireBothClasses(
            Counts counts,
            String split
    ) {
        if (counts.legitimate() == 0
                || counts.fraud() == 0) {
            throw new IllegalArgumentException(
                    split + " must contain both classes"
            );
        }
    }

    private static void deleteTree(Path root) throws IOException {
        if (!Files.exists(root)) {
            return;
        }

        try (Stream<Path> paths = Files.walk(root)) {
            for (Path path :
                    paths.sorted(
                            Comparator.reverseOrder()
                    ).toList()) {

                Files.deleteIfExists(path);
            }
        }
    }

    private static float[] firstLayerWeights(Model model) {
        if (!(model.getBlock() instanceof SequentialBlock network)) {
            throw new IllegalStateException("Expected FinGuard MLP");
        }

        var firstLayer = network.getChildren().get(0).getValue();
        var parameter = firstLayer.getParameters().get("weight");

        if (parameter == null || parameter.getArray() == null) {
            throw new IllegalStateException(
                    "Pretrained weights are not initialized"
            );
        }

        var shape = parameter.getArray().getShape();

        if (shape.dimension() != 2
                || shape.get(0) != 32
                || shape.get(1) != INPUT_SIZE) {
            throw new IllegalArgumentException(
                    "Pretrained model must have 70 inputs and 32 first-layer units"
            );
        }

        return parameter.getArray().toFloatArray();
    }

    private record StartingCheckpoint(
            Path directory,
            String modelName,
            Path normalization
    ) {
        private StartingCheckpoint {
            Objects.requireNonNull(directory);
            Objects.requireNonNull(normalization);

            if (modelName == null || modelName.isBlank()) {
                throw new IllegalArgumentException(
                        "Source model name is required"
                );
            }
        }
    }

    public record Settings(
            int epochs,
            int batchSize,
            int shuffleBlockSize,
            float learningRate,
            float maxPositiveWeight
    ) {
        public Settings {
            if (epochs <= 0
                    || batchSize <= 0
                    || shuffleBlockSize < batchSize
                    || !Float.isFinite(learningRate)
                    || learningRate <= 0.0f
                    || !Float.isFinite(maxPositiveWeight)
                    || maxPositiveWeight < 1.0f) {

                throw new IllegalArgumentException(
                        "Invalid training settings"
                );
            }
        }

        public static Settings defaults() {
            return new Settings(
                    5,
                    512,
                    8192,
                    0.0002f,
                    1000.0f
            );
        }
    }

    public record Counts(
            long legitimate,
            long fraud
    ) {
        public long total() {
            return Math.addExact(
                    legitimate,
                    fraud
            );
        }
    }

    public record EpochResult(
            int epoch,
            double trainLoss,
            double validationLoss,
            boolean best
    ) {
    }

    public record Report(
            Path directory,
            Counts train,
            Counts validation,
            float positiveWeight,
            int bestEpoch,
            double bestValidationLoss,
            List<EpochResult> history
    ) {
        public Report {
            history = List.copyOf(history);
        }
    }
}