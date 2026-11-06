package ru.finguard.training.bank;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ai.onnxruntime.OrtException;
import ru.finguard.core.feature.V3InputNormalizer;
import ru.finguard.core.neural.OnnxNeuralPredictor;
import ru.finguard.training.export.OnnxMlpExporter;
import ru.finguard.training.model.FraudMlp;


import java.io.InputStream;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.io.IOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.charset.StandardCharsets;
import java.util.Comparator;
import java.util.Objects;
import java.util.stream.Stream;

public final class BankV3OnnxRelease {

    private static final int INPUT_SIZE = V3InputNormalizer.INPUT_SIZE;

    private BankV3OnnxRelease() {
    }

    static Published publish(
            Path trainedModelDirectory,
            Path destination
    ) throws IOException, OrtException {
        return publishInternal(
                trainedModelDirectory,
                destination,
                null,
                null
        );
    }

    public static Published publish(
            Path trainedModelDirectory,
            Path destination,
            BankV3ModelQualityEvaluator.Report quality,
            BankV3CandidateReleaseGate.Criteria criteria
    ) throws IOException, OrtException {
        Objects.requireNonNull(quality);
        Objects.requireNonNull(criteria);

        BankV3CandidateReleaseGate.Decision decision =
                BankV3CandidateReleaseGate.assess(quality, criteria);

        if (!decision.eligibleForManualReview()) {
            throw new IllegalStateException(
                    "Model candidate rejected: "
                            + String.join("; ", decision.issues())
            );
        }

        return publishInternal(
                trainedModelDirectory,
                destination,
                quality,
                criteria
        );
    }

    private static Published publishInternal(
            Path trainedModelDirectory,
            Path destination,
            BankV3ModelQualityEvaluator.Report quality,
            BankV3CandidateReleaseGate.Criteria criteria
    ) throws IOException, OrtException {
        Objects.requireNonNull(trainedModelDirectory);
        Objects.requireNonNull(destination);

        Path source = trainedModelDirectory.toAbsolutePath();
        Path checkpointDirectory = source.resolve("checkpoints");
        Path normalization = source.resolve("normalization.csv");
        Path trainingMetadata = source.resolve("training.properties");

        if (!Files.isDirectory(checkpointDirectory)
                || !Files.isRegularFile(normalization)
                || !Files.isRegularFile(trainingMetadata)) {
            throw new IOException(
                    "Incomplete trained model artifacts"
            );
        }

        V3InputNormalizer.load(normalization);

        Path output = destination.toAbsolutePath();

        if (Files.exists(output)) {
            throw new IOException(
                    "Model release already exists: " + output
            );
        }

        Files.createDirectories(output.getParent());

        Path stage = Files.createTempDirectory(
                output.getParent(),
                "finguard-release-"
        );

        boolean published = false;

        try {
            Path onnx = stage.resolve("finguard-model.onnx");
            Path profile = stage.resolve("normalization.csv");

            Files.copy(normalization, profile);
            Files.copy(
                    trainingMetadata,
                    stage.resolve("training.properties")
            );

            if (quality != null) {
                writeQualityReport(
                        stage.resolve("quality.properties"),
                        quality,
                        criteria
                );
            }

            try (Model model = FraudMlp.create()) {
                try {
                    model.load(checkpointDirectory, "best");
                } catch (MalformedModelException exception) {
                    throw new IOException(
                            "Cannot load trained MLP checkpoint",
                            exception
                    );
                }

                OnnxMlpExporter.exportLogits(model, onnx);

                if (!Files.isRegularFile(onnx)
                        || Files.size(onnx) == 0) {
                    throw new IOException(
                            "ONNX export was not created"
                    );
                }

                verifyParity(model, onnx, profile);
            }

            if (quality != null) {
                Path metadata = stage.resolve("quality.properties");

                if (!Files.isRegularFile(metadata)) {
                    throw new IOException(
                            "Approved quality metadata was not created"
                    );
                }

                Files.writeString(
                        metadata,
                        "modelSha256=" + sha256(onnx) + "\n"
                                + "normalizationSha256="
                                + sha256(profile) + "\n",
                        StandardCharsets.UTF_8,
                        StandardOpenOption.APPEND
                );
            }

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

            return new Published(
                    output.resolve("finguard-model.onnx"),
                    output.resolve("normalization.csv"),
                    output.resolve("training.properties")
            );
        } finally {
            if (!published) {
                deleteTree(stage);
            }
        }
    }

    private static void writeQualityReport(
            Path destination,
            BankV3ModelQualityEvaluator.Report report,
            BankV3CandidateReleaseGate.Criteria criteria
    ) throws IOException {
        var validation = report.validation();
        var test = report.test();

        var noVelocity = report.missingHistory().get(
                BankV3ModelQualityEvaluator.Scenario.NO_VELOCITY_HISTORY
        );

        var noRecipient = report.missingHistory().get(
                BankV3ModelQualityEvaluator.Scenario.NO_RECIPIENT_HISTORY_AND_IBAN
        );

        String contents = """
                format=bank-v3-candidate-1
                eligibleForManualReview=true
                automaticActivation=false
                fpBudgetPerThousand=%d
                reviewThresholdLogit=%s
                validationLegitimate=%d
                validationFraud=%d
                validationFpPerThousand=%s
                validationRecall=%s
                testLegitimate=%d
                testFraud=%d
                testFpPerThousand=%s
                testRecall=%s
                testAveragePrecision=%s
                noVelocityFpPerThousand=%s
                noRecipientRecall=%s
                minValidationLegitimate=%d
                minValidationFraud=%d
                minTestLegitimate=%d
                minTestFraud=%d
                maxTestFpPerThousand=%s
                minTestRecall=%s
                minTestAveragePrecision=%s
                maxNoVelocityFpPerThousand=%s
                minNoRecipientRecall=%s
                """.formatted(
                report.fpBudgetPerThousand(),
                report.thresholdLogit(),
                validation.fp() + validation.tn(),
                validation.tp() + validation.fn(),
                validation.falsePositivesPerThousand(),
                validation.recall(),
                test.fp() + test.tn(),
                test.tp() + test.fn(),
                test.falsePositivesPerThousand(),
                test.recall(),
                test.averagePrecision(),
                noVelocity.falsePositivesPerThousand(),
                noRecipient.recall(),
                criteria.minValidationLegitimate(),
                criteria.minValidationFraud(),
                criteria.minTestLegitimate(),
                criteria.minTestFraud(),
                criteria.maxTestFpPerThousand(),
                criteria.minTestRecall(),
                criteria.minTestAveragePrecision(),
                criteria.maxNoVelocityFpPerThousand(),
                criteria.minNoRecipientRecall()
        );

        Files.writeString(
                destination,
                contents,
                StandardCharsets.UTF_8
        );
    }

    private static String sha256(Path path) throws IOException {
        MessageDigest digest;

        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException(
                    "SHA-256 is unavailable",
                    exception
            );
        }

        try (InputStream input = Files.newInputStream(path)) {
            byte[] buffer = new byte[8192];
            int count;

            while ((count = input.read(buffer)) != -1) {
                digest.update(buffer, 0, count);
            }
        }

        return HexFormat.of().formatHex(digest.digest());
    }

    private static void verifyParity(
            Model model,
            Path onnx,
            Path profile
    ) throws IOException, OrtException {
        V3InputNormalizer normalizer =
                V3InputNormalizer.load(profile);

        float[] missing = new float[INPUT_SIZE];

        float[] amount = new float[INPUT_SIZE];
        amount[0] = 100.0f;
        amount[35] = 1.0f;

        float[] iban = amount.clone();
        iban[34] = 1.0f;
        iban[69] = 1.0f;

        float[][] samples = {
                missing,
                amount,
                iban
        };

        try (Predictor<NDList, NDList> djl =
                     model.newPredictor(new NoopTranslator());
             OnnxNeuralPredictor runtime =
                     new OnnxNeuralPredictor(onnx)) {

            for (float[] sample : samples) {
                float[] normalized = normalizer.normalize(sample);

                float expected = predictDjl(djl, model, normalized);
                float actual = runtime.predictLogit(normalized);

                double tolerance =
                        0.0001 * Math.max(1.0, Math.abs(expected));

                if (!Float.isFinite(expected)
                        || !Float.isFinite(actual)
                        || Math.abs(expected - actual) > tolerance) {
                    throw new IOException(
                            "DJL and ONNX predictions do not match"
                    );
                }
            }
        }
    }

    private static float predictDjl(
            Predictor<NDList, NDList> predictor,
            Model model,
            float[] input
    ) throws IOException {
        try (NDManager manager =
                     model.getNDManager().newSubManager()) {

            var tensor = manager.create(input).reshape(
                    new Shape(1, INPUT_SIZE)
            );

            try (NDList output =
                         predictor.predict(new NDList(tensor))) {

                float[] logits =
                        output.singletonOrThrow().toFloatArray();

                if (logits.length != 1) {
                    throw new IOException(
                            "Unexpected DJL output shape"
                    );
                }

                return logits[0];
            } catch (TranslateException exception) {
                throw new IOException(
                        "DJL prediction failed",
                        exception
                );
            }
        }
    }

    private static void deleteTree(Path directory) throws IOException {
        if (!Files.exists(directory)) {
            return;
        }

        try (Stream<Path> files = Files.walk(directory)) {
            for (Path path : files.sorted(
                    Comparator.reverseOrder()
            ).toList()) {
                Files.deleteIfExists(path);
            }
        }
    }

    public record Published(
            Path modelPath,
            Path normalizationPath,
            Path trainingMetadataPath
    ) {
    }
}