package ru.finguard.training.bank;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;
import ru.finguard.core.neural.OnnxModelRegistry;

import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class BankV3TrainingIntegrationTest {

    @TempDir
    Path directory;

    @Test
    @Timeout(value = 180, unit = TimeUnit.SECONDS)
    void shouldTrainFineTuneExportAndRegisterBankModels() throws Exception {
        List<BankV3TrainingSample> samples = new ArrayList<>();

        addSamples(samples, "train", "2025-01-10", 80);
        addSamples(samples, "validation", "2025-02-10", 40);
        addSamples(samples, "test", "2025-03-10", 40);

        Path dataset = directory.resolve("dataset");

        var prepared = new BankV3DatasetPreparer().prepare(
                samples.stream(),
                dataset,
                Instant.parse("2025-02-01T00:00:00Z"),
                Instant.parse("2025-03-01T00:00:00Z"),
                Instant.parse("2025-04-01T00:00:00Z")
        );

        assertEquals(80, prepared.train().total());
        assertEquals(40, prepared.validation().total());
        assertEquals(40, prepared.test().total());

        var limits = new BankV3ManualTrainingManager.ResourceLimits(
                1, 16, 64, 16
        );

        var settings = new BankV3FromScratchTrainer.Settings(
                1, 8, 16, 0.001f, 20.0f
        );

        Path trainedV1 = directory.resolve("bank-v1");
        Path releaseV1 = directory.resolve("release-v1");
        Path trainedV2 = directory.resolve("bank-v2");
        Path releaseV2 = directory.resolve("release-v2");

        OnnxModelRegistry registry = new OnnxModelRegistry();

        try (var manager = new BankV3ManualTrainingManager(limits)) {

            var smokeCriteria = new BankV3CandidateReleaseGate.Criteria(
                    5,
                    1,
                    1,
                    1,
                    1,
                    1000.0,
                    0.0,
                    0.0,
                    1000.0,
                    0.0
            );

            var firstRequest = new BankV3ManualTrainingManager.Request(
                    dataset,
                    trainedV1,
                    releaseV1,
                    settings,
                    5,
                    16,
                    smokeCriteria
            );

            long firstJob = manager.startFromScratch(firstRequest);

            assertCompleted(manager, firstJob);

            var first = manager.result(firstJob).orElseThrow();

            assertEquals(80, first.training().train().total());
            assertEquals(40, first.training().validation().total());
            assertTrue(Double.isFinite(
                    first.quality().test().averagePrecision()
            ));

            assertTrue(Files.isRegularFile(
                    first.release().modelPath()
            ));

            Path qualityFile = first.release()
                    .modelPath()
                    .getParent()
                    .resolve("quality.properties");

            assertTrue(Files.isRegularFile(qualityFile));

            String qualityMetadata = Files.readString(qualityFile);

            assertTrue(
                    qualityMetadata.contains("eligibleForManualReview=true")
            );

            assertTrue(
                    qualityMetadata.contains("reviewThresholdLogit=")
            );

            assertTrue(
                    qualityMetadata.contains("automaticActivation=false")
            );

            assertTrue(Files.isRegularFile(
                    first.release().normalizationPath()
            ));

            registry.register(
                    "bank-v1",
                    first.release().modelPath(),
                    first.release().normalizationPath()
            );

            assertEquals(
                    "bank-v1",
                    registry.validateForActivation("bank-v1").version()
            );

            var secondRequest = new BankV3ManualTrainingManager.Request(
                    dataset,
                    trainedV2,
                    releaseV2,
                    settings,
                    5,
                    16,
                    smokeCriteria
            );

            var source = new BankV3ManualTrainingManager.SourceCheckpoint(
                    trainedV1.resolve("checkpoints"),
                    "best",
                    trainedV1.resolve("normalization.csv")
            );

            long secondJob = manager.startFineTune(
                    secondRequest,
                    source
            );

            assertCompleted(manager, secondJob);

            var second = manager.result(secondJob).orElseThrow();

            assertTrue(Files.isRegularFile(
                    second.release().modelPath()
            ));

            registry.register(
                    "bank-v2",
                    second.release().modelPath(),
                    second.release().normalizationPath()
            );

            assertEquals(
                    "bank-v2",
                    registry.validateForActivation("bank-v2").version()
            );

            assertEquals(2, registry.versions().size());

            assertArrayEquals(
                    Files.readAllBytes(
                            trainedV1.resolve("normalization.csv")
                    ),
                    Files.readAllBytes(
                            trainedV2.resolve("normalization.csv")
                    )
            );

            assertTrue(Files.isRegularFile(
                    trainedV1.resolve("training.properties")
            ));

            assertTrue(Files.isRegularFile(
                    trainedV2.resolve("training.properties")
            ));

            assertTrue(
                    Files.readString(
                            trainedV2.resolve("training.properties")
                    ).contains("trainingMode=FINE_TUNE")
            );
        }
    }

    private void assertCompleted(
            BankV3ManualTrainingManager manager,
            long jobId
    ) throws InterruptedException {
        var status = manager.awaitCompletion(jobId);

        assertEquals(
                BankV3ManualTrainingManager.State.SUCCEEDED,
                status.state(),
                () -> "Training failed: " + status.error()
        );
    }

    private void addSamples(
            List<BankV3TrainingSample> target,
            String prefix,
            String date,
            int count
    ) {
        Instant start = Instant.parse(date + "T00:00:00Z");

        for (int i = 0; i < count; i++) {
            int label = i % 4 == 0 ? 1 : 0;

            float[] input = new float[70];

            input[0] = label == 1 ? 5000.0f + i : 100.0f + i;
            input[35] = 1.0f;

            input[9] = i % 8;
            input[44] = 1.0f;

            input[19] = i % 4;
            input[54] = 1.0f;

            input[34] = i % 2;
            input[69] = 1.0f;

            Instant eventTime = start.plusSeconds(i * 60L);

            target.add(new BankV3TrainingSample(
                    prefix + "-" + i,
                    i % 10,
                    eventTime,
                    eventTime.plusSeconds(600),
                    label,
                    input
            ));
        }
    }
}