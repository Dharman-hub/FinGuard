package ru.finguard.training.bank;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import ru.finguard.core.feature.V3InputNormalizer;
import ru.finguard.core.neural.OnnxModelRegistry;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

class BankV3FromScratchTrainerTest {

    @TempDir
    Path directory;

    @Test
    void shouldTrainOnBankV3DatasetAndSaveCheckpoint() throws Exception {
        Path dataset = directory.resolve("dataset");
        Files.createDirectories(dataset);

        write(dataset.resolve("train.fgbin"), 32);
        write(dataset.resolve("validation.fgbin"), 16);
        write(dataset.resolve("test.fgbin"), 8);

        Path output = directory.resolve("model-v1");

        var result = BankV3FromScratchTrainer.train(
                dataset,
                output,
                new BankV3FromScratchTrainer.Settings(
                        1, 8, 16, 0.001f, 20.0f
                )
        );

        assertEquals(1, result.bestEpoch());
        assertEquals(32, result.train().total());
        assertEquals(16, result.validation().total());
        assertTrue(Double.isFinite(result.bestValidationLoss()));

        assertTrue(Files.isRegularFile(
                output.resolve("normalization.csv")
        ));

        assertTrue(Files.isRegularFile(
                output.resolve("training.properties")
        ));

        assertNotNull(V3InputNormalizer.load(
                output.resolve("normalization.csv")
        ));

        try (Stream<Path> files = Files.list(
                output.resolve("checkpoints")
        )) {
            assertTrue(files.anyMatch(
                    path -> path.getFileName()
                            .toString().endsWith(".params")
            ));
        }
    }

    @Test
    void shouldRejectInvalidSettings() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new BankV3FromScratchTrainer.Settings(
                        1, 32, 8, 0.001f, 20.0f
                )
        );
    }

    @Test
    void shouldFineTuneExistingModelWithoutOverwritingSource()
            throws Exception {
        Path dataset = directory.resolve("dataset");
        Files.createDirectories(dataset);

        write(dataset.resolve("train.fgbin"), 32);
        write(dataset.resolve("validation.fgbin"), 16);
        write(dataset.resolve("test.fgbin"), 8);

        var settings = new BankV3FromScratchTrainer.Settings(
                1, 8, 16, 0.001f, 20.0f
        );

        Path original = directory.resolve("original");

        BankV3FromScratchTrainer.train(
                dataset,
                original,
                settings
        );

        Path checkpoint = original.resolve("checkpoints");

        Path originalParameters;

        try (Stream<Path> files = Files.list(checkpoint)) {
            originalParameters = files
                    .filter(path -> path.toString().endsWith(".params"))
                    .findFirst()
                    .orElseThrow();
        }

        byte[] parametersBefore =
                Files.readAllBytes(originalParameters);

        Path adapted = directory.resolve("adapted");

        var report = BankV3FromScratchTrainer.fineTune(
                dataset,
                adapted,
                settings,
                checkpoint,
                "best",
                original.resolve("normalization.csv")
        );

        assertEquals(1, report.bestEpoch());
        assertEquals(32, report.train().total());

        assertArrayEquals(
                parametersBefore,
                Files.readAllBytes(originalParameters)
        );

        assertArrayEquals(
                Files.readAllBytes(original.resolve("normalization.csv")),
                Files.readAllBytes(adapted.resolve("normalization.csv"))
        );

        String metadata = Files.readString(
                adapted.resolve("training.properties")
        );

        assertTrue(metadata.contains("trainingMode=FINE_TUNE"));
        assertTrue(metadata.contains("sourceModel=best"));

        try (Stream<Path> files =
                     Files.list(adapted.resolve("checkpoints"))) {
            assertTrue(
                    files.anyMatch(
                            path -> path.toString().endsWith(".params")
                    )
            );
        }
    }

    @Test
    void shouldExportBankModelAndRegisterNewVersion()
            throws Exception {
        Path dataset = directory.resolve("dataset");
        Files.createDirectories(dataset);

        write(dataset.resolve("train.fgbin"), 32);
        write(dataset.resolve("validation.fgbin"), 16);
        write(dataset.resolve("test.fgbin"), 8);

        Path trained = directory.resolve("trained");

        BankV3FromScratchTrainer.train(
                dataset,
                trained,
                new BankV3FromScratchTrainer.Settings(
                        1,
                        8,
                        16,
                        0.001f,
                        20.0f
                )
        );

        Path releaseDirectory = directory.resolve("release");

        BankV3OnnxRelease.Published release =
                BankV3OnnxRelease.publish(
                        trained,
                        releaseDirectory
                );

        assertTrue(Files.isRegularFile(release.modelPath()));
        assertTrue(Files.isRegularFile(release.normalizationPath()));
        assertTrue(Files.isRegularFile(release.trainingMetadataPath()));

        OnnxModelRegistry registry = new OnnxModelRegistry();

        var registered = registry.register(
                "bank-v1",
                release.modelPath(),
                release.normalizationPath()
        );

        assertEquals("bank-v1", registered.version());
        assertEquals(70, registered.inputSize());
        assertEquals(35, registered.featureCount());

        assertEquals(
                registered,
                registry.validateForActivation("bank-v1")
        );

        assertEquals(1, registry.versions().size());

        assertThrows(
                IOException.class,
                () -> BankV3OnnxRelease.publish(
                        trained,
                        releaseDirectory
                )
        );
    }

    private void write(Path path, int count) throws Exception {
        try (BankV3DatasetBinary.Writer writer =
                     BankV3DatasetBinary.openWriter(path)) {

            for (int i = 0; i < count; i++) {
                float[] values = new float[70];

                values[0] = i % 4 == 0
                        ? 5000.0f
                        : 100.0f + i;

                values[35] = 1.0f;
                values[9] = i;
                values[44] = 1.0f;

                writer.write(new BankV3TrainingSample(
                        "transfer-" + path.getFileName() + "-" + i,
                        i % 5,
                        Instant.parse("2025-01-01T00:00:00Z")
                                .plusSeconds(i),
                        Instant.parse("2025-01-10T00:00:00Z"),
                        i % 4 == 0 ? 1 : 0,
                        values
                ));
            }
        }
    }
}