package ru.finguard.training.bank;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

class BankV3DatasetPreparerTest {

    @TempDir
    Path directory;

    private final BankV3DatasetPreparer preparer =
            new BankV3DatasetPreparer();

    @Test
    void shouldSplitAndReadBankDataset() throws IOException {
        Path output = directory.resolve("bank-v3");

        BankV3DatasetPreparer.Result result = preparer.prepare(
                samples().stream(),
                output,
                date("2025-02-01"),
                date("2025-03-01"),
                date("2025-04-01")
        );

        assertEquals(2, result.train().total());
        assertEquals(2, result.validation().total());
        assertEquals(2, result.test().total());
        assertEquals(1, result.skippedLateConfirmations());

        assertTrue(Files.exists(output.resolve("manifest.properties")));

        try (BankV3DatasetBinary.Reader reader =
                     BankV3DatasetBinary.openReader(
                             output.resolve("train.fgbin")
                     )) {

            BankV3TrainingSample first = reader.readNext();
            BankV3TrainingSample second = reader.readNext();

            assertNotNull(first);
            assertNotNull(second);
            assertNull(reader.readNext());

            assertEquals(70, first.modelInput().length);
            assertEquals(500.0f, first.modelInput()[0]);
            assertEquals(1.0f, first.modelInput()[35]);
        }
    }

    @Test
    void shouldRejectDuplicateTransfers() {
        Path output = directory.resolve("duplicates");

        List<BankV3TrainingSample> records = samples();

        Stream<BankV3TrainingSample> duplicated = Stream.concat(
                records.stream(),
                Stream.of(records.get(0))
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> preparer.prepare(
                        duplicated,
                        output,
                        date("2025-02-01"),
                        date("2025-03-01"),
                        date("2025-04-01")
                )
        );

        assertFalse(Files.exists(output));
    }

    @Test
    void shouldRejectDatasetWithoutFraudInEverySplit() {
        Path output = directory.resolve("invalid");

        assertThrows(
                IllegalArgumentException.class,
                () -> preparer.prepare(
                        Stream.of(
                                sample("train", "2025-01-01", "2025-01-02", 0),
                                sample("validation", "2025-02-01", "2025-02-02", 0),
                                sample("test", "2025-03-01", "2025-03-02", 0)
                        ),
                        output,
                        date("2025-02-01"),
                        date("2025-03-01"),
                        date("2025-04-01")
                )
        );

        assertFalse(Files.exists(output));
    }

    private List<BankV3TrainingSample> samples() {
        return List.of(
                sample("t1", "2025-01-01", "2025-01-02", 0),
                sample("t2", "2025-01-03", "2025-01-04", 1),
                sample("v1", "2025-02-01", "2025-02-02", 0),
                sample("v2", "2025-02-05", "2025-02-06", 1),
                sample("e1", "2025-03-01", "2025-03-02", 0),
                sample("e2", "2025-03-05", "2025-03-06", 1),
                sample("late", "2025-01-01", "2025-03-01", 1)
        );
    }

    private BankV3TrainingSample sample(
            String id,
            String event,
            String confirmed,
            int label
    ) {
        float[] input = new float[70];
        input[0] = 500.0f;
        input[35] = 1.0f;

        return new BankV3TrainingSample(
                id,
                20L,
                date(event),
                date(confirmed),
                label,
                input
        );
    }

    private Instant date(String value) {
        return Instant.parse(value + "T00:00:00Z");
    }
}