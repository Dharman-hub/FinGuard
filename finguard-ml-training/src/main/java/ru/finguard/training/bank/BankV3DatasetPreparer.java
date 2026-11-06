package ru.finguard.training.bank;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Instant;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;
import java.util.stream.Stream;

public final class BankV3DatasetPreparer {

    public Result prepare(
            Stream<BankV3TrainingSample> samples,
            Path destination,
            Instant trainingEnd,
            Instant validationEnd,
            Instant labelsAsOf
    ) throws IOException {
        Objects.requireNonNull(samples);
        Objects.requireNonNull(destination);
        Objects.requireNonNull(trainingEnd);
        Objects.requireNonNull(validationEnd);
        Objects.requireNonNull(labelsAsOf);

        if (!trainingEnd.isBefore(validationEnd)
                || !validationEnd.isBefore(labelsAsOf)) {
            throw new IllegalArgumentException(
                    "Expected trainingEnd < validationEnd < labelsAsOf"
            );
        }

        Path output = destination.toAbsolutePath();

        if (Files.exists(output)) {
            throw new IOException(
                    "Dataset destination already exists: " + output
            );
        }

        Path parent = output.getParent();
        Files.createDirectories(parent);

        Path stage = Files.createTempDirectory(
                parent,
                "bank-v3-stage-"
        );

        Set<String> transferIds = new HashSet<>();

        long[] fraud = new long[3];
        long[] legit = new long[3];
        long[] skipped = {0};

        boolean published = false;

        try {
            try (samples;
                 BankV3DatasetBinary.Writer train =
                         BankV3DatasetBinary.openWriter(
                                 stage.resolve("train.fgbin")
                         );
                 BankV3DatasetBinary.Writer validation =
                         BankV3DatasetBinary.openWriter(
                                 stage.resolve("validation.fgbin")
                         );
                 BankV3DatasetBinary.Writer test =
                         BankV3DatasetBinary.openWriter(
                                 stage.resolve("test.fgbin")
                         )) {

                for (BankV3TrainingSample sample :
                        (Iterable<BankV3TrainingSample>) samples::iterator) {

                    Objects.requireNonNull(
                            sample,
                            "Training sample must not be null"
                    );

                    if (!transferIds.add(sample.transferId())) {
                        throw new IllegalArgumentException(
                                "Duplicate transfer ID: "
                                        + sample.transferId()
                        );
                    }

                    if (sample.eventTime().isAfter(labelsAsOf)) {
                        throw new IllegalArgumentException(
                                "Future transfer: " + sample.transferId()
                        );
                    }

                    int split = sample.eventTime().isBefore(trainingEnd)
                            ? 0
                            : sample.eventTime().isBefore(validationEnd)
                            ? 1
                            : 2;

                    Instant cutoff = split == 0
                            ? trainingEnd
                            : split == 1
                            ? validationEnd
                            : labelsAsOf;

                    if (sample.confirmedAt().isAfter(cutoff)) {
                        skipped[0]++;
                        continue;
                    }

                    BankV3DatasetBinary.Writer target = switch (split) {
                        case 0 -> train;
                        case 1 -> validation;
                        default -> test;
                    };

                    target.write(sample);

                    if (sample.label() == 1) {
                        fraud[split]++;
                    } else {
                        legit[split]++;
                    }
                }
            }

            for (int i = 0; i < 3; i++) {
                if (fraud[i] == 0 || legit[i] == 0) {
                    throw new IllegalArgumentException(
                            "Each split needs confirmed fraud "
                                    + "and legitimate records; split=" + i
                    );
                }
            }

            String manifest = """
                    format=bank-v3-1
                    inputSize=70
                    trainingEnd=%s
                    validationEnd=%s
                    labelsAsOf=%s
                    trainLegitimate=%d
                    trainFraud=%d
                    validationLegitimate=%d
                    validationFraud=%d
                    testLegitimate=%d
                    testFraud=%d
                    skippedLateConfirmations=%d
                    """.formatted(
                    trainingEnd,
                    validationEnd,
                    labelsAsOf,
                    legit[0],
                    fraud[0],
                    legit[1],
                    fraud[1],
                    legit[2],
                    fraud[2],
                    skipped[0]
            );

            Files.writeString(
                    stage.resolve("manifest.properties"),
                    manifest,
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

            return new Result(
                    output,
                    new SplitStats(legit[0], fraud[0]),
                    new SplitStats(legit[1], fraud[1]),
                    new SplitStats(legit[2], fraud[2]),
                    skipped[0]
            );
        } finally {
            if (!published) {
                Files.deleteIfExists(stage.resolve("train.fgbin"));
                Files.deleteIfExists(stage.resolve("validation.fgbin"));
                Files.deleteIfExists(stage.resolve("test.fgbin"));
                Files.deleteIfExists(stage.resolve("manifest.properties"));
                Files.deleteIfExists(stage);
            }
        }
    }

    public record SplitStats(long legitimate, long fraud) {
        public long total() {
            return Math.addExact(legitimate, fraud);
        }
    }

    public record Result(
            Path directory,
            SplitStats train,
            SplitStats validation,
            SplitStats test,
            long skippedLateConfirmations
    ) {
    }
}