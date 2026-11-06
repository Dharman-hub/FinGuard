package ru.finguard.training.dataset.synsepa;

import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.function.Consumer;

public final class SynSepV3BatchReader {

    private static final int MAGIC = 0x46475633;
    private static final int VERSION = 1;
    private static final int STORED_INPUT_SIZE = 70;
    private static final int HEADER_SIZE = 20;
    private static final int RECORD_SIZE = 294;

    private SynSepV3BatchReader() {
    }

    public static void main(String[] args) throws IOException {
        Path directory = Path.of(
                "data", "synsepa", "experiment-v3"
        );

        System.out.println(
                "========== SYNSEPA V3 BATCH READER =========="
        );
        System.out.println("Source: original SynSEPA");
        System.out.println("Test: NOT USED");
        System.out.println();

        Summary train = forEachBatch(
                directory.resolve("train.fgbin"),
                70,
                512,
                batch -> {
                }
        );

        printSummary("TRAIN", train);

        Summary validation = forEachBatch(
                directory.resolve("validation.fgbin"),
                70,
                512,
                batch -> {
                }
        );

        printSummary("VALIDATION", validation);

        if (train.total() != 1_283_654
                || train.invoice() != 636
                || train.impersonation() != 1017
                || train.ceo() != 377) {
            throw new IllegalStateException(
                    "Unexpected TRAIN distribution"
            );
        }

        if (validation.total() != 276_807
                || validation.invoice() != 128
                || validation.impersonation() != 216
                || validation.ceo() != 83) {
            throw new IllegalStateException(
                    "Unexpected VALIDATION distribution"
            );
        }

        System.out.println();
        System.out.println(
                "SYNSEPA V3 BATCH READER SUCCESSFUL"
        );
    }

    public static Summary forEachBatch(
            Path file,
            int inputSize,
            int batchSize,
            Consumer<Batch> consumer
    ) throws IOException {
        if (inputSize != STORED_INPUT_SIZE) {
            throw new IllegalArgumentException(
                    "Input size must be 70"
            );
        }

        if (batchSize <= 0) {
            throw new IllegalArgumentException(
                    "Batch size must be positive"
            );
        }

        if (consumer == null) {
            throw new IllegalArgumentException(
                    "Batch consumer is required"
            );
        }

        float[] batchInputs = new float[batchSize * inputSize];
        byte[] batchLabels = new byte[batchSize];
        byte[] batchTypes = new byte[batchSize];
        float[] storedInputs = new float[STORED_INPUT_SIZE];

        long legitimate = 0;
        long invoice = 0;
        long impersonation = 0;
        long ceo = 0;
        long available = 0;
        long changed = 0;
        long processed = 0;

        int buffered = 0;

        try (DataInputStream input = new DataInputStream(
                new BufferedInputStream(
                        Files.newInputStream(file),
                        1024 * 1024
                )
        )) {
            int magic = input.readInt();
            int version = input.readInt();
            int storedSize = input.readInt();
            long total = input.readLong();

            if (magic != MAGIC
                    || version != VERSION
                    || storedSize != STORED_INPUT_SIZE
                    || total < 0) {
                throw new IOException(
                        "Invalid FinGuard V3 binary header: " + file
                );
            }

            long expectedSize = Math.addExact(
                    HEADER_SIZE,
                    Math.multiplyExact(total, RECORD_SIZE)
            );

            if (Files.size(file) != expectedSize) {
                throw new IOException(
                        "Unexpected binary file size: " + file
                );
            }

            for (long row = 0; row < total; row++) {
                int accountId = input.readInt();
                long timestamp = input.readLong();
                int label = input.readUnsignedByte();
                int type = input.readUnsignedByte();

                if (accountId < 0
                        || accountId >= 10_000
                        || timestamp <= 0) {
                    throw new IOException(
                            "Invalid transaction metadata at row " + row
                    );
                }

                if (label == 0) {
                    if (type != 0) {
                        throw new IOException(
                                "Legitimate transfer has fraud type"
                        );
                    }

                    legitimate++;
                } else if (label == 1) {
                    switch (type) {
                        case 1 -> invoice++;
                        case 2 -> impersonation++;
                        case 3 -> ceo++;
                        default -> throw new IOException(
                                "Invalid fraud type at row " + row
                        );
                    }
                } else {
                    throw new IOException(
                            "Invalid fraud label at row " + row
                    );
                }

                for (int i = 0; i < STORED_INPUT_SIZE; i++) {
                    storedInputs[i] = input.readFloat();

                    if (!Float.isFinite(storedInputs[i])) {
                        throw new IOException(
                                "Non-finite model input at row " + row
                        );
                    }
                }

                for (int i = 35; i < STORED_INPUT_SIZE; i++) {
                    float mask = storedInputs[i];

                    if (mask != 0.0f && mask != 1.0f) {
                        throw new IOException(
                                "Invalid feature mask at row " + row
                        );
                    }
                }

                float changedValue = storedInputs[34];
                float changeMask = storedInputs[69];

                if ((changedValue != 0.0f
                        && changedValue != 1.0f)
                        || (changeMask == 0.0f
                        && changedValue != 0.0f)) {
                    throw new IOException(
                            "Invalid IBAN feature at row " + row
                    );
                }

                if (changeMask == 1.0f) {
                    available++;

                    if (changedValue == 1.0f) {
                        changed++;
                    }
                }

                int offset = buffered * inputSize;

                System.arraycopy(
                        storedInputs,
                        0,
                        batchInputs,
                        offset,
                        STORED_INPUT_SIZE
                );

                batchLabels[buffered] = (byte) label;
                batchTypes[buffered] = (byte) type;

                buffered++;
                processed++;

                if (buffered == batchSize) {
                    consumer.accept(
                            new Batch(
                                    Arrays.copyOf(
                                            batchInputs,
                                            buffered * inputSize
                                    ),
                                    Arrays.copyOf(batchLabels, buffered),
                                    Arrays.copyOf(batchTypes, buffered),
                                    buffered,
                                    inputSize
                            )
                    );

                    buffered = 0;
                }
            }

            if (buffered > 0) {
                consumer.accept(
                        new Batch(
                                Arrays.copyOf(
                                        batchInputs,
                                        buffered * inputSize
                                ),
                                Arrays.copyOf(batchLabels, buffered),
                                Arrays.copyOf(batchTypes, buffered),
                                buffered,
                                inputSize
                        )
                );
            }

            if (input.read() != -1) {
                throw new IOException(
                        "Unexpected data after last record"
                );
            }

            if (processed != total) {
                throw new IOException(
                        "Incomplete binary dataset"
                );
            }
        }

        return new Summary(
                processed,
                legitimate,
                invoice,
                impersonation,
                ceo,
                available,
                changed
        );
    }

    private static void printSummary(
            String name,
            Summary summary
    ) {
        System.out.println();
        System.out.println(
                "========== " + name + " =========="
        );

        System.out.println("Transfers: " + summary.total());
        System.out.println("Legitimate: " + summary.legitimate());
        System.out.println("Invoice: " + summary.invoice());
        System.out.println(
                "Impersonation: " + summary.impersonation()
        );
        System.out.println("CEO: " + summary.ceo());
        System.out.println(
                "IBAN feature available: " + summary.available()
        );
        System.out.println(
                "New supplier IBAN: " + summary.changed()
        );
    }

    public record Batch(
            float[] inputs,
            byte[] labels,
            byte[] fraudTypes,
            int size,
            int inputSize
    ) {
    }

    public record Summary(
            long total,
            long legitimate,
            long invoice,
            long impersonation,
            long ceo,
            long available,
            long changed
    ) {
    }
}