package ru.finguard.training.dataset.synsepa;

import ru.finguard.core.feature.FeatureExtractor;
import ru.finguard.core.feature.NewSupplierIbanDetector;
import ru.finguard.core.feature.RecipientAccountInputEncoder;

import java.io.BufferedReader;
import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public final class SynSepV3DatasetApplication {
    private static final int MAGIC = 0x46475633;
    private static final int VERSION = 1;
    private static final int INPUT_SIZE = 70;

    private SynSepV3DatasetApplication() {
    }

    public static void main(String[] args) throws IOException {
        Path source = Path.of("data", "synsep_full_dataset.csv");
        Path features = Path.of(
                "data", "synsepa", "experiment-v3", "iban_features.csv"
        );
        Path destination = Path.of(
                "data", "synsepa", "experiment-v3-regenerated"
        );

        Report report = generate(source, features, destination);

        if (report.train() != 1_283_654
                || report.validation() != 276_807
                || report.test() != 274_867
                || report.excludedRomance() != 4_232) {
            throw new IOException(
                    "Unexpected SynSEPA v3 distribution: " + report
            );
        }

        System.out.println("SynSEPA v3 created: " + report);
    }

    public static Report generate(
            Path source,
            Path features,
            Path destination
    ) throws IOException {
        Objects.requireNonNull(source);
        Objects.requireNonNull(features);
        Objects.requireNonNull(destination);

        Map<String, Signal> signals = readSignals(features);
        Files.createDirectories(destination);

        FeatureExtractor extractor = new FeatureExtractor();
        long[] excludedRomance = {0};

        try (Writer train = new Writer(destination.resolve("train.fgbin"));
             Writer validation = new Writer(
                     destination.resolve("validation.fgbin")
             );
             Writer test = new Writer(destination.resolve("test.fgbin"))) {

            try {
                SynSepChronologicalReader.forEachAccount(
                        source,
                        transactions -> {
                            String account =
                                    transactions.get(0).accountId();

                            int accountId = numericAccountId(account);

                            Writer writer = switch (partition(account)) {
                                case TRAIN -> train;
                                case VALIDATION -> validation;
                                case TEST -> test;
                            };

                            Set<Instant> seen = new HashSet<>();
                            Set<Instant> ambiguous = new HashSet<>();

                            for (var transaction : transactions) {
                                if (!romance(transaction)
                                        && !seen.add(
                                        transaction.timestamp()
                                )) {
                                    ambiguous.add(
                                            transaction.timestamp()
                                    );
                                }
                            }

                            SynSepHistoryBuilder.forEachTransaction(
                                    transactions,
                                    prepared -> {
                                        var transaction =
                                                prepared.transaction();

                                        Signal signal = signals.remove(
                                                transaction.transactionId()
                                        );

                                        if (signal == null
                                                || !account.equals(
                                                signal.accountId()
                                        )) {
                                            throw new IllegalStateException(
                                                    "Missing or mismatched IBAN signal: "
                                                            + transaction.transactionId()
                                            );
                                        }

                                        if (romance(transaction)) {
                                            excludedRomance[0]++;
                                            return;
                                        }

                                        float ibanMask = ambiguous.contains(
                                                transaction.timestamp()
                                        )
                                                ? 0.0f
                                                : signal.mask();

                                        float ibanValue = ibanMask == 0.0f
                                                ? 0.0f
                                                : signal.value();

                                        float[] base = extractor.extract(
                                                prepared.transfer(),
                                                prepared.snapshot()
                                        ).toModelInput();

                                        for (int index = 0; index < 34; index++) {
                                            if (base[34 + index] == 0.0f) {
                                                base[index] = 0.0f;
                                            }
                                        }

                                        float[] input =
                                                RecipientAccountInputEncoder.encode(
                                                        base,
                                                        new NewSupplierIbanDetector.Result(
                                                                ibanValue,
                                                                ibanMask
                                                        )
                                                );

                                        int label = transaction.fraud()
                                                ? 1
                                                : 0;

                                        byte type = fraudType(
                                                transaction.fraud(),
                                                transaction.fraudType()
                                        );

                                        try {
                                            writer.write(
                                                    accountId,
                                                    transaction.timestamp(),
                                                    label,
                                                    type,
                                                    input
                                            );
                                        } catch (IOException exception) {
                                            throw new UncheckedIOException(
                                                    exception
                                            );
                                        }
                                    }
                            );
                        }
                );
            } catch (UncheckedIOException exception) {
                throw exception.getCause();
            }

            if (!signals.isEmpty()) {
                throw new IOException(
                        "Unmatched IBAN signals: " + signals.size()
                );
            }

            return new Report(
                    train.count(),
                    validation.count(),
                    test.count(),
                    excludedRomance[0]
            );
        }
    }

    private static Map<String, Signal> readSignals(Path features)
            throws IOException {
        Map<String, Signal> signals = new HashMap<>();

        try (BufferedReader reader = Files.newBufferedReader(
                features,
                StandardCharsets.UTF_8
        )) {
            String header = "account_id,transaction_id,timestamp,"
                    + "NEW_SUPPLIER_IBAN_AFTER_HISTORY,"
                    + "mask_NEW_SUPPLIER_IBAN_AFTER_HISTORY";

            if (!header.equals(reader.readLine())) {
                throw new IOException(
                        "Unexpected IBAN features header"
                );
            }

            String line;

            while ((line = reader.readLine()) != null) {
                String[] columns = line.split(",", -1);

                if (columns.length != 5) {
                    throw new IOException(
                            "Invalid IBAN features row"
                    );
                }

                float value = parseBinary(columns[3]);
                float mask = parseBinary(columns[4]);

                if (mask == 0.0f && value != 0.0f) {
                    throw new IOException(
                            "Invalid missing IBAN value"
                    );
                }

                if (signals.putIfAbsent(
                        columns[1],
                        new Signal(columns[0], value, mask)
                ) != null) {
                    throw new IOException(
                            "Duplicate IBAN transaction ID: "
                                    + columns[1]
                    );
                }
            }
        }

        return signals;
    }

    private static Partition partition(String accountId) {
        long hash = accountId.hashCode() ^ 0x9E3779B97F4A7C15L;

        hash = (hash ^ (hash >>> 30))
                * 0xBF58476D1CE4E5B9L;

        hash = (hash ^ (hash >>> 27))
                * 0x94D049BB133111EBL;

        hash ^= hash >>> 31;

        int bucket = (int) Long.remainderUnsigned(hash, 100);

        if (bucket < 70) {
            return Partition.TRAIN;
        }

        if (bucket < 85) {
            return Partition.VALIDATION;
        }

        return Partition.TEST;
    }

    private static int numericAccountId(String account) {
        if (!account.matches("ACC_[0-9]{6}")) {
            throw new IllegalArgumentException(
                    "Invalid SynSEPA account ID: " + account
            );
        }

        return Integer.parseInt(account.substring(4));
    }

    private static boolean romance(
            SynSepChronologicalReader.Transaction transaction
    ) {
        return transaction.fraud()
                && "romance".equalsIgnoreCase(
                transaction.fraudType()
        );
    }

    private static byte fraudType(boolean fraud, String type) {
        if (!fraud) {
            return 0;
        }

        return switch (type.toLowerCase(Locale.ROOT)) {
            case "invoice" -> 1;
            case "impersonation" -> 2;
            case "ceo" -> 3;
            default -> throw new IllegalArgumentException(
                    "Unexpected fraud type: " + type
            );
        };
    }

    private static float parseBinary(String value) throws IOException {
        return switch (value) {
            case "0" -> 0.0f;
            case "1" -> 1.0f;
            default -> throw new IOException(
                    "Invalid binary value: " + value
            );
        };
    }

    public record Report(
            long train,
            long validation,
            long test,
            long excludedRomance
    ) {
        public long total() {
            return train + validation + test;
        }
    }

    private enum Partition {
        TRAIN, VALIDATION, TEST
    }

    private record Signal(
            String accountId,
            float value,
            float mask
    ) {
    }

    private static final class Writer implements AutoCloseable {
        private final Path path;
        private final DataOutputStream output;
        private long count;

        private Writer(Path path) throws IOException {
            this.path = path;

            output = new DataOutputStream(
                    new BufferedOutputStream(
                            Files.newOutputStream(
                                    path,
                                    StandardOpenOption.CREATE_NEW,
                                    StandardOpenOption.WRITE
                            ),
                            1024 * 1024
                    )
            );

            output.writeInt(MAGIC);
            output.writeInt(VERSION);
            output.writeInt(INPUT_SIZE);
            output.writeLong(0L);
        }

        private void write(
                int accountId,
                Instant timestamp,
                int label,
                byte type,
                float[] inputs
        ) throws IOException {
            if (inputs.length != INPUT_SIZE) {
                throw new IllegalArgumentException(
                        "Invalid v3 input size"
                );
            }

            output.writeInt(accountId);
            output.writeLong(timestamp.toEpochMilli());
            output.writeByte(label);
            output.writeByte(type);

            for (float value : inputs) {
                output.writeFloat(value);
            }

            count++;
        }

        private long count() {
            return count;
        }

        @Override
        public void close() throws IOException {
            output.close();

            try (RandomAccessFile patch =
                         new RandomAccessFile(path.toFile(), "rw")) {
                patch.seek(12L);
                patch.writeLong(count);
            }
        }
    }
}