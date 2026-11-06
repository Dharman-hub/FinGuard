package ru.finguard.training.dataset.synsepa;

import java.io.BufferedReader;
import java.io.IOException;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneOffset;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.Consumer;

public final class SynSepChronologicalReader {

    private static final List<String> EXPECTED_HEADER = List.of(
            "transaction_id",
            "account_id",
            "persona",
            "timestamp",
            "sender_iban",
            "beneficiary_iban",
            "beneficiary_country",
            "country_type",
            "amount",
            "remittance_category",
            "remittance_text",
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "time_since_last_txn",
            "is_new_beneficiary",
            "is_fraud",
            "fraud_type"
    );

    private SynSepChronologicalReader() {
    }

    public static void forEachAccount(
            Path source,
            Consumer<List<Transaction>> consumer
    ) throws IOException {
        Objects.requireNonNull(source);
        Objects.requireNonNull(consumer);

        try (BufferedReader reader = new BufferedReader(
                Files.newBufferedReader(source, StandardCharsets.UTF_8),
                1024 * 1024
        )) {
            String header = reader.readLine();

            if (header == null) {
                throw new IOException("SynSEPA dataset is empty");
            }

            String[] columns = parseCsv(header);

            if (columns.length > 0) {
                columns[0] = columns[0].replace("\uFEFF", "");
            }

            if (!EXPECTED_HEADER.equals(Arrays.asList(columns))) {
                throw new IOException(
                        "Unexpected SynSEPA CSV header"
                );
            }

            String currentAccount = null;
            List<Transaction> accountTransactions = new ArrayList<>();
            Set<String> processedAccounts = new HashSet<>();

            String line;
            long lineNumber = 1;
            long total = 0;

            while ((line = reader.readLine()) != null) {
                lineNumber++;

                if (line.isBlank()) {
                    throw new IOException(
                            "Empty CSV row at line " + lineNumber
                    );
                }

                Transaction transaction;

                try {
                    transaction = parseTransaction(
                            line,
                            lineNumber
                    );
                } catch (RuntimeException exception) {
                    throw new IOException(
                            "Invalid SynSEPA row at line "
                                    + lineNumber,
                            exception
                    );
                }

                if (!transaction.accountId().equals(currentAccount)) {
                    if (currentAccount != null) {
                        processAccount(
                                accountTransactions,
                                consumer
                        );
                    }

                    if (!processedAccounts.add(
                            transaction.accountId()
                    )) {
                        throw new IOException(
                                "Account records are not contiguous: "
                                        + transaction.accountId()
                        );
                    }

                    currentAccount = transaction.accountId();
                    accountTransactions = new ArrayList<>();
                }

                accountTransactions.add(transaction);
                total++;

                if (total % 200_000 == 0) {
                    System.out.println(
                            "Read SynSEPA transactions: " + total
                    );
                }
            }

            if (!accountTransactions.isEmpty()) {
                processAccount(
                        accountTransactions,
                        consumer
                );
            }

            System.out.println(
                    "Total SynSEPA transactions read: " + total
            );
        }
    }

    private static void processAccount(
            List<Transaction> transactions,
            Consumer<List<Transaction>> consumer
    ) {
        transactions.sort(
                Comparator.comparing(Transaction::timestamp)
                        .thenComparingLong(Transaction::lineNumber)
        );

        consumer.accept(List.copyOf(transactions));
    }

    private static Transaction parseTransaction(
            String line,
            long lineNumber
    ) {
        String[] fields = parseCsv(line);

        if (fields.length != EXPECTED_HEADER.size()) {
            throw new IllegalArgumentException(
                    "Expected 18 CSV columns, got " + fields.length
            );
        }

        String transactionId = required(fields[0]);
        String accountId = required(fields[1]);

        Instant timestamp = LocalDateTime.parse(
                required(fields[3]).replace(' ', 'T')
        ).toInstant(ZoneOffset.UTC);

        String senderIban = required(fields[4]);
        String beneficiaryIban = required(fields[5]);
        String beneficiaryCountry = required(fields[6]);
        String countryType = required(fields[7]);

        BigDecimal amount = new BigDecimal(
                required(fields[8])
        );

        if (amount.signum() <= 0) {
            throw new IllegalArgumentException(
                    "Transfer amount must be positive"
            );
        }

        String remittanceCategory = required(fields[9]);

        int fraudLabel = Integer.parseInt(
                required(fields[16])
        );

        if (fraudLabel != 0 && fraudLabel != 1) {
            throw new IllegalArgumentException(
                    "Fraud label must be 0 or 1"
            );
        }

        String fraudType = required(fields[17]);

        if ((fraudLabel == 0 && !"none".equals(fraudType))
                || (fraudLabel == 1 && "none".equals(fraudType))) {
            throw new IllegalArgumentException(
                    "Fraud label and type are inconsistent"
            );
        }

        return new Transaction(
                transactionId,
                accountId,
                timestamp,
                senderIban,
                beneficiaryIban,
                beneficiaryCountry,
                countryType,
                amount,
                remittanceCategory,
                fraudLabel == 1,
                fraudType,
                lineNumber
        );
    }

    private static String required(String value) {
        String result = value.trim();

        if (result.isEmpty()) {
            throw new IllegalArgumentException(
                    "Required CSV field is empty"
            );
        }

        return result;
    }

    private static String[] parseCsv(String line) {
        List<String> fields = new ArrayList<>();
        StringBuilder current = new StringBuilder();

        boolean quoted = false;

        for (int i = 0; i < line.length(); i++) {
            char character = line.charAt(i);

            if (character == '"') {
                if (quoted
                        && i + 1 < line.length()
                        && line.charAt(i + 1) == '"') {
                    current.append('"');
                    i++;
                } else {
                    quoted = !quoted;
                }
            } else if (character == ',' && !quoted) {
                fields.add(current.toString());
                current.setLength(0);
            } else {
                current.append(character);
            }
        }

        if (quoted) {
            throw new IllegalArgumentException(
                    "Unclosed CSV quotation"
            );
        }

        fields.add(current.toString());

        return fields.toArray(String[]::new);
    }

    public record Transaction(
            String transactionId,
            String accountId,
            Instant timestamp,
            String senderIban,
            String beneficiaryIban,
            String beneficiaryCountry,
            String countryType,
            BigDecimal amount,
            String remittanceCategory,
            boolean fraud,
            String fraudType,
            long lineNumber
    ) {
    }
}