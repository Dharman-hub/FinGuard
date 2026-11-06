
package ru.finguard.training.dataset.synsepa;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.PushbackReader;
import java.io.Reader;
import java.math.BigDecimal;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

public final class SynSepDatasetAudit {
    private static final Set<String> REQUIRED_COLUMNS = Set.of(
            "transaction_id", "account_id", "timestamp", "sender_iban",
            "beneficiary_iban", "amount", "is_fraud", "fraud_type"
    );
    private static final Set<String> FRAUD_TYPES = Set.of(
            "none", "romance", "impersonation", "invoice", "ceo"
    );
    private static final Set<String> ALLOWED_FEATURE_COLUMNS = Set.of(
            "amount", "hour_of_day", "day_of_week", "is_weekend",
            "country_type", "beneficiary_country", "time_since_last_txn",
            "is_new_beneficiary"
    );
    private static final DateTimeFormatter TIMESTAMP_FORMAT =
            DateTimeFormatter.ofPattern("uuuu-MM-dd HH:mm:ss");

    private SynSepDatasetAudit() {
    }

    public record Report(long transactions, long legitimate, long fraud,
                         long accounts, Map<String, Long> fraudByType) {
        public Report {
            fraudByType = Map.copyOf(fraudByType);
        }
    }

    public static Set<String> allowedFeatureColumns() {
        return ALLOWED_FEATURE_COLUMNS;
    }

    public static void validateFeatureColumns(Set<String> selected) {
        if (!ALLOWED_FEATURE_COLUMNS.containsAll(selected)) {
            Set<String> unsupported = new HashSet<>(selected);
            unsupported.removeAll(ALLOWED_FEATURE_COLUMNS);
            throw new IllegalArgumentException("Forbidden or unverified features: " + unsupported);
        }
    }

    public static Report audit(Path csv) throws IOException {
        try (Reader source = Files.newBufferedReader(csv, StandardCharsets.UTF_8);
             CsvReader reader = new CsvReader(source)) {
            List<String> header = reader.next();
            if (header == null) {
                throw new IllegalArgumentException("Empty CSV file");
            }
            Map<String, Integer> columns = new HashMap<>();
            for (int i = 0; i < header.size(); i++) {
                String name = header.get(i);
                if (i == 0 && name.startsWith("\uFEFF")) {
                    name = name.substring(1);
                }
                if (columns.putIfAbsent(name, i) != null) {
                    throw new IllegalArgumentException("Duplicate column: " + name);
                }
            }
            if (!columns.keySet().containsAll(REQUIRED_COLUMNS)) {
                Set<String> missing = new HashSet<>(REQUIRED_COLUMNS);
                missing.removeAll(columns.keySet());
                throw new IllegalArgumentException("Missing columns: " + missing);
            }

            Set<String> transactionIds = new HashSet<>();
            Map<String, LocalDateTime> lastTimestamp = new HashMap<>();
            Map<String, Long> fraudByType = new LinkedHashMap<>();
            long transactions = 0;
            long fraud = 0;
            long legitimate = 0;
            List<String> record;

            while ((record = reader.next()) != null) {
                long number = ++transactions;
                if (record.size() != header.size()) {
                    throw invalid(number, "Column count differs from header");
                }

                String id = value(record, columns, "transaction_id");
                String account = value(record, columns, "account_id");
                String sender = value(record, columns, "sender_iban");
                String beneficiary = value(record, columns, "beneficiary_iban");

                if (id.isBlank() || account.isBlank() || sender.isBlank() || beneficiary.isBlank()) {
                    throw invalid(number, "Required identifier is blank");
                }
                if (!transactionIds.add(id)) {
                    throw invalid(number, "Duplicate transaction_id: " + id);
                }

                BigDecimal amount;
                LocalDateTime timestamp;
                try {
                    amount = new BigDecimal(value(record, columns, "amount"));
                    timestamp = LocalDateTime.parse(
                            value(record, columns, "timestamp"),
                            TIMESTAMP_FORMAT
                    );
                } catch (RuntimeException e) {
                    throw invalid(number, "Invalid amount or timestamp");
                }

                if (amount.signum() <= 0) {
                    throw invalid(number, "Amount must be positive");
                }

                LocalDateTime previous = lastTimestamp.put(account, timestamp);
                if (previous != null && timestamp.isBefore(previous)) {
                    throw invalid(number, "History is not chronological for account: " + account);
                }

                String label = value(record, columns, "is_fraud");
                String type = value(record, columns, "fraud_type");

                if (!FRAUD_TYPES.contains(type)) {
                    throw invalid(number, "Unknown fraud type: " + type);
                }

                if ("0".equals(label)) {
                    if (!"none".equals(type)) {
                        throw invalid(number, "Legitimate transfer has fraud_type " + type);
                    }
                    legitimate++;
                } else if ("1".equals(label)) {
                    if ("none".equals(type)) {
                        throw invalid(number, "Fraud transfer has fraud_type none");
                    }
                    fraud++;
                    fraudByType.merge(type, 1L, Long::sum);
                } else {
                    throw invalid(number, "Fraud label must be 0 or 1");
                }
            }

            if (transactions == 0) {
                throw new IllegalArgumentException("Dataset has no transactions");
            }

            return new Report(transactions, legitimate, fraud,
                    lastTimestamp.size(), fraudByType);
        }
    }

    private static String value(
            List<String> values,
            Map<String, Integer> columns,
            String name
    ) {
        return values.get(columns.get(name));
    }

    private static IllegalArgumentException invalid(long row, String message) {
        return new IllegalArgumentException("Record " + row + ": " + message);
    }

    public static void main(String[] args) throws IOException {
        if (args.length != 1) {
            throw new IllegalArgumentException(
                    "Usage: SynSepDatasetAudit <synsep_full_dataset.csv>"
            );
        }

        Report report = audit(Path.of(args[0]));

        System.out.println("Transactions: " + report.transactions());
        System.out.println("Legitimate: " + report.legitimate());
        System.out.println("Fraud: " + report.fraud());
        System.out.println("Accounts: " + report.accounts());
        System.out.println("Fraud types: " + report.fraudByType());
        System.out.println(
                "Fraud rate: " + (100.0 * report.fraud() / report.transactions()) + "%"
        );
    }

    private static final class CsvReader implements AutoCloseable {
        private final PushbackReader reader;

        private CsvReader(Reader source) {
            reader = new PushbackReader(new BufferedReader(source, 1 << 16), 1);
        }

        private List<String> next() throws IOException {
            List<String> values = new ArrayList<>();
            StringBuilder field = new StringBuilder();
            boolean quoted = false;
            boolean atStart = true;
            boolean justClosed = false;
            boolean consumed = false;
            int c;

            while ((c = reader.read()) != -1) {
                consumed = true;

                if (quoted) {
                    if (c == '"') {
                        int next = reader.read();
                        if (next == '"') {
                            field.append('"');
                        } else {
                            quoted = false;
                            justClosed = true;
                            if (next != -1) {
                                reader.unread(next);
                            }
                        }
                    } else {
                        field.append((char) c);
                    }
                } else if (c == ',' || c == '\n' || c == '\r') {
                    values.add(field.toString());
                    field.setLength(0);
                    atStart = true;
                    justClosed = false;

                    if (c == '\r') {
                        int next = reader.read();
                        if (next != '\n' && next != -1) {
                            reader.unread(next);
                        }
                    }

                    if (c != ',') {
                        return values;
                    }
                } else if (c == '"' && atStart) {
                    quoted = true;
                    atStart = false;
                } else {
                    if (justClosed || c == '"') {
                        throw new IllegalArgumentException("Malformed CSV quoting");
                    }
                    field.append((char) c);
                    atStart = false;
                }
            }

            if (quoted) {
                throw new IllegalArgumentException("Unclosed quoted CSV field");
            }
            if (!consumed) {
                return null;
            }

            values.add(field.toString());
            return values;
        }

        @Override
        public void close() throws IOException {
            reader.close();
        }
    }
}
