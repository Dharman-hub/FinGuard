package ru.finguard.training.dataset.synsepa;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

public final class SynSepIbanFeaturesApplication {

    private static final Path SOURCE = Path.of(
            "data", "synsep_full_dataset.csv"
    );

    private static final Path DESTINATION = Path.of(
            "data", "synsepa", "experiment-v3", "iban_features.csv"
    );

    private SynSepIbanFeaturesApplication() {
    }

    public static void main(String[] args) throws IOException {
        Path source = args.length > 0
                ? Path.of(args[0])
                : SOURCE;

        Path destination = args.length > 1
                ? Path.of(args[1])
                : DESTINATION;

        Map<String, List<Transfer>> transfersByAccount =
                new HashMap<>();

        long total = 0;

        try (BufferedReader reader = Files.newBufferedReader(
                source,
                StandardCharsets.UTF_8
        )) {
            String headerLine = reader.readLine();

            if (headerLine == null) {
                throw new IOException("Empty SynSEPA CSV");
            }

            List<String> headers = parseCsv(headerLine);

            int transactionIndex = requiredColumn(
                    headers, "transaction_id"
            );

            int accountIndex = requiredColumn(
                    headers, "account_id"
            );

            int timestampIndex = requiredColumn(
                    headers, "timestamp"
            );

            int beneficiaryIndex = requiredColumn(
                    headers, "beneficiary_iban"
            );

            int categoryIndex = requiredColumn(
                    headers, "remittance_category"
            );

            String line;

            while ((line = reader.readLine()) != null) {
                List<String> fields = parseCsv(line);

                if (fields.size() != headers.size()) {
                    throw new IOException(
                            "Wrong number of columns at data row "
                                    + (total + 1)
                    );
                }

                String transactionId = fields.get(transactionIndex);
                String accountId = fields.get(accountIndex);
                String timestamp = fields.get(timestampIndex);

                if (transactionId.isBlank()
                        || accountId.isBlank()
                        || timestamp.isBlank()) {
                    throw new IOException(
                            "Missing transaction metadata at data row "
                                    + (total + 1)
                    );
                }

                Transfer transfer = new Transfer(
                        transactionId,
                        accountId,
                        timestamp,
                        fields.get(beneficiaryIndex),
                        "supplier".equalsIgnoreCase(
                                fields.get(categoryIndex).trim()
                        ),
                        total
                );

                transfersByAccount.computeIfAbsent(
                        accountId,
                        ignored -> new ArrayList<>()
                ).add(transfer);

                total++;
            }
        }

        if (destination.getParent() != null) {
            Files.createDirectories(destination.getParent());
        }

        long available = 0;
        long newIban = 0;
        long supplierTransfers = 0;

        try (BufferedWriter writer = Files.newBufferedWriter(
                destination,
                StandardCharsets.UTF_8
        )) {
            writer.write(
                    "account_id,transaction_id,timestamp,"
                            + "NEW_SUPPLIER_IBAN_AFTER_HISTORY,"
                            + "mask_NEW_SUPPLIER_IBAN_AFTER_HISTORY"
            );

            writer.newLine();

            List<String> accounts = new ArrayList<>(
                    transfersByAccount.keySet()
            );

            accounts.sort(String::compareTo);

            for (String account : accounts) {
                List<Transfer> transfers =
                        transfersByAccount.get(account);

                transfers.sort(
                        Comparator.comparing(Transfer::timestamp)
                                .thenComparingLong(Transfer::ordinal)
                );

                Set<String> supplierIbans = new HashSet<>();

                for (Transfer transfer : transfers) {
                    int feature = 0;
                    int mask = 0;

                    if (transfer.supplier()) {
                        supplierTransfers++;

                        String iban = normalizeIban(
                                transfer.iban()
                        );

                        if (!iban.isEmpty()) {
                            if (!supplierIbans.isEmpty()) {
                                mask = 1;

                                feature = supplierIbans.contains(iban)
                                        ? 0
                                        : 1;

                                available++;
                                newIban += feature;
                            }

                            supplierIbans.add(iban);
                        }
                    }

                    writer.write(transfer.accountId());
                    writer.write(',');
                    writer.write(transfer.transactionId());
                    writer.write(',');
                    writer.write(transfer.timestamp());
                    writer.write(',');
                    writer.write(Integer.toString(feature));
                    writer.write(',');
                    writer.write(Integer.toString(mask));
                    writer.newLine();
                }
            }
        }

        System.out.println(
                "========== SYNSEPA IBAN FEATURES =========="
        );

        System.out.println("Transactions: " + total);
        System.out.println(
                "Accounts: " + transfersByAccount.size()
        );

        System.out.println(
                "Supplier transfers: " + supplierTransfers
        );

        System.out.println("Feature available: " + available);
        System.out.println(
                "New supplier IBAN after history: " + newIban
        );

        System.out.println(
                "Saved: " + destination.toAbsolutePath()
        );

        System.out.println("Completed");
    }

    private static int requiredColumn(
            List<String> headers,
            String name
    ) throws IOException {
        int index = headers.indexOf(name);

        if (index < 0) {
            throw new IOException(
                    "Missing SynSEPA column: " + name
            );
        }

        return index;
    }

    private static String normalizeIban(String iban) {
        if (iban == null) {
            return "";
        }

        StringBuilder normalized =
                new StringBuilder(iban.length());

        for (int i = 0; i < iban.length(); i++) {
            char symbol = iban.charAt(i);

            if (!Character.isWhitespace(symbol)
                    && !Character.isSpaceChar(symbol)) {
                normalized.append(
                        Character.toUpperCase(symbol)
                );
            }
        }

        return normalized.toString().toUpperCase(Locale.ROOT);
    }

    private static List<String> parseCsv(
            String line
    ) throws IOException {
        List<String> fields = new ArrayList<>();
        StringBuilder current = new StringBuilder();

        boolean inQuotes = false;

        for (int i = 0; i < line.length(); i++) {
            char ch = line.charAt(i);

            if (ch == '"') {
                if (inQuotes
                        && i + 1 < line.length()
                        && line.charAt(i + 1) == '"') {
                    current.append('"');
                    i++;
                } else {
                    inQuotes = !inQuotes;
                }
            } else if (ch == ',' && !inQuotes) {
                fields.add(current.toString());
                current.setLength(0);
            } else {
                current.append(ch);
            }
        }

        if (inQuotes) {
            throw new IOException(
                    "Unclosed quoted CSV field"
            );
        }

        fields.add(current.toString());

        return fields;
    }

    private record Transfer(
            String transactionId,
            String accountId,
            String timestamp,
            String iban,
            boolean supplier,
            long ordinal
    ) {
    }
}