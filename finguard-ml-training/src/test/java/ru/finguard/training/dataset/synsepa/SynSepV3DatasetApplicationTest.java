package ru.finguard.training.dataset.synsepa;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SynSepV3DatasetApplicationTest {

    @TempDir
    Path directory;

    @Test
    void shouldGenerateV3WithoutLegacyBinaryDataset() throws Exception {
        Path source = directory.resolve("synsepa.csv");
        Path features = directory.resolve("iban.csv");

        StringBuilder csv = new StringBuilder(
                "transaction_id,account_id,persona,timestamp,"
                        + "sender_iban,beneficiary_iban,"
                        + "beneficiary_country,country_type,amount,"
                        + "remittance_category,remittance_text,"
                        + "hour_of_day,day_of_week,is_weekend,"
                        + "time_since_last_txn,is_new_beneficiary,"
                        + "is_fraud,fraud_type\n"
        );

        StringBuilder iban = new StringBuilder(
                "account_id,transaction_id,timestamp,"
                        + "NEW_SUPPLIER_IBAN_AFTER_HISTORY,"
                        + "mask_NEW_SUPPLIER_IBAN_AFTER_HISTORY\n"
        );

        String[] fraudTypes = {
                "invoice",
                "romance",
                "impersonation",
                "ceo"
        };

        for (int accountIndex = 0; accountIndex < 4; accountIndex++) {
            String account = String.format(
                    Locale.ROOT, "ACC_%06d", accountIndex
            );

            for (int transactionIndex = 0;
                 transactionIndex < 2;
                 transactionIndex++) {

                String id = "TXN_"
                        + (accountIndex * 2 + transactionIndex + 1);

                String timestamp = accountIndex == 2
                        || transactionIndex == 0
                        ? "2024-01-01 09:00:00"
                        : "2024-01-02 09:00:00";

                String fraudType = transactionIndex == 0
                        ? "none"
                        : fraudTypes[accountIndex];

                csv.append(id).append(",")
                        .append(account)
                        .append(",student,")
                        .append(timestamp)
                        .append(",DE111,FR222,FR,eu_cross_border,")
                        .append(transactionIndex == 0 ? "100" : "150")
                        .append(",transfer,Payment,10,1,0,,1,")
                        .append(transactionIndex)
                        .append(",")
                        .append(fraudType)
                        .append("\n");

                iban.append(account).append(",")
                        .append(id).append(",")
                        .append(timestamp)
                        .append(",1,1\n");
            }
        }

        Files.writeString(source, csv.toString());
        Files.writeString(features, iban.toString());

        Path first = directory.resolve("first");
        Path second = directory.resolve("second");

        var report = SynSepV3DatasetApplication.generate(
                source, features, first
        );

        assertEquals(7, report.total());
        assertEquals(1, report.excludedRomance());

        var repeated = SynSepV3DatasetApplication.generate(
                source, features, second
        );

        assertEquals(report, repeated);

        long legitimate = 0;
        long invoice = 0;
        long impersonation = 0;
        long ceo = 0;

        AtomicInteger missingIban = new AtomicInteger();

        for (String split : new String[]{
                "train", "validation", "test"
        }) {
            String name = split + ".fgbin";

            assertTrue(Files.isRegularFile(first.resolve(name)));

            assertEquals(
                    -1L,
                    Files.mismatch(
                            first.resolve(name),
                            second.resolve(name)
                    )
            );

            var summary = SynSepV3BatchReader.forEachBatch(
                    first.resolve(name),
                    70,
                    4,
                    batch -> {
                        for (int i = 0; i < batch.size(); i++) {
                            float mask = batch.inputs()[i * 70 + 69];

                            if (mask == 0.0f) {
                                missingIban.incrementAndGet();

                                assertEquals(
                                        0.0f,
                                        batch.inputs()[i * 70 + 34]
                                );
                            }
                        }
                    }
            );

            legitimate += summary.legitimate();
            invoice += summary.invoice();
            impersonation += summary.impersonation();
            ceo += summary.ceo();
        }

        assertEquals(4, legitimate);
        assertEquals(1, invoice);
        assertEquals(1, impersonation);
        assertEquals(1, ceo);
        assertEquals(2, missingIban.get());
    }

    @Test
    @EnabledIfSystemProperty(
            named = "finguard.verify.synsepa",
            matches = "true"
    )
    void shouldReproduceOriginalV3DatasetExactly() throws Exception {
        Path root = Path.of("data", "synsepa");

        if (!Files.isDirectory(root)) {
            root = Path.of("finguard-ml-training", "data", "synsepa");
        }

        Path original = root.resolve("experiment-v3");
        Path regenerated = root.resolve("experiment-v3-regenerated");

        for (String split : new String[]{
                "train",
                "validation",
                "test"
        }) {
            String filename = split + ".fgbin";

            Path expected = original.resolve(filename);
            Path actual = regenerated.resolve(filename);

            assertTrue(
                    Files.isRegularFile(expected),
                    "Missing original dataset: " + expected
            );

            assertTrue(
                    Files.isRegularFile(actual),
                    "Missing regenerated dataset: " + actual
            );

            assertEquals(
                    -1L,
                    Files.mismatch(expected, actual),
                    "Dataset differs: " + filename
            );
        }
    }
}