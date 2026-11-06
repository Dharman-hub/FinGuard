package ru.finguard.training.dataset.synsepa;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import ru.finguard.training.dataset.synsepa.SynSepChronologicalReader.Transaction;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SynSepChronologicalReaderTest {

    @TempDir
    Path directory;

    @Test
    void shouldGroupTransactionsByAccountAndSortByTime()
            throws Exception {
        Path source = createDataset();

        List<List<Transaction>> accounts = readAccounts(source);

        assertEquals(2, accounts.size());
        assertEquals(2, accounts.get(0).size());
        assertEquals(1, accounts.get(1).size());

        assertEquals(
                "ACC_000000",
                accounts.get(0).get(0).accountId()
        );

        assertEquals(
                "TXN_00000001",
                accounts.get(0).get(0).transactionId()
        );

        assertEquals(
                "TXN_00000002",
                accounts.get(0).get(1).transactionId()
        );

        assertEquals(
                Instant.parse("2024-01-02T11:00:00Z"),
                accounts.get(0).get(0).timestamp()
        );

        assertEquals(
                "ACC_000001",
                accounts.get(1).get(0).accountId()
        );
    }

    @Test
    void shouldReadFraudLabelsAndRecipients() throws Exception {
        List<List<Transaction>> accounts = readAccounts(
                createDataset()
        );

        Transaction fraud = accounts.get(0).get(1);

        assertTrue(fraud.fraud());

        assertEquals("invoice", fraud.fraudType());
        assertEquals("FR222", fraud.beneficiaryIban());
        assertEquals("FR", fraud.beneficiaryCountry());
        assertEquals("eu_cross_border", fraud.countryType());
        assertEquals("150.00", fraud.amount().toPlainString());
    }

    @Test
    void shouldRejectRepeatedNoncontiguousAccount()
            throws Exception {
        Path source = directory.resolve("noncontiguous.csv");

        Files.writeString(source, HEADER
                + row("TXN_1", "ACC_1", "2024-01-01 10:00:00", "0", "none")
                + row("TXN_2", "ACC_2", "2024-01-01 11:00:00", "0", "none")
                + row("TXN_3", "ACC_1", "2024-01-01 12:00:00", "1", "romance"));

        assertThrows(
                IOException.class,
                () -> readAccounts(source)
        );
    }

    @Test
    void shouldRejectInvalidFraudLabel() throws Exception {
        Path source = directory.resolve("invalid-label.csv");

        Files.writeString(source, HEADER
                + row("TXN_1", "ACC_1", "2024-01-01 10:00:00", "2", "none"));

        assertThrows(
                IOException.class,
                () -> readAccounts(source)
        );
    }

    @Test
    void shouldRejectNonPositiveAmount() throws Exception {
        Path source = directory.resolve("invalid-amount.csv");

        Files.writeString(source, HEADER
                + "TXN_1,ACC_1,student,2024-01-01 10:00:00,"
                + "DE111,DE222,DE,domestic,0.00,transfer,"
                + "Payment,10,0,0,,1,0,none\n");

        assertThrows(
                IOException.class,
                () -> readAccounts(source)
        );
    }

    @Test
    void shouldRejectInvalidHeader() throws Exception {
        Path source = directory.resolve("invalid-header.csv");

        Files.writeString(
                source,
                "transaction_id,account_id,timestamp\n"
        );

        assertThrows(
                IOException.class,
                () -> readAccounts(source)
        );
    }

    @Test
    void shouldPreserveOriginalLineNumbers() throws Exception {
        List<List<Transaction>> accounts = readAccounts(
                createDataset()
        );

        assertEquals(
                3,
                accounts.get(0).get(0).lineNumber()
        );

        assertEquals(
                2,
                accounts.get(0).get(1).lineNumber()
        );
    }

    private Path createDataset() throws IOException {
        Path source = directory.resolve("synsepa.csv");

        Files.writeString(source, HEADER
                + "TXN_00000002,ACC_000000,student,"
                + "2024-01-03 14:10:00,DE111,FR222,FR,"
                + "eu_cross_border,150.00,transfer,"
                + "\"Invoice, rent\",14,2,0,100.0,1,1,invoice\n"
                + "TXN_00000001,ACC_000000,student,"
                + "2024-01-02 11:00:00,DE111,DE333,DE,"
                + "domestic,22.27,shopping,"
                + "Grocery shopping,11,1,0,,0,0,none\n"
                + row(
                        "TXN_00000003",
                        "ACC_000001",
                        "2024-01-04 10:00:00",
                        "0",
                        "none"
                ));

        return source;
    }

    private String row(
            String transactionId,
            String accountId,
            String timestamp,
            String label,
            String fraudType
    ) {
        return transactionId + ","
                + accountId + ",student,"
                + timestamp + ",DE111,DE222,DE,"
                + "domestic,50.00,transfer,"
                + "Peer payment,10,0,0,,1,"
                + label + ","
                + fraudType + "\n";
    }

    private List<List<Transaction>> readAccounts(Path source)
            throws IOException {
        List<List<Transaction>> result = new ArrayList<>();

        SynSepChronologicalReader.forEachAccount(
                source,
                result::add
        );

        return result;
    }

    private static final String HEADER =
            "transaction_id,account_id,persona,timestamp,"
                    + "sender_iban,beneficiary_iban,"
                    + "beneficiary_country,country_type,amount,"
                    + "remittance_category,remittance_text,"
                    + "hour_of_day,day_of_week,is_weekend,"
                    + "time_since_last_txn,is_new_beneficiary,"
                    + "is_fraud,fraud_type\n";
}