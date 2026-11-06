
package ru.finguard.training.dataset.synsepa;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class SynSepDatasetAuditTest {
    @TempDir
    Path directory;

    private static final String HEADER =
            "transaction_id,account_id,timestamp,sender_iban,beneficiary_iban,amount,is_fraud,fraud_type,remittance_text\n";

    private static final String GOOD =
            "T1,A1,2024-01-01 12:00:00,DE1,DE2,25.50,0,none,Normal transfer\n" +
            "T2,A1,2024-01-02 12:00:00,DE1,DE3,1500.00,1,impersonation,Safe account\n" +
            "T3,A2,2024-01-01 09:00:00,FR1,FR2,35.00,0,none,Peer payment\n" +
            "T4,A2,2024-01-03 09:00:00,FR1,FR3,500.00,1,romance,Gift\n";

    private Path dataset(String content) throws IOException {
        Path file = directory.resolve("synsep.csv");
        Files.writeString(file, content, StandardCharsets.UTF_8);
        return file;
    }

    @Test
    void countsOnlyTransfersAndChecksLabels() throws IOException {
        SynSepDatasetAudit.Report report =
                SynSepDatasetAudit.audit(dataset(HEADER + GOOD));

        assertEquals(4, report.transactions());
        assertEquals(2, report.legitimate());
        assertEquals(2, report.fraud());
        assertEquals(2, report.accounts());
        assertEquals(1L, report.fraudByType().get("romance"));
        assertEquals(1L, report.fraudByType().get("impersonation"));
    }

    @Test
    void handlesCommaAndDoubleQuoteInRemittanceText() throws IOException {
        String line =
                "T1,A1,2024-01-01 12:00:00,DE1,DE2,25.50,0,none,\"Normal, transfer with \"\"quote\"\"\"\n";

        assertEquals(
                1,
                SynSepDatasetAudit.audit(dataset(HEADER + line)).transactions()
        );
    }

    @Test
    void rejectsMissingRequiredColumn() throws IOException {
        String csv =
                "transaction_id,account_id,timestamp,amount,is_fraud,fraud_type\n";

        assertThrows(
                IllegalArgumentException.class,
                () -> SynSepDatasetAudit.audit(dataset(csv))
        );
    }

    @Test
    void rejectsDuplicateTransactionIds() throws IOException {
        String csv = HEADER + GOOD +
                "T4,A3,2024-01-04 09:00:00,FR1,FR3,22.00,0,none,OK\n";

        assertThrows(
                IllegalArgumentException.class,
                () -> SynSepDatasetAudit.audit(dataset(csv))
        );
    }

    @Test
    void rejectsFraudWithoutFraudType() throws IOException {
        String csv = HEADER +
                "T1,A1,2024-01-01 12:00:00,DE1,DE2,50.00,1,none,Unknown\n";

        assertThrows(
                IllegalArgumentException.class,
                () -> SynSepDatasetAudit.audit(dataset(csv))
        );
    }

    @Test
    void rejectsNonPositiveAmount() throws IOException {
        String csv = HEADER +
                "T1,A1,2024-01-01 12:00:00,DE1,DE2,0,0,none,OK\n";

        assertThrows(
                IllegalArgumentException.class,
                () -> SynSepDatasetAudit.audit(dataset(csv))
        );
    }

    @Test
    void rejectsFutureHistoryOutOfOrderWithinAccount() throws IOException {
        String csv = HEADER +
                "T1,A1,2024-02-01 12:00:00,DE1,DE2,100,0,none,OK\n" +
                "T2,A1,2024-01-01 12:00:00,DE1,DE2,100,0,none,OK\n";

        assertThrows(
                IllegalArgumentException.class,
                () -> SynSepDatasetAudit.audit(dataset(csv))
        );
    }

    @Test
    void allowsTransactionsFromDifferentAccountsInArbitraryOrder() throws IOException {
        String csv = HEADER +
                "T1,A1,2024-02-01 12:00:00,DE1,DE2,100,0,none,OK\n" +
                "T2,A2,2024-01-01 12:00:00,DE3,DE4,100,0,none,OK\n";

        assertEquals(
                2,
                SynSepDatasetAudit.audit(dataset(csv)).accounts()
        );
    }

    @Test
    void rejectsFraudLabelsAndTextFromFeatureSet() {
        assertDoesNotThrow(
                () -> SynSepDatasetAudit.validateFeatureColumns(
                        Set.of("amount", "hour_of_day")
                )
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> SynSepDatasetAudit.validateFeatureColumns(
                        Set.of("amount", "is_fraud")
                )
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> SynSepDatasetAudit.validateFeatureColumns(
                        Set.of("fraud_type")
                )
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> SynSepDatasetAudit.validateFeatureColumns(
                        Set.of("remittance_text")
                )
        );
    }
}
