package ru.finguard.training.dataset.synsepa;

import org.junit.jupiter.api.Test;
import ru.finguard.core.feature.FeatureName;
import ru.finguard.core.feature.FeatureExtractor;
import ru.finguard.training.dataset.synsepa.SynSepChronologicalReader.Transaction;
import ru.finguard.training.dataset.synsepa.SynSepHistoryBuilder.PreparedTransaction;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class SynSepHistoryBuilderTest {

    @Test
    void shouldExcludeCurrentAndSameTimestampTransactionsFromHistory() {
        var transactions = List.of(
                transaction(1, "ACC_000001", "FR111", "10", 0, false),
                transaction(2, "ACC_000001", "FR222", "20", 0, false),
                transaction(3, "ACC_000001", "FR333", "30", 600, false)
        );

        var prepared = prepare(transactions);

        assertEquals(
                0.0f,
                value(prepared.get(0), FeatureName.TRANSFERS_LAST_HOUR)
        );

        assertEquals(
                0.0f,
                value(prepared.get(1), FeatureName.TRANSFERS_LAST_HOUR)
        );

        assertEquals(
                2.0f,
                value(prepared.get(2), FeatureName.TRANSFERS_LAST_HOUR)
        );

        assertEquals(
                15.0f,
                value(prepared.get(2), FeatureName.AVERAGE_AMOUNT_30D),
                0.0001f
        );
    }

    @Test
    void shouldTrackPreviousTransfersToTheSameRecipient() {
        var prepared = prepare(List.of(
                transaction(1, "ACC_000001", "FR111", "100", 0, false),
                transaction(2, "ACC_000001", "FR111", "150", 600, false),
                transaction(3, "ACC_000001", "DE222", "200", 1200, false)
        ));

        assertEquals(
                prepared.get(0).transfer().recipientId(),
                prepared.get(1).transfer().recipientId()
        );

        assertEquals(
                2L,
                prepared.get(2).transfer().recipientId()
        );

        assertEquals(
                600.0f,
                value(
                        prepared.get(1),
                        FeatureName.SECONDS_SINCE_LAST_TRANSFER_TO_RECIPIENT
                )
        );

        assertEquals(
                1.0f,
                value(
                        prepared.get(1),
                        FeatureName.RECIPIENT_SHARE_30D
                ),
                0.0001f
        );
    }

    @Test
    void shouldExpireOldTransactionsFromRollingWindows() {
        var prepared = prepare(List.of(
                transaction(1, "ACC_000001", "FR111", "100",
                        "2024-01-01T09:00:00Z", false),
                transaction(2, "ACC_000001", "FR222", "200",
                        "2024-02-01T09:00:00Z", false)
        ));

        assertEquals(
                0.0f,
                value(
                        prepared.get(1),
                        FeatureName.TRANSFERS_LAST_30_DAYS
                )
        );

        assertEquals(
                0.0f,
                mask(
                        prepared.get(1),
                        FeatureName.AVERAGE_AMOUNT_30D
                )
        );

        assertEquals(
                31.0f,
                value(
                        prepared.get(1),
                        FeatureName.HISTORY_DAYS_AVAILABLE
                )
        );
    }

    @Test
    void shouldCalculateMedianAndPercentile() {
        var prepared = prepare(List.of(
                transaction(1, "ACC_000001", "FR111", "10", 0, false),
                transaction(2, "ACC_000001", "FR222", "20", 60, false),
                transaction(3, "ACC_000001", "FR333", "30", 120, false),
                transaction(4, "ACC_000001", "FR444", "100", 180, false)
        ));

        PreparedTransaction fourth = prepared.get(3);

        assertEquals(
                20.0f,
                value(fourth, FeatureName.MEDIAN_AMOUNT_30D),
                0.0001f
        );

        assertEquals(
                30.0f,
                value(fourth, FeatureName.PERCENTILE_95_AMOUNT_30D),
                0.0001f
        );
    }

    @Test
    void shouldKeepUnavailableBankStatusesMissing() {
        var prepared = prepare(List.of(
                transaction(1, "ACC_000001", "FR111", "100", 0, true),
                transaction(2, "ACC_000001", "FR222", "200", 3600, false)
        ));

        PreparedTransaction second = prepared.get(1);

        assertEquals(
                0.0f,
                mask(
                        second,
                        FeatureName.COMPLETED_TRANSFERS_TOTAL
                )
        );

        assertEquals(
                0.0f,
                mask(
                        second,
                        FeatureName.RISK_REVIEW_TRANSFERS_30D
                )
        );
    }

    @Test
    void shouldCreateTransferWithEuroCurrencyAndStableUserId() {
        var prepared = prepare(List.of(
                transaction(1, "ACC_000001", "FR111", "100", 0, false)
        ));

        assertEquals(2L, prepared.get(0).transfer().userId());
        assertEquals(1L, prepared.get(0).transfer().recipientId());
        assertEquals("EUR", prepared.get(0).transfer().currency());
        assertEquals(
                100.0f,
                value(
                        prepared.get(0),
                        FeatureName.CURRENT_TRANSFER_AMOUNT
                )
        );
    }

    @Test
    void shouldRejectTransactionsFromDifferentAccounts() {
        var transactions = List.of(
                transaction(1, "ACC_000001", "FR111", "100", 0, false),
                transaction(2, "ACC_000002", "FR222", "200", 60, false)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> prepare(transactions)
        );
    }

    @Test
    void shouldRejectUnsortedTransactions() {
        var transactions = List.of(
                transaction(1, "ACC_000001", "FR111", "100", 600, false),
                transaction(2, "ACC_000001", "FR222", "200", 0, false)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> prepare(transactions)
        );
    }

    private List<PreparedTransaction> prepare(
            List<Transaction> transactions
    ) {
        List<PreparedTransaction> result = new ArrayList<>();

        SynSepHistoryBuilder.forEachTransaction(
                transactions,
                result::add
        );

        return result;
    }

    private float value(
            PreparedTransaction prepared,
            FeatureName feature
    ) {
        return new FeatureExtractor()
                .extract(prepared.transfer(), prepared.snapshot())
                .values()[feature.ordinal()];
    }

    private float mask(
            PreparedTransaction prepared,
            FeatureName feature
    ) {
        return new FeatureExtractor()
                .extract(prepared.transfer(), prepared.snapshot())
                .presenceMask()[feature.ordinal()];
    }

    private Transaction transaction(
            long lineNumber,
            String accountId,
            String recipient,
            String amount,
            long seconds,
            boolean fraud
    ) {
        return transaction(
                lineNumber,
                accountId,
                recipient,
                amount,
                Instant.parse("2024-01-01T09:00:00Z")
                        .plusSeconds(seconds)
                        .toString(),
                fraud
        );
    }

    private Transaction transaction(
            long lineNumber,
            String accountId,
            String recipient,
            String amount,
            String timestamp,
            boolean fraud
    ) {
        return new Transaction(
                "TXN_" + lineNumber,
                accountId,
                Instant.parse(timestamp),
                "DE111",
                recipient,
                "FR",
                "eu_cross_border",
                new BigDecimal(amount),
                "transfer",
                fraud,
                fraud ? "romance" : "none",
                lineNumber
        );
    }
}