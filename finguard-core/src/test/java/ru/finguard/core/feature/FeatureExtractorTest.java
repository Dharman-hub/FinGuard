package ru.finguard.core.feature;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import ru.finguard.core.model.ActivityStatistics;
import ru.finguard.core.model.AmountStatistics;
import ru.finguard.core.model.RecipientStatistics;
import ru.finguard.core.model.RiskHistoryStatistics;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.model.VelocityStatistics;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.Optional;
import java.util.OptionalLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class FeatureExtractorTest {

    private FeatureExtractor extractor;
    private TransferData transfer;

    @BeforeEach
    void setUp() {
        extractor = new FeatureExtractor();

        transfer = new TransferData(
                1042L,
                125L,
                458L,
                new BigDecimal("30000.00"),
                "RUB",
                Instant.parse("2026-10-24T10:00:00Z")
        );
    }

    @Test
    void shouldExtractAvailableFeatures() {
        AmountStatistics amount = new AmountStatistics(
                Optional.of(new BigDecimal("5000.00")),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty()
        );

        TransactionHistorySnapshot history = new TransactionHistorySnapshot(
                amount,
                VelocityStatistics.empty(),
                RecipientStatistics.empty(),
                ActivityStatistics.empty(),
                RiskHistoryStatistics.empty(),
                transfer.timestamp()
        );

        FeatureVector vector = extractor.extract(transfer, history);

        assertEquals(30000.0f, vector.values()[FeatureName.CURRENT_TRANSFER_AMOUNT.ordinal()]);
        assertEquals(5000.0f, vector.values()[FeatureName.AVERAGE_AMOUNT_30D.ordinal()]);
        assertEquals(6.0f, vector.values()[FeatureName.AMOUNT_TO_AVERAGE_RATIO.ordinal()]);

        assertEquals(1.0f, vector.presenceMask()[FeatureName.CURRENT_TRANSFER_AMOUNT.ordinal()]);
        assertEquals(1.0f, vector.presenceMask()[FeatureName.AVERAGE_AMOUNT_30D.ordinal()]);
        assertEquals(1.0f, vector.presenceMask()[FeatureName.AMOUNT_TO_AVERAGE_RATIO.ordinal()]);
    }

    @Test
    void shouldHandleMissingHistory() {
        TransactionHistorySnapshot history = TransactionHistorySnapshot.empty(
                transfer.timestamp()
        );

        FeatureVector vector = extractor.extract(transfer, history);

        assertEquals(30000.0f, vector.values()[FeatureName.CURRENT_TRANSFER_AMOUNT.ordinal()]);
        assertEquals(1.0f, vector.presenceMask()[FeatureName.CURRENT_TRANSFER_AMOUNT.ordinal()]);
        assertEquals(0.0f, vector.presenceMask()[FeatureName.AVERAGE_AMOUNT_30D.ordinal()]);
        assertEquals(0.0f, vector.presenceMask()[FeatureName.AMOUNT_TO_AVERAGE_RATIO.ordinal()]);
    }

    @Test
    void shouldDistinguishZeroFromMissingValue() {
        VelocityStatistics velocity = new VelocityStatistics(
                OptionalLong.of(0),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                Optional.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty()
        );

        TransactionHistorySnapshot history = new TransactionHistorySnapshot(
                AmountStatistics.empty(),
                velocity,
                RecipientStatistics.empty(),
                ActivityStatistics.empty(),
                RiskHistoryStatistics.empty(),
                transfer.timestamp()
        );

        FeatureVector vector = extractor.extract(transfer, history);

        int knownIndex = FeatureName.TRANSFERS_LAST_5_MINUTES.ordinal();
        int missingIndex = FeatureName.TRANSFERS_LAST_HOUR.ordinal();

        assertEquals(0.0f, vector.values()[knownIndex]);
        assertEquals(1.0f, vector.presenceMask()[knownIndex]);

        assertEquals(0.0f, vector.values()[missingIndex]);
        assertEquals(0.0f, vector.presenceMask()[missingIndex]);
    }

    @Test
    void shouldCreateCorrectModelInput() {
        TransactionHistorySnapshot history = TransactionHistorySnapshot.empty(
                transfer.timestamp()
        );

        FeatureVector vector = extractor.extract(transfer, history);

        int featureCount = FeatureName.values().length;

        assertEquals(34, featureCount);
        assertEquals(featureCount, vector.values().length);
        assertEquals(featureCount, vector.presenceMask().length);
        assertEquals(featureCount * 2, vector.toModelInput().length);
    }

    @Test
    void shouldProtectFeatureVectorFromModification() {
        float[] values = new float[FeatureName.values().length];
        float[] mask = new float[FeatureName.values().length];

        values[0] = 100.0f;
        mask[0] = 1.0f;

        FeatureVector vector = new FeatureVector(values, mask);

        values[0] = 200.0f;
        mask[0] = 0.0f;

        assertEquals(100.0f, vector.values()[0]);
        assertEquals(1.0f, vector.presenceMask()[0]);

        float[] extractedValues = vector.values();
        extractedValues[0] = 300.0f;

        assertEquals(100.0f, vector.values()[0]);
    }

    @Test
    void shouldRejectInvalidFeatureVector() {
        float[] values = new float[FeatureName.values().length];
        float[] mask = new float[FeatureName.values().length];

        values[0] = Float.NaN;

        assertThrows(IllegalArgumentException.class,
                () -> new FeatureVector(values, mask));
    }
}
