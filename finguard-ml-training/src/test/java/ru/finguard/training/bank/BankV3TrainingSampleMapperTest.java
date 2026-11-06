package ru.finguard.training.bank;

import org.junit.jupiter.api.Test;
import ru.finguard.core.feature.FeatureName;
import ru.finguard.core.feature.NewSupplierIbanDetector;
import ru.finguard.core.feature.V3InputNormalizer;
import ru.finguard.core.model.*;
import ru.finguard.core.training.*;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.Optional;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.*;

class BankV3TrainingSampleMapperTest {

    private final BankV3TrainingSampleMapper mapper =
            new BankV3TrainingSampleMapper();

    private final Instant timestamp =
            Instant.parse("2026-10-10T10:00:00Z");

    @Test
    void shouldPrepareSeventyInputsWithIban() {
        var confirmed = new ConfirmedTrainingTransfer(
                "transfer-1",
                transfer(),
                TransactionHistorySnapshot.empty(timestamp),
                Optional.of(new NewSupplierIbanDetector.Result(1.0f, 1.0f)),
                TrainingLabel.FRAUD,
                timestamp.plusSeconds(3600)
        );

        BankV3TrainingSample sample = mapper.map(confirmed);
        float[] input = sample.modelInput();

        assertEquals(70, input.length);
        assertEquals(500.0f, input[0]);
        assertEquals(1.0f, input[35]);
        assertEquals(1.0f, input[34]);
        assertEquals(1.0f, input[69]);
        assertEquals(1, sample.label());
        assertEquals(20L, sample.userId());
    }

    @Test
    void shouldPreserveKnownZeroAndMissingMasks() {
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
                timestamp
        );

        float[] input = mapper.map(confirmed(history)).modelInput();

        int observed = FeatureName.TRANSFERS_LAST_5_MINUTES.ordinal();
        int missing = FeatureName.TRANSFERS_LAST_HOUR.ordinal();

        assertEquals(0.0f, input[observed]);
        assertEquals(1.0f, input[V3InputNormalizer.FEATURE_COUNT + observed]);

        assertEquals(0.0f, input[missing]);
        assertEquals(0.0f, input[V3InputNormalizer.FEATURE_COUNT + missing]);

        assertEquals(0.0f, input[69]);
    }

    @Test
    void shouldMapLegitimateLabelToZero() {
        var confirmed = new ConfirmedTrainingTransfer(
                "transfer-1",
                transfer(),
                TransactionHistorySnapshot.empty(timestamp),
                TrainingLabel.LEGITIMATE,
                timestamp.plusSeconds(3600)
        );

        assertEquals(0, mapper.map(confirmed).label());
    }

    @Test
    void shouldRejectHistoryFromFuture() {
        TransactionHistorySnapshot future =
                TransactionHistorySnapshot.empty(
                        timestamp.plusSeconds(60)
                );

        assertThrows(
                IllegalArgumentException.class,
                () -> mapper.map(confirmed(future))
        );
    }

    @Test
    void shouldProtectPreparedInputFromModification() {
        BankV3TrainingSample sample = mapper.map(
                confirmed(TransactionHistorySnapshot.empty(timestamp))
        );

        float[] input = sample.modelInput();
        input[0] = -100.0f;

        assertEquals(500.0f, sample.modelInput()[0]);
    }

    @Test
    void shouldCloseBankDataStream() {
        AtomicBoolean closed = new AtomicBoolean();

        TrainingDataProvider provider = () ->
                Stream.of(
                        confirmed(TransactionHistorySnapshot.empty(timestamp))
                ).onClose(() -> closed.set(true));

        try (Stream<BankV3TrainingSample> stream = mapper.stream(provider)) {
            assertEquals(1L, stream.count());
        }

        assertTrue(closed.get());
    }

    private ConfirmedTrainingTransfer confirmed(
            TransactionHistorySnapshot history
    ) {
        return new ConfirmedTrainingTransfer(
                "transfer-1",
                transfer(),
                history,
                TrainingLabel.FRAUD,
                timestamp.plusSeconds(3600)
        );
    }

    private TransferData transfer() {
        return new TransferData(
                10L,
                20L,
                30L,
                new BigDecimal("500.00"),
                "RUB",
                timestamp
        );
    }
}