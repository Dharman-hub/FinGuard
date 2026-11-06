package ru.finguard.training.bank;

import ru.finguard.core.feature.FeatureExtractor;
import ru.finguard.core.feature.NewSupplierIbanDetector;
import ru.finguard.core.feature.RecipientAccountInputEncoder;
import ru.finguard.core.training.ConfirmedTrainingTransfer;
import ru.finguard.core.training.TrainingDataProvider;
import ru.finguard.core.training.TrainingLabel;

import java.util.Objects;
import java.util.stream.Stream;

public final class BankV3TrainingSampleMapper {

    private static final NewSupplierIbanDetector.Result UNKNOWN_IBAN =
            new NewSupplierIbanDetector.Result(0.0f, 0.0f);

    private final FeatureExtractor extractor = new FeatureExtractor();

    public BankV3TrainingSample map(ConfirmedTrainingTransfer confirmed) {
        Objects.requireNonNull(confirmed);

        var transfer = confirmed.transfer();
        var history = confirmed.history();

        if (history.calculatedAt().isAfter(transfer.timestamp())) {
            throw new IllegalArgumentException("History snapshot contains future information");
        }

        if (confirmed.confirmedAt().isBefore(transfer.timestamp())) {
            throw new IllegalArgumentException("Confirmation predates transfer");
        }

        float[] baseInput = extractor.extract(transfer, history).toModelInput();

        float[] modelInput = RecipientAccountInputEncoder.encode(
                baseInput,
                confirmed.ibanSignal().orElse(UNKNOWN_IBAN)
        );

        int label = confirmed.label() == TrainingLabel.FRAUD ? 1 : 0;

        return new BankV3TrainingSample(
                confirmed.transferId(),
                transfer.userId(),
                transfer.timestamp(),
                confirmed.confirmedAt(),
                label,
                modelInput
        );
    }

    public Stream<BankV3TrainingSample> stream(TrainingDataProvider provider) {
        Objects.requireNonNull(provider);

        Stream<ConfirmedTrainingTransfer> source = Objects.requireNonNull(
                provider.streamConfirmedTransfers(),
                "Training data stream is required"
        );

        return source.map(this::map);
    }
}