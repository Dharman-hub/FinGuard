package ru.finguard.core.training;

import ru.finguard.core.feature.NewSupplierIbanDetector;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;

import java.time.Instant;
import java.util.Objects;
import java.util.Optional;

public record ConfirmedTrainingTransfer(
        String transferId,
        TransferData transfer,
        TransactionHistorySnapshot history,
        Optional<NewSupplierIbanDetector.Result> ibanSignal,
        TrainingLabel label,
        Instant confirmedAt
) {
    public ConfirmedTrainingTransfer {
        if (transferId == null || transferId.isBlank()) {
            throw new IllegalArgumentException(
                    "Transfer ID is required"
            );
        }

        Objects.requireNonNull(transfer);
        Objects.requireNonNull(history);
        Objects.requireNonNull(ibanSignal);
        Objects.requireNonNull(label);
        Objects.requireNonNull(confirmedAt);
    }

    public ConfirmedTrainingTransfer(
            String transferId,
            TransferData transfer,
            TransactionHistorySnapshot history,
            TrainingLabel label,
            Instant confirmedAt
    ) {
        this(
                transferId,
                transfer,
                history,
                Optional.empty(),
                label,
                confirmedAt
        );
    }
}