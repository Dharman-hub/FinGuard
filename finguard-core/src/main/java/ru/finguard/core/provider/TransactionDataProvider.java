package ru.finguard.core.provider;

import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;

public interface TransactionDataProvider {
    TransactionHistorySnapshot loadSnapshot(TransferData transfer);
}
