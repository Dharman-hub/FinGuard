package ru.finguard.core.provider;

import ru.finguard.core.model.TransferData;

@FunctionalInterface
public interface ConfirmedFraudProvider {

    boolean isConfirmedFraud(TransferData transfer);
}