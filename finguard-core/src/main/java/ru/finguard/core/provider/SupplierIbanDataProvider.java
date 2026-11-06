package ru.finguard.core.provider;

import ru.finguard.core.model.SupplierIbanContext;
import ru.finguard.core.model.TransferData;

@FunctionalInterface
public interface SupplierIbanDataProvider {

    SupplierIbanContext loadContext(TransferData transfer);
}