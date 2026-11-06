package ru.finguard.core.model;

import ru.finguard.core.feature.NewSupplierIbanDetector;

public record SupplierIbanContext(
        boolean supplierTransfer,
        String currentIban,
        Boolean hasPreviousSupplierIban,
        Boolean currentIbanUsedBefore
) {

    public static SupplierIbanContext unknown() {
        return new SupplierIbanContext(false, null, null, null);
    }

    public NewSupplierIbanDetector.Result toSignal() {
        return NewSupplierIbanDetector.detect(
                supplierTransfer,
                currentIban,
                hasPreviousSupplierIban,
                currentIbanUsedBefore
        );
    }
}