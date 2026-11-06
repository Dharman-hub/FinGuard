
package ru.finguard.core.feature;

public final class NewSupplierIbanDetector {

    private static final Result UNKNOWN = new Result(0.0f, 0.0f);
    private static final Result KNOWN = new Result(0.0f, 1.0f);
    private static final Result NEW = new Result(1.0f, 1.0f);

    private NewSupplierIbanDetector() {
    }

    public static Result detect(
            boolean supplierTransfer,
            String currentIban,
            Boolean hasPreviousSupplierIban,
            Boolean currentIbanUsedBefore
    ) {
        if (!supplierTransfer
                || currentIban == null
                || currentIban.isBlank()
                || hasPreviousSupplierIban == null
                || currentIbanUsedBefore == null) {
            return UNKNOWN;
        }

        if (!hasPreviousSupplierIban) {
            return UNKNOWN;
        }

        return currentIbanUsedBefore ? KNOWN : NEW;
    }

    public record Result(float value, float mask) {

        public Result {
            if ((value != 0.0f && value != 1.0f)
                    || (mask != 0.0f && mask != 1.0f)
                    || (mask == 0.0f && value != 0.0f)) {
                throw new IllegalArgumentException(
                        "Invalid supplier IBAN feature"
                );
            }
        }

        public boolean isAvailable() {
            return mask == 1.0f;
        }

        public boolean isNewIban() {
            return isAvailable() && value == 1.0f;
        }
    }
}
