package ru.finguard.core.feature;

import java.util.Objects;

public final class RecipientAccountInputEncoder {

    public static final int BASE_FEATURE_COUNT = 34;
    public static final int BASE_INPUT_SIZE = 68;

    public static final int EXTENDED_FEATURE_COUNT = 35;
    public static final int EXTENDED_INPUT_SIZE = 70;

    public static final int NEW_SUPPLIER_IBAN_INDEX = 34;
    public static final int NEW_SUPPLIER_IBAN_MASK_INDEX = 69;

    private RecipientAccountInputEncoder() {
    }

    public static float[] encode(
            float[] baseInput,
            NewSupplierIbanDetector.Result ibanSignal
    ) {
        Objects.requireNonNull(baseInput, "Model inputs are required");
        Objects.requireNonNull(ibanSignal, "IBAN signal is required");

        if (baseInput.length != BASE_INPUT_SIZE) {
            throw new IllegalArgumentException(
                    "Expected 68 base model inputs, got " + baseInput.length
            );
        }

        for (int i = 0; i < BASE_INPUT_SIZE; i++) {
            if (!Float.isFinite(baseInput[i])) {
                throw new IllegalArgumentException(
                        "Non-finite model input at index " + i
                );
            }
        }

        for (int i = BASE_FEATURE_COUNT; i < BASE_INPUT_SIZE; i++) {
            float mask = baseInput[i];

            if (mask != 0.0f && mask != 1.0f) {
                throw new IllegalArgumentException(
                        "Invalid presence mask at index " + i
                );
            }
        }

        float[] result = new float[EXTENDED_INPUT_SIZE];

        System.arraycopy(
                baseInput,
                0,
                result,
                0,
                BASE_FEATURE_COUNT
        );

        result[NEW_SUPPLIER_IBAN_INDEX] = ibanSignal.value();

        System.arraycopy(
                baseInput,
                BASE_FEATURE_COUNT,
                result,
                EXTENDED_FEATURE_COUNT,
                BASE_FEATURE_COUNT
        );

        result[NEW_SUPPLIER_IBAN_MASK_INDEX] = ibanSignal.mask();

        return result;
    }
}