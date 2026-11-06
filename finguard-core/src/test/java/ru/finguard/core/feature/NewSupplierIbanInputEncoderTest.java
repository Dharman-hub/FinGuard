package ru.finguard.core.feature;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class NewSupplierIbanInputEncoderTest {

    private static final String IBAN = "DE89370400440532013000";

    @Test
    void shouldEncodeNewSupplierIban() {
        float[] original = baseInput();

        var signal = NewSupplierIbanDetector.detect(
                true,
                IBAN,
                true,
                false
        );

        float[] result = RecipientAccountInputEncoder.encode(
                original,
                signal
        );

        assertEquals(70, result.length);
        assertEquals(1.0f, result[34]);
        assertEquals(1.0f, result[69]);

        for (int i = 0; i < 34; i++) {
            assertEquals(original[i], result[i]);
            assertEquals(original[34 + i], result[35 + i]);
        }
    }

    @Test
    void shouldEncodePreviouslyUsedIban() {
        var signal = NewSupplierIbanDetector.detect(
                true,
                IBAN,
                true,
                true
        );

        float[] result = RecipientAccountInputEncoder.encode(
                baseInput(),
                signal
        );

        assertEquals(0.0f, result[34]);
        assertEquals(1.0f, result[69]);
    }

    @Test
    void shouldEncodeUnknownIbanHistory() {
        var signal = NewSupplierIbanDetector.detect(
                true,
                IBAN,
                false,
                false
        );

        float[] result = RecipientAccountInputEncoder.encode(
                baseInput(),
                signal
        );

        assertEquals(0.0f, result[34]);
        assertEquals(0.0f, result[69]);
    }

    @Test
    void shouldNotModifyOriginalInput() {
        float[] original = baseInput();
        float[] copy = original.clone();

        var signal = NewSupplierIbanDetector.detect(
                true,
                IBAN,
                true,
                false
        );

        RecipientAccountInputEncoder.encode(original, signal);

        assertArrayEquals(copy, original);
    }

    @Test
    void shouldRejectInvalidInputSize() {
        var signal = NewSupplierIbanDetector.detect(
                true,
                IBAN,
                true,
                false
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> RecipientAccountInputEncoder.encode(
                        new float[69],
                        signal
                )
        );
    }

    @Test
    void shouldRejectNonFiniteBaseInput() {
        float[] input = baseInput();
        input[10] = Float.NaN;

        var signal = new NewSupplierIbanDetector.Result(0.0f, 1.0f);

        assertThrows(
                IllegalArgumentException.class,
                () -> RecipientAccountInputEncoder.encode(input, signal)
        );
    }

    @Test
    void shouldRejectInvalidBasePresenceMask() {
        float[] input = baseInput();
        input[40] = 0.5f;

        var signal = new NewSupplierIbanDetector.Result(0.0f, 1.0f);

        assertThrows(
                IllegalArgumentException.class,
                () -> RecipientAccountInputEncoder.encode(input, signal)
        );
    }

    private float[] baseInput() {
        float[] input = new float[68];

        for (int i = 0; i < 34; i++) {
            input[i] = i + 0.25f;
            input[34 + i] = i % 2;
        }

        return input;
    }
}