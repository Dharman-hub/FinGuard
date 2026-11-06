
package ru.finguard.core.feature;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NewSupplierIbanDetectorTest {

    private static final String IBAN = "DE89370400440532013000";

    @Test
    void shouldDetectNewSupplierIban() {
        var result = NewSupplierIbanDetector.detect(
                true,
                IBAN,
                true,
                false
        );

        assertEquals(1.0f, result.value());
        assertEquals(1.0f, result.mask());
        assertTrue(result.isNewIban());
    }

    @Test
    void shouldRecognizePreviouslyUsedIban() {
        var result = NewSupplierIbanDetector.detect(
                true,
                IBAN,
                true,
                true
        );

        assertEquals(0.0f, result.value());
        assertEquals(1.0f, result.mask());
        assertFalse(result.isNewIban());
    }

    @Test
    void shouldMarkMissingHistoryAsUnknown() {
        var result = NewSupplierIbanDetector.detect(
                true,
                IBAN,
                false,
                false
        );

        assertEquals(0.0f, result.value());
        assertEquals(0.0f, result.mask());
    }

    @Test
    void shouldIgnoreNonSupplierTransfer() {
        var result = NewSupplierIbanDetector.detect(
                false,
                IBAN,
                true,
                false
        );

        assertFalse(result.isAvailable());
    }

    @Test
    void shouldHandleMissingBankData() {
        var result = NewSupplierIbanDetector.detect(
                true,
                IBAN,
                null,
                null
        );

        assertFalse(result.isAvailable());
    }

    @Test
    void shouldHandleMissingIban() {
        var result = NewSupplierIbanDetector.detect(
                true,
                " ",
                true,
                false
        );

        assertFalse(result.isAvailable());
    }
}
