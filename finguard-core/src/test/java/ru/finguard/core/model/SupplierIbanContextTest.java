package ru.finguard.core.model;

import org.junit.jupiter.api.Test;
import ru.finguard.core.feature.NewSupplierIbanDetector;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SupplierIbanContextTest {

    private static final String IBAN = "DE89370400440532013000";

    @Test
    void shouldDetectNewSupplierIban() {
        SupplierIbanContext context = new SupplierIbanContext(
                true,
                IBAN,
                true,
                false
        );

        NewSupplierIbanDetector.Result result = context.toSignal();

        assertEquals(1.0f, result.value());
        assertEquals(1.0f, result.mask());
        assertTrue(result.isNewIban());
    }

    @Test
    void shouldRecognizeKnownIban() {
        SupplierIbanContext context = new SupplierIbanContext(
                true,
                IBAN,
                true,
                true
        );

        NewSupplierIbanDetector.Result result = context.toSignal();

        assertEquals(0.0f, result.value());
        assertEquals(1.0f, result.mask());
        assertFalse(result.isNewIban());
    }

    @Test
    void shouldSupportMissingHistory() {
        NewSupplierIbanDetector.Result result =
                SupplierIbanContext.unknown().toSignal();

        assertEquals(0.0f, result.value());
        assertEquals(0.0f, result.mask());
        assertFalse(result.isAvailable());
    }
}