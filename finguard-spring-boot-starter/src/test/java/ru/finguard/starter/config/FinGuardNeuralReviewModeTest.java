package ru.finguard.starter.config;

import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import ru.finguard.starter.autoconfigure.FinGuardAutoConfiguration;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;

class FinGuardNeuralReviewModeTest {

    @Test
    void shouldUseLowAlertsThreshold() {
        var neural = new FinGuardProperties.Neural();
        neural.setMode(NeuralReviewMode.LOW_ALERTS);

        assertEquals(5.048477f, neural.resolveReviewThresholdLogit());
    }

    @Test
    void shouldUseBalancedThreshold() {
        var neural = new FinGuardProperties.Neural();
        neural.setMode(NeuralReviewMode.BALANCED);

        assertEquals(3.7777753f, neural.resolveReviewThresholdLogit());
    }

    @Test
    void shouldUseHighSensitivityThreshold() {
        var neural = new FinGuardProperties.Neural();
        neural.setMode(NeuralReviewMode.HIGH_SENSITIVITY);

        assertEquals(1.5145587f, neural.resolveReviewThresholdLogit());
    }

    @Test
    void shouldKeepCustomThresholdCompatible() {
        var neural = new FinGuardProperties.Neural();
        neural.setReviewThresholdLogit(2.5f);

        assertEquals(
                NeuralReviewMode.CUSTOM,
                neural.getMode()
        );

        assertEquals(2.5f, neural.resolveReviewThresholdLogit());
    }

    @Test
    void shouldRejectConflictingSettings() {
        var neural = new FinGuardProperties.Neural();
        neural.setMode(NeuralReviewMode.BALANCED);
        neural.setReviewThresholdLogit(2.5f);

        assertThrows(
                IllegalStateException.class,
                neural::resolveReviewThresholdLogit
        );
    }

    @Test
    void shouldBindModeFromSpringConfiguration() {
        new ApplicationContextRunner()
                .withConfiguration(
                        AutoConfigurations.of(
                                FinGuardAutoConfiguration.class
                        )
                )
                .withPropertyValues(
                        "finguard.neural.mode=BALANCED"
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    var neural = context
                            .getBean(FinGuardProperties.class)
                            .getNeural();

                    assertEquals(
                            NeuralReviewMode.BALANCED,
                            neural.getMode()
                    );

                    assertEquals(
                            3.7777753f,
                            neural.resolveReviewThresholdLogit()
                    );
                });
    }
}