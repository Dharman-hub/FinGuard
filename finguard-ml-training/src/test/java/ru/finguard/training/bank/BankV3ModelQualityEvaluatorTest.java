package ru.finguard.training.bank;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;

class BankV3ModelQualityEvaluatorTest {

    @Test
    void shouldCalibrateFiveFalsePositivesPerThousandOnValidation() {
        List<BankV3ModelQualityEvaluator.Score> validation =
                new ArrayList<>();

        for (int i = 0; i < 1000; i++) {
            validation.add(
                    new BankV3ModelQualityEvaluator.Score(0, i)
            );
        }

        validation.add(
                new BankV3ModelQualityEvaluator.Score(1, 1000.0f)
        );
        validation.add(
                new BankV3ModelQualityEvaluator.Score(1, -1.0f)
        );

        var report = BankV3ModelQualityEvaluator.evaluateScores(
                validation,
                List.of(
                        new BankV3ModelQualityEvaluator.Score(1, 1000.0f),
                        new BankV3ModelQualityEvaluator.Score(0, 999.0f),
                        new BankV3ModelQualityEvaluator.Score(0, 0.0f)
                ),
                5
        );

        assertEquals(994.0f, report.thresholdLogit());
        assertEquals(5, report.validation().fp());
        assertEquals(1, report.test().fp());
        assertEquals(1, report.test().tp());
        assertEquals(500.0, report.test().falsePositivesPerThousand());
    }

    @Test
    void shouldPreserveObservedZeroWhenHistoryIsAvailable() {
        float[] input = new float[70];

        input[9] = 0.0f;
        input[44] = 1.0f;

        BankV3ModelQualityEvaluator.applyScenario(
                input,
                BankV3ModelQualityEvaluator.Scenario.ORIGINAL
        );

        assertEquals(0.0f, input[9]);
        assertEquals(1.0f, input[44]);
    }

    @Test
    void shouldHideBothValuesAndMasksForVelocityHistory() {
        float[] input = new float[70];

        input[9] = 5.0f;
        input[44] = 1.0f;
        input[19] = 3.0f;
        input[54] = 1.0f;

        BankV3ModelQualityEvaluator.applyScenario(
                input,
                BankV3ModelQualityEvaluator.Scenario.NO_VELOCITY_HISTORY
        );

        assertEquals(0.0f, input[9]);
        assertEquals(0.0f, input[44]);
        assertEquals(3.0f, input[19]);
        assertEquals(1.0f, input[54]);
    }

    @Test
    void shouldHideRecipientHistoryAndIban() {
        float[] input = new float[70];

        input[19] = 3.0f;
        input[54] = 1.0f;
        input[34] = 1.0f;
        input[69] = 1.0f;

        BankV3ModelQualityEvaluator.applyScenario(
                input,
                BankV3ModelQualityEvaluator.Scenario.NO_RECIPIENT_HISTORY_AND_IBAN
        );

        assertEquals(0.0f, input[19]);
        assertEquals(0.0f, input[54]);
        assertEquals(0.0f, input[34]);
        assertEquals(0.0f, input[69]);
    }

    @Test
    void shouldKeepOnlyCurrentAmount() {
        float[] input = new float[70];

        input[0] = 500.0f;
        input[35] = 1.0f;
        input[9] = 4.0f;
        input[44] = 1.0f;

        BankV3ModelQualityEvaluator.applyScenario(
                input,
                BankV3ModelQualityEvaluator.Scenario.CURRENT_AMOUNT_ONLY
        );

        assertEquals(500.0f, input[0]);
        assertEquals(1.0f, input[35]);
        assertEquals(0.0f, input[9]);
        assertEquals(0.0f, input[44]);
    }
}