package ru.finguard.training.evaluation;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class FraudMetricsTest {

    private final List<FraudMetrics.Prediction> predictions = List.of(
            new FraudMetrics.Prediction(1, 0.9),
            new FraudMetrics.Prediction(0, 0.8),
            new FraudMetrics.Prediction(1, 0.7),
            new FraudMetrics.Prediction(0, 0.2)
    );

    @Test
    void shouldCalculateConfusionMatrix() {
        var result = FraudMetrics.evaluate(predictions, 0.75);

        assertEquals(1, result.truePositives());
        assertEquals(1, result.falsePositives());
        assertEquals(1, result.trueNegatives());
        assertEquals(1, result.falseNegatives());
    }

    @Test
    void shouldCalculatePrecisionRecallAndF1() {
        var result = FraudMetrics.evaluate(predictions, 0.75);

        assertEquals(0.5, result.precision(), 0.000001);
        assertEquals(0.5, result.recall(), 0.000001);
        assertEquals(0.5, result.f1(), 0.000001);
        assertEquals(0.5, result.falsePositiveRate(), 0.000001);
    }

    @Test
    void shouldCalculateAveragePrecision() {
        var result = FraudMetrics.evaluate(predictions, 0.75);

        assertEquals(5.0 / 6.0, result.averagePrecision(), 0.000001);
    }

    @Test
    void shouldChooseBestThreshold() {
        double threshold = FraudMetrics.findBestThreshold(predictions);

        assertEquals(0.7, threshold, 0.000001);

        var result = FraudMetrics.evaluate(predictions, threshold);

        assertEquals(0.8, result.f1(), 0.000001);
    }

    @Test
    void shouldHandleEqualScores() {
        var sameScores = List.of(
                new FraudMetrics.Prediction(1, 0.5),
                new FraudMetrics.Prediction(0, 0.5)
        );

        var result = FraudMetrics.evaluate(sameScores, 0.5);

        assertEquals(0.5, result.averagePrecision(), 0.000001);
        assertEquals(1, result.truePositives());
        assertEquals(1, result.falsePositives());
    }

    @Test
    void shouldRejectSingleClassDataset() {
        var samples = List.of(
                new FraudMetrics.Prediction(0, 0.2),
                new FraudMetrics.Prediction(0, 0.3)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> FraudMetrics.evaluate(samples, 0.5)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> FraudMetrics.findBestThreshold(samples)
        );
    }

    @Test
    void shouldRejectInvalidScores() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new FraudMetrics.Prediction(1, Float.NaN)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> new FraudMetrics.Prediction(0, 1.5)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> FraudMetrics.evaluate(predictions, -0.1)
        );
    }
}