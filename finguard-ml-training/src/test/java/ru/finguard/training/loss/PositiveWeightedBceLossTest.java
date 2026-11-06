package ru.finguard.training.loss;

import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class PositiveWeightedBceLossTest {

    @Test
    void shouldApplyHigherWeightToPositiveClass() {
        try (NDManager manager = NDManager.newBaseManager("PyTorch")) {
            var labels = manager.create(
                    new float[]{0.0f, 1.0f},
                    new Shape(2, 1)
            );

            var logits = manager.zeros(new Shape(2, 1));

            var loss = new PositiveWeightedBceLoss(3.0f);

            float result = loss.evaluate(
                    new NDList(labels),
                    new NDList(logits)
            ).getFloat();

            assertEquals(
                    2.0 * Math.log(2.0),
                    result,
                    0.0001
            );
        }
    }

    @Test
    void shouldUseNormalBceWhenWeightIsOne() {
        try (NDManager manager = NDManager.newBaseManager("PyTorch")) {
            var labels = manager.create(
                    new float[]{0.0f, 1.0f},
                    new Shape(2, 1)
            );

            var logits = manager.zeros(new Shape(2, 1));

            var loss = new PositiveWeightedBceLoss(1.0f);

            float result = loss.evaluate(
                    new NDList(labels),
                    new NDList(logits)
            ).getFloat();

            assertEquals(Math.log(2.0), result, 0.0001);
        }
    }

    @Test
    void shouldRejectInvalidWeight() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new PositiveWeightedBceLoss(0.0f)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> new PositiveWeightedBceLoss(Float.NaN)
        );
    }
}