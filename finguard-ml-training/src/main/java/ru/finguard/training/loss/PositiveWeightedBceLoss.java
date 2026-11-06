package ru.finguard.training.loss;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.nn.Activation;
import ai.djl.training.loss.Loss;

public final class PositiveWeightedBceLoss extends Loss {

    private final float positiveWeight;

    public PositiveWeightedBceLoss(float positiveWeight) {
        super("PositiveWeightedBceLoss");

        if (!Float.isFinite(positiveWeight) || positiveWeight < 1.0f) {
            throw new IllegalArgumentException("Positive weight must be finite and at least 1");
        }

        this.positiveWeight = positiveWeight;
    }

    @Override
    public NDArray evaluate(NDList labels, NDList predictions) {
        NDArray logits = predictions.singletonOrThrow();
        NDArray targets = labels.singletonOrThrow().reshape(logits.getShape());

        NDArray baseLoss = Activation.relu(logits)
                .sub(logits.mul(targets))
                .add(Activation.softPlus(logits.abs().neg()));

        NDArray weights = targets.mul(positiveWeight - 1.0f).add(1.0f);

        return baseLoss.mul(weights).mean();
    }
}