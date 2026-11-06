package ru.finguard.core.neural;

public interface NeuralPredictor {

    float predictLogit(float[] normalizedInput);
}
