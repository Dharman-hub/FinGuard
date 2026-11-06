package ru.finguard.training.model;

import ai.djl.Model;
import ai.djl.nn.Activation;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ru.finguard.core.feature.V3InputNormalizer;

public final class FraudMlp {

    public static final int INPUT_SIZE = V3InputNormalizer.INPUT_SIZE;

    private FraudMlp() {
    }

    public static Model create() {
        Model model = Model.newInstance("finguard-mlp", "PyTorch");

        SequentialBlock network = new SequentialBlock();

        network.add(Linear.builder().setUnits(32).build());
        network.add(Activation::relu);
        network.add(Linear.builder().setUnits(16).build());
        network.add(Activation::relu);
        network.add(Linear.builder().setUnits(1).build());

        model.setBlock(network);

        return model;
    }
}