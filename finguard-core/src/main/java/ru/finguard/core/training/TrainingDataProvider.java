package ru.finguard.core.training;

import java.util.stream.Stream;

@FunctionalInterface
public interface TrainingDataProvider {

    Stream<ConfirmedTrainingTransfer> streamConfirmedTransfers();
}