package ru.finguard.core.monitoring;

import ru.finguard.core.api.RiskEngine;
import ru.finguard.core.model.RiskAssessment;
import ru.finguard.core.model.TransferData;

import java.util.Objects;

public final class MonitoringRiskEngine implements RiskEngine {

    private final RiskEngine delegate;
    private final RiskMetrics metrics;

    public MonitoringRiskEngine(RiskEngine delegate, RiskMetrics metrics) {
        this.delegate = Objects.requireNonNull(delegate);
        this.metrics = Objects.requireNonNull(metrics);
    }

    @Override
    public RiskAssessment evaluate(TransferData transfer) {
        long started = System.nanoTime();

        try {
            RiskAssessment assessment = Objects.requireNonNull(
                    delegate.evaluate(transfer),
                    "RiskEngine returned null"
            );

            metrics.recordSuccess(assessment, System.nanoTime() - started);
            return assessment;
        } catch (RuntimeException exception) {
            metrics.recordFailure(System.nanoTime() - started);
            throw exception;
        }
    }
}