package ru.finguard.core.monitoring;

@FunctionalInterface
public interface RiskMetricsSink {

    void publish(RiskMetrics.Snapshot snapshot);
}