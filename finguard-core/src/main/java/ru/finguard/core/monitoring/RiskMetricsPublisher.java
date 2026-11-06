package ru.finguard.core.monitoring;

import java.util.Objects;

public final class RiskMetricsPublisher {

    private final RiskMetrics metrics;
    private final RiskMetricsSink sink;

    public RiskMetricsPublisher(
            RiskMetrics metrics,
            RiskMetricsSink sink
    ) {
        this.metrics = Objects.requireNonNull(metrics);
        this.sink = Objects.requireNonNull(sink);
    }

    public RiskMetrics.Snapshot publish() {
        RiskMetrics.Snapshot snapshot = metrics.snapshot();
        sink.publish(snapshot);
        return snapshot;
    }
}