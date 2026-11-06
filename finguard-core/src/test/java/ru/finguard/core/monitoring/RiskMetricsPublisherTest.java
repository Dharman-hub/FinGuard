package ru.finguard.core.monitoring;

import org.junit.jupiter.api.Test;
import ru.finguard.core.model.DecisionReason;
import ru.finguard.core.model.RiskAssessment;
import ru.finguard.core.model.RiskDecision;

import java.util.List;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;

class RiskMetricsPublisherTest {

    @Test
    void shouldPublishCurrentSnapshotOnDemand() {
        RiskMetrics metrics = new RiskMetrics();
        AtomicReference<RiskMetrics.Snapshot> received =
                new AtomicReference<>();

        RiskMetricsPublisher publisher =
                new RiskMetricsPublisher(metrics, received::set);

        assertNull(received.get());

        RiskAssessment assessment = new RiskAssessment(
                RiskDecision.APPROVED,
                0,
                List.of(),
                Optional.empty(),
                DecisionReason.LOW_RISK
        );

        metrics.recordSuccess(assessment, 2_000_000L);

        assertNull(received.get());

        RiskMetrics.Snapshot published = publisher.publish();

        assertSame(published, received.get());
        assertEquals(1L, published.total());
        assertEquals(1L, published.approved());
        assertEquals(0L, published.failed());
        assertEquals(2.0, published.averageLatencyMillis());
    }

    @Test
    void shouldPreserveMetricsWhenSinkFails() {
        RiskMetrics metrics = new RiskMetrics();
        metrics.recordFailure(5_000_000L);

        RiskMetricsSink sink = snapshot -> {
            throw new IllegalStateException(
                    "Monitoring system unavailable"
            );
        };

        RiskMetricsPublisher publisher =
                new RiskMetricsPublisher(metrics, sink);

        assertThrows(
                IllegalStateException.class,
                publisher::publish
        );

        assertEquals(1L, metrics.snapshot().total());
        assertEquals(1L, metrics.snapshot().failed());
        assertEquals(5.0, metrics.snapshot().maxLatencyMillis());
    }

    @Test
    void shouldRejectNullDependencies() {
        RiskMetrics metrics = new RiskMetrics();

        assertThrows(
                NullPointerException.class,
                () -> new RiskMetricsPublisher(null, snapshot -> {
                })
        );

        assertThrows(
                NullPointerException.class,
                () -> new RiskMetricsPublisher(metrics, null)
        );
    }
}