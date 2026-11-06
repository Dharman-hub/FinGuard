package ru.finguard.core.monitoring;

import ru.finguard.core.model.DecisionReason;
import ru.finguard.core.model.RiskAssessment;

import java.util.Arrays;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.LongAdder;

public final class RiskMetrics {

    private static final long[] LATENCY_BUCKET_LIMITS_NANOS = {
            100_000L,
            250_000L,
            500_000L,
            1_000_000L,
            2_000_000L,
            3_000_000L,
            5_000_000L,
            10_000_000L,
            20_000_000L,
            30_000_000L,
            50_000_000L,
            75_000_000L,
            100_000_000L,
            150_000_000L,
            250_000_000L,
            500_000_000L,
            1_000_000_000L,
            2_000_000_000L,
            5_000_000_000L,
            10_000_000_000L,
            30_000_000_000L,
            60_000_000_000L
    };

    private final LongAdder approved = new LongAdder();
    private final LongAdder review = new LongAdder();
    private final LongAdder rejected = new LongAdder();
    private final LongAdder failed = new LongAdder();
    private final LongAdder dependencyUnavailable = new LongAdder();
    private final LongAdder totalDurationNanos = new LongAdder();
    private final AtomicLong maxDurationNanos = new AtomicLong();
    private final LongAdder[] latencyBuckets =
            new LongAdder[LATENCY_BUCKET_LIMITS_NANOS.length + 1];

    public RiskMetrics() {
        for (int i = 0; i < latencyBuckets.length; i++) {
            latencyBuckets[i] = new LongAdder();
        }
    }

    void recordSuccess(RiskAssessment assessment, long durationNanos) {
        Objects.requireNonNull(assessment);
        recordDuration(durationNanos);

        switch (assessment.decision()) {
            case APPROVED -> approved.increment();
            case REVIEW -> review.increment();
            case REJECTED -> rejected.increment();
        }

        if (assessment.reason() == DecisionReason.DEPENDENCY_UNAVAILABLE) {
            dependencyUnavailable.increment();
        }
    }

    void recordFailure(long durationNanos) {
        recordDuration(durationNanos);
        failed.increment();
    }

    private void recordDuration(long durationNanos) {
        long elapsed = Math.max(0L, durationNanos);
        totalDurationNanos.add(elapsed);
        maxDurationNanos.accumulateAndGet(elapsed, Math::max);

        int index = Arrays.binarySearch(
                LATENCY_BUCKET_LIMITS_NANOS, elapsed
        );

        if (index < 0) {
            index = -index - 1;
        }

        latencyBuckets[index].increment();
    }

    public Snapshot snapshot() {
        long approvedCount = approved.sum();
        long reviewCount = review.sum();
        long rejectedCount = rejected.sum();
        long failedCount = failed.sum();
        long totalCount = approvedCount + reviewCount
                + rejectedCount + failedCount;

        double averageMillis = totalCount == 0L
                ? 0.0
                : totalDurationNanos.sum() / 1_000_000.0 / totalCount;

        long[] histogram = new long[latencyBuckets.length];
        long samples = 0L;

        for (int i = 0; i < latencyBuckets.length; i++) {
            histogram[i] = latencyBuckets[i].sum();
            samples += histogram[i];
        }

        long maximumNanos = maxDurationNanos.get();

        return new Snapshot(
                totalCount,
                approvedCount,
                reviewCount,
                rejectedCount,
                dependencyUnavailable.sum(),
                failedCount,
                averageMillis,
                maximumNanos / 1_000_000.0,
                percentileMillis(histogram, samples, 0.95, maximumNanos),
                percentileMillis(histogram, samples, 0.99, maximumNanos)
        );
    }

    private static double percentileMillis(
            long[] histogram,
            long samples,
            double percentile,
            long maximumNanos
    ) {
        if (samples == 0L) {
            return 0.0;
        }

        long rank = (long) Math.ceil(samples * percentile);
        long cumulative = 0L;

        for (int i = 0; i < histogram.length; i++) {
            cumulative += histogram[i];

            if (cumulative >= rank) {
                long boundNanos = i < LATENCY_BUCKET_LIMITS_NANOS.length
                        ? LATENCY_BUCKET_LIMITS_NANOS[i]
                        : maximumNanos;

                return Math.min(boundNanos, maximumNanos) / 1_000_000.0;
            }
        }

        return maximumNanos / 1_000_000.0;
    }

    public record Snapshot(
            long total,
            long approved,
            long review,
            long rejected,
            long dependencyUnavailable,
            long failed,
            double averageLatencyMillis,
            double maxLatencyMillis,
            double p95LatencyMillis,
            double p99LatencyMillis
    ) {
    }
}