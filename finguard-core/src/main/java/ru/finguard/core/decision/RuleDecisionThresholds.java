package ru.finguard.core.decision;

import java.util.concurrent.atomic.AtomicReference;

public final class RuleDecisionThresholds {

    private final AtomicReference<Snapshot> current;

    public RuleDecisionThresholds(
            int reviewThreshold,
            int rejectThreshold
    ) {
        current = new AtomicReference<>(
                new Snapshot(reviewThreshold, rejectThreshold)
        );
    }

    public Snapshot snapshot() {
        return current.get();
    }

    public void update(
            int reviewThreshold,
            int rejectThreshold
    ) {
        current.set(new Snapshot(
                reviewThreshold,
                rejectThreshold
        ));
    }

    public record Snapshot(
            int reviewThreshold,
            int rejectThreshold
    ) {
        public Snapshot {
            if (reviewThreshold <= 0
                    || reviewThreshold >= rejectThreshold
                    || rejectThreshold > 100) {
                throw new IllegalArgumentException(
                        "Invalid risk thresholds"
                );
            }
        }
    }
}