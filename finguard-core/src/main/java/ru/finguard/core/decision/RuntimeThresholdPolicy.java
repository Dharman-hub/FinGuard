package ru.finguard.core.decision;

import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicReference;

public final class RuntimeThresholdPolicy {

    private final AtomicReference<Snapshot> active;
    private PairedMutationPermit pairedPermit;

    public RuntimeThresholdPolicy(
            long version,
            int reviewThreshold,
            int rejectThreshold,
            Float neuralReviewThresholdLogit
    ) {
        this(
                version,
                reviewThreshold,
                rejectThreshold,
                neuralReviewThresholdLogit,
                null
        );
    }

    public RuntimeThresholdPolicy(
            long version,
            int reviewThreshold,
            int rejectThreshold,
            Float neuralReviewThresholdLogit,
            Float neuralRejectThresholdLogit
    ) {
        this(
                version,
                reviewThreshold,
                rejectThreshold,
                neuralReviewThresholdLogit,
                neuralRejectThresholdLogit,
                NeuralDataQualityPolicy.unrestricted()
        );
    }

    public RuntimeThresholdPolicy(
            long version,
            int reviewThreshold,
            int rejectThreshold,
            Float neuralReviewThresholdLogit,
            Float neuralRejectThresholdLogit,
            NeuralDataQualityPolicy neuralDataQualityPolicy
    ) {
        active = new AtomicReference<>(
                new Snapshot(
                        version,
                        reviewThreshold,
                        rejectThreshold,
                        Optional.ofNullable(neuralReviewThresholdLogit),
                        Optional.ofNullable(neuralRejectThresholdLogit),
                        neuralDataQualityPolicy
                )
        );
    }

    public Snapshot snapshot() {
        return active.get();
    }

    public synchronized PairedMutationPermit restrictToPairedMutations() {
        if (pairedPermit != null) {
            throw new IllegalStateException(
                    "Paired mutation protection is already enabled"
            );
        }

        pairedPermit = new PairedMutationPermit();
        return pairedPermit;
    }

    public synchronized Snapshot activatePaired(
            PairedMutationPermit permit,
            long expectedVersion,
            long newVersion,
            int reviewThreshold,
            int rejectThreshold,
            Float neuralReviewThresholdLogit,
            Float neuralRejectThresholdLogit,
            NeuralDataQualityPolicy quality
    ) {
        if (pairedPermit == null || pairedPermit != permit) {
            throw new IllegalStateException(
                    "Invalid paired mutation permit"
            );
        }

        return activateInternal(
                expectedVersion,
                newVersion,
                reviewThreshold,
                rejectThreshold,
                neuralReviewThresholdLogit,
                neuralRejectThresholdLogit,
                quality
        );
    }

    public static final class PairedMutationPermit {
        private PairedMutationPermit() {
        }
    }

    public Snapshot activate(
            long expectedVersion,
            long newVersion,
            int reviewThreshold,
            int rejectThreshold,
            Float neuralReviewThresholdLogit
    ) {
        Snapshot previous = active.get();

        return activate(
                expectedVersion,
                newVersion,
                reviewThreshold,
                rejectThreshold,
                neuralReviewThresholdLogit,
                previous.neuralRejectThresholdLogit().orElse(null),
                previous.neuralDataQualityPolicy()
        );
    }

    public Snapshot activate(
            long expectedVersion,
            long newVersion,
            int reviewThreshold,
            int rejectThreshold,
            Float neuralReviewThresholdLogit,
            Float neuralRejectThresholdLogit
    ) {
        Snapshot previous = active.get();

        return activate(
                expectedVersion,
                newVersion,
                reviewThreshold,
                rejectThreshold,
                neuralReviewThresholdLogit,
                neuralRejectThresholdLogit,
                previous.neuralDataQualityPolicy()
        );
    }

    public synchronized Snapshot activate(
            long expectedVersion,
            long newVersion,
            int reviewThreshold,
            int rejectThreshold,
            Float neuralReviewThresholdLogit,
            Float neuralRejectThresholdLogit,
            NeuralDataQualityPolicy quality
    ) {
        if (pairedPermit != null) {
            throw new IllegalStateException(
                    "Direct policy changes are disabled in paired mode"
            );
        }

        return activateInternal(
                expectedVersion,
                newVersion,
                reviewThreshold,
                rejectThreshold,
                neuralReviewThresholdLogit,
                neuralRejectThresholdLogit,
                quality
        );
    }

    private Snapshot activateInternal(
            long expectedVersion,
            long newVersion,
            int reviewThreshold,
            int rejectThreshold,
            Float neuralReviewThresholdLogit,
            Float neuralRejectThresholdLogit,
            NeuralDataQualityPolicy quality
    ) {
        Snapshot next = new Snapshot(
                newVersion,
                reviewThreshold,
                rejectThreshold,
                Optional.ofNullable(neuralReviewThresholdLogit),
                Optional.ofNullable(neuralRejectThresholdLogit),
                quality
        );

        Snapshot previous = active.get();

        if (previous.version() != expectedVersion) {
            throw new IllegalStateException(
                    "Policy version conflict: expected "
                            + expectedVersion
                            + ", current "
                            + previous.version()
            );
        }

        if (newVersion <= expectedVersion) {
            throw new IllegalArgumentException(
                    "New policy version must be greater"
            );
        }

        if (previous.neuralReviewThresholdLogit().isPresent()
                != next.neuralReviewThresholdLogit().isPresent()) {
            throw new IllegalArgumentException(
                    "Neural availability cannot change at runtime"
            );
        }

        if (!active.compareAndSet(previous, next)) {
            throw new IllegalStateException(
                    "Policy was changed concurrently"
            );
        }

        return next;
    }

    public record Snapshot(
            long version,
            int reviewThreshold,
            int rejectThreshold,
            Optional<Float> neuralReviewThresholdLogit,
            Optional<Float> neuralRejectThresholdLogit,
            NeuralDataQualityPolicy neuralDataQualityPolicy
    ) {
        public Snapshot(
                long version,
                int reviewThreshold,
                int rejectThreshold,
                Optional<Float> neuralReviewThresholdLogit
        ) {
            this(
                    version,
                    reviewThreshold,
                    rejectThreshold,
                    neuralReviewThresholdLogit,
                    Optional.empty(),
                    NeuralDataQualityPolicy.unrestricted()
            );
        }

        public Snapshot(
                long version,
                int reviewThreshold,
                int rejectThreshold,
                Optional<Float> neuralReviewThresholdLogit,
                Optional<Float> neuralRejectThresholdLogit
        ) {
            this(
                    version,
                    reviewThreshold,
                    rejectThreshold,
                    neuralReviewThresholdLogit,
                    neuralRejectThresholdLogit,
                    NeuralDataQualityPolicy.unrestricted()
            );
        }

        public Snapshot {
            Objects.requireNonNull(neuralReviewThresholdLogit);
            Objects.requireNonNull(neuralRejectThresholdLogit);
            Objects.requireNonNull(neuralDataQualityPolicy);

            if (version < 0 || version == Long.MAX_VALUE) {
                throw new IllegalArgumentException(
                        "Invalid policy version"
                );
            }

            if (reviewThreshold <= 0
                    || reviewThreshold >= rejectThreshold
                    || rejectThreshold > 100) {
                throw new IllegalArgumentException(
                        "Invalid rule thresholds"
                );
            }

            if (neuralReviewThresholdLogit.isPresent()
                    && !Float.isFinite(
                    neuralReviewThresholdLogit.orElseThrow())) {
                throw new IllegalArgumentException(
                        "Invalid neural review threshold"
                );
            }

            if (neuralRejectThresholdLogit.isPresent()) {
                if (neuralReviewThresholdLogit.isEmpty()) {
                    throw new IllegalArgumentException(
                            "Neural rejection requires neural evaluation"
                    );
                }

                float reject = neuralRejectThresholdLogit.orElseThrow();
                float review = neuralReviewThresholdLogit.orElseThrow();

                if (!Float.isFinite(reject) || reject <= review) {
                    throw new IllegalArgumentException(
                            "Neural reject threshold must be greater than review threshold"
                    );
                }
            }
        }
    }
}