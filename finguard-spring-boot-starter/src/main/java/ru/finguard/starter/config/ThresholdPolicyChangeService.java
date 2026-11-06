
package ru.finguard.starter.config;

import ru.finguard.core.decision.RuntimeThresholdPolicy;

import java.util.Objects;
import java.util.Optional;

public final class ThresholdPolicyChangeService {
    private final FinGuardSettingsManager settingsManager;
    private final SettingsAuditStore auditStore;

    public ThresholdPolicyChangeService(
            FinGuardSettingsManager settingsManager,
            SettingsAuditStore auditStore) {
        this.settingsManager = Objects.requireNonNull(settingsManager);
        this.auditStore = auditStore;
    }

    public Preview previewRuleThresholds(int reviewThreshold, int rejectThreshold) {
        RuntimeThresholdPolicy.Snapshot current =
                settingsManager.runtimePolicy().snapshot();

        return buildPreview(
                ChangeType.RULE_THRESHOLDS,
                current,
                auditStore == null ? current.version() : auditStore.lastVersion(),
                reviewThreshold,
                rejectThreshold,
                current.neuralReviewThresholdLogit()
        );
    }

    public Preview previewNeuralThreshold(float thresholdLogit) {
        RuntimeThresholdPolicy.Snapshot current =
                settingsManager.runtimePolicy().snapshot();

        return buildPreview(
                ChangeType.NEURAL_THRESHOLD,
                current,
                auditStore == null ? current.version() : auditStore.lastVersion(),
                current.reviewThreshold(),
                current.rejectThreshold(),
                Optional.of(thresholdLogit)
        );
    }

    public void activate(Preview preview, String actorId) {
        Objects.requireNonNull(preview);

        if (actorId == null || actorId.isBlank()) {
            throw new IllegalArgumentException("Actor ID is required");
        }

        RuntimeThresholdPolicy.Snapshot current =
                settingsManager.runtimePolicy().snapshot();

        long auditVersion = auditStore == null
                ? current.version()
                : auditStore.lastVersion();

        if (!current.equals(preview.before())
                || auditVersion != preview.before().version()) {
            throw new IllegalStateException(
                    "Policy changed since preview was created"
            );
        }

        Preview verified = buildPreview(
                preview.type(),
                current,
                auditVersion,
                preview.after().reviewThreshold(),
                preview.after().rejectThreshold(),
                preview.after().neuralReviewThresholdLogit()
        );

        if (!verified.equals(preview)) {
            throw new IllegalArgumentException("Invalid policy preview");
        }

        switch (preview.type()) {
            case RULE_THRESHOLDS -> settingsManager.updateRuleThresholds(
                    preview.after().reviewThreshold(),
                    preview.after().rejectThreshold(),
                    preview.before().version(),
                    actorId
            );

            case NEURAL_THRESHOLD -> settingsManager.setCustomNeuralThreshold(
                    preview.after().neuralReviewThresholdLogit().orElseThrow(),
                    preview.before().version(),
                    actorId
            );
        }
    }

    static Preview buildPreview(
            ChangeType type,
            RuntimeThresholdPolicy.Snapshot current,
            long auditVersion,
            int reviewThreshold,
            int rejectThreshold,
            Optional<Float> neuralThreshold) {

        Objects.requireNonNull(type);
        Objects.requireNonNull(current);
        Objects.requireNonNull(neuralThreshold);

        if (current.version() != auditVersion) {
            throw new IllegalStateException(
                    "Runtime policy and audit versions are inconsistent"
            );
        }

        if (current.version() >= Long.MAX_VALUE - 1) {
            throw new IllegalStateException("Policy version limit reached");
        }

        RuntimeThresholdPolicy.Snapshot proposed =
                new RuntimeThresholdPolicy.Snapshot(
                        current.version() + 1,
                        reviewThreshold,
                        rejectThreshold,
                        neuralThreshold,
                        current.neuralRejectThresholdLogit(),
                        current.neuralDataQualityPolicy()
                );

        if (current.neuralReviewThresholdLogit().isPresent()
                != proposed.neuralReviewThresholdLogit().isPresent()) {
            throw new IllegalArgumentException(
                    "Neural availability cannot be changed"
            );
        }

        switch (type) {
            case RULE_THRESHOLDS -> {
                if (!current.neuralReviewThresholdLogit().equals(
                        proposed.neuralReviewThresholdLogit())) {
                    throw new IllegalArgumentException(
                            "Neural threshold cannot change with rule thresholds"
                    );
                }
            }
            case NEURAL_THRESHOLD -> {
                if (current.reviewThreshold() != proposed.reviewThreshold()
                        || current.rejectThreshold() != proposed.rejectThreshold()) {
                    throw new IllegalArgumentException(
                            "Rule thresholds cannot change with neural threshold"
                    );
                }
            }
        }

        if (current.reviewThreshold() == proposed.reviewThreshold()
                && current.rejectThreshold() == proposed.rejectThreshold()
                && current.neuralReviewThresholdLogit().equals(
                proposed.neuralReviewThresholdLogit())) {
            throw new IllegalArgumentException(
                    "Policy values are unchanged"
            );
        }

        return new Preview(type, current, proposed);
    }

    public enum ChangeType {
        RULE_THRESHOLDS,
        NEURAL_THRESHOLD
    }

    public record Preview(
            ChangeType type,
            RuntimeThresholdPolicy.Snapshot before,
            RuntimeThresholdPolicy.Snapshot after
    ) {
        public Preview {
            Objects.requireNonNull(type);
            Objects.requireNonNull(before);
            Objects.requireNonNull(after);
        }
    }
}
