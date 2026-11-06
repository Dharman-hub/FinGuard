package ru.finguard.core.decision;

import ru.finguard.core.model.RiskDecision;
import ru.finguard.core.scoring.RuleScore;

import java.util.Objects;

public final class DecisionEngine {

    private static final int DEFAULT_REVIEW_THRESHOLD = 40;
    private static final int DEFAULT_REJECT_THRESHOLD = 80;

    private final RuleDecisionThresholds thresholds;

    public DecisionEngine() {
        this(DEFAULT_REVIEW_THRESHOLD, DEFAULT_REJECT_THRESHOLD);
    }

    public DecisionEngine(int reviewThreshold, int rejectThreshold) {
        this(new RuleDecisionThresholds(
                reviewThreshold,
                rejectThreshold
        ));
    }

    public DecisionEngine(RuleDecisionThresholds thresholds) {
        this.thresholds = Objects.requireNonNull(
                thresholds,
                "Rule decision thresholds are required"
        );
    }

    public RuleDecisionThresholds.Snapshot thresholdSnapshot() {
        return thresholds.snapshot();
    }

    public RiskDecision decide(RuleScore score) {
        return decide(score, false);
    }

    public RiskDecision decide(
            RuleScore score,
            boolean bankConfirmedFraud
    ) {
        return decide(
                score,
                bankConfirmedFraud,
                thresholdSnapshot()
        );
    }

    public RiskDecision decide(
            RuleScore score,
            boolean bankConfirmedFraud,
            RuleDecisionThresholds.Snapshot applied
    ) {
        Objects.requireNonNull(score, "Rule score is required");
        Objects.requireNonNull(applied, "Threshold snapshot is required");

        if (bankConfirmedFraud
                && score.confirmedScore() >= applied.rejectThreshold()) {
            return RiskDecision.REJECTED;
        }

        if (bankConfirmedFraud
                || score.maximumScore() >= applied.reviewThreshold()) {
            return RiskDecision.REVIEW;
        }

        return RiskDecision.APPROVED;
    }

    public RiskDecision decideWithNeural(
            RuleScore score,
            Float neuralLogit,
            float neuralReviewThreshold
    ) {
        return decideWithNeural(
                score,
                neuralLogit,
                neuralReviewThreshold,
                false
        );
    }

    public RiskDecision decideWithNeural(
            RuleScore score,
            Float neuralLogit,
            float neuralReviewThreshold,
            boolean bankConfirmedFraud
    ) {
        return decideWithNeural(
                score,
                neuralLogit,
                neuralReviewThreshold,
                bankConfirmedFraud,
                thresholdSnapshot()
        );
    }

    public RiskDecision decideWithNeural(
            RuleScore score,
            Float neuralLogit,
            float neuralReviewThreshold,
            boolean bankConfirmedFraud,
            RuleDecisionThresholds.Snapshot applied
    ) {
        if (!Float.isFinite(neuralReviewThreshold)) {
            throw new IllegalArgumentException(
                    "Invalid neural review threshold"
            );
        }

        if (neuralLogit != null && !Float.isFinite(neuralLogit)) {
            throw new IllegalArgumentException(
                    "Invalid neural logit"
            );
        }

        RiskDecision ruleDecision = decide(
                score,
                bankConfirmedFraud,
                applied
        );

        if (ruleDecision == RiskDecision.REJECTED) {
            return RiskDecision.REJECTED;
        }

        if (neuralLogit == null
                || neuralLogit > neuralReviewThreshold) {
            return RiskDecision.REVIEW;
        }

        return ruleDecision;
    }

    public RiskDecision decideWithNeural(
            RuleScore score,
            Float neuralLogit,
            float neuralReviewThreshold,
            Float neuralRejectThreshold,
            boolean bankConfirmedFraud,
            RuleDecisionThresholds.Snapshot applied
    ) {
        if (!Float.isFinite(neuralReviewThreshold)) {
            throw new IllegalArgumentException(
                    "Invalid neural review threshold"
            );
        }

        if (neuralRejectThreshold != null
                && (!Float.isFinite(neuralRejectThreshold)
                || neuralRejectThreshold <= neuralReviewThreshold)) {
            throw new IllegalArgumentException(
                    "Neural reject threshold must be greater than review threshold"
            );
        }

        if (neuralLogit != null && !Float.isFinite(neuralLogit)) {
            throw new IllegalArgumentException(
                    "Invalid neural logit"
            );
        }

        RiskDecision rulesDecision = decide(
                score,
                bankConfirmedFraud,
                applied
        );

        if (rulesDecision == RiskDecision.REJECTED) {
            return RiskDecision.REJECTED;
        }

        if (neuralLogit == null) {
            return RiskDecision.REVIEW;
        }

        if (neuralRejectThreshold != null
                && neuralLogit > neuralRejectThreshold) {
            return RiskDecision.REJECTED;
        }

        if (neuralLogit > neuralReviewThreshold) {
            return RiskDecision.REVIEW;
        }

        return rulesDecision;
    }
}