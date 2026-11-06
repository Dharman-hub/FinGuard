package ru.finguard.core.model;

import java.util.List;
import java.util.Objects;
import java.util.Optional;

public record RiskAssessment(
        RiskDecision decision,
        int riskScore,
        List<String> factors,
        Optional<NeuralRiskDetails> neural,
        DecisionReason reason,
        Optional<RiskEvaluationTrace> trace
) {
    public RiskAssessment {
        Objects.requireNonNull(decision, "Decision is required");
        Objects.requireNonNull(factors, "Factors are required");
        Objects.requireNonNull(neural, "Neural details are required");
        Objects.requireNonNull(reason, "Decision reason is required");
        Objects.requireNonNull(trace, "Evaluation trace is required");

        if (riskScore < 0 || riskScore > 100) {
            throw new IllegalArgumentException(
                    "Risk score must be between 0 and 100"
            );
        }

        factors = List.copyOf(factors);
    }

    public RiskAssessment(
            RiskDecision decision,
            int riskScore,
            List<String> factors,
            Optional<NeuralRiskDetails> neural,
            DecisionReason reason
    ) {
        this(
                decision,
                riskScore,
                factors,
                neural,
                reason,
                Optional.empty()
        );
    }

    public RiskAssessment(
            RiskDecision decision,
            int riskScore,
            List<String> factors
    ) {
        this(
                decision,
                riskScore,
                factors,
                Optional.empty(),
                DecisionReason.UNSPECIFIED,
                Optional.empty()
        );
    }
}