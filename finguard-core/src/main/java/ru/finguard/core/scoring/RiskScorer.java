package ru.finguard.core.scoring;

import ru.finguard.core.rule.RuleResult;
import ru.finguard.core.rule.RuleStatus;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public final class RiskScorer {

    private static final Map<String, Integer> DEFAULT_WEIGHTS = Map.of(
            "UNUSUAL_AMOUNT", 25,
            "HIGH_VELOCITY", 35,
            "NEW_RECIPIENT", 20,
            "UNUSUAL_ACTIVITY", 20
    );

    private final Map<String, Integer> weights;

    public RiskScorer() {
        this(DEFAULT_WEIGHTS);
    }

    public RiskScorer(Map<String, Integer> weights) {
        Objects.requireNonNull(weights);

        if (weights.isEmpty()) {
            throw new IllegalArgumentException("At least one rule weight is required");
        }

        for (Map.Entry<String, Integer> entry : weights.entrySet()) {
            String code = entry.getKey();
            Integer weight = entry.getValue();

            if (code == null || code.isBlank()) {
                throw new IllegalArgumentException("Rule code cannot be blank");
            }

            if (weight == null || weight <= 0 || weight > 100) {
                throw new IllegalArgumentException("Rule weight must be between 1 and 100");
            }
        }

        this.weights = Map.copyOf(weights);
    }

    public RuleScore calculate(List<RuleResult> results) {
        Objects.requireNonNull(results);

        if (results.size() != weights.size()) {
            throw new IllegalArgumentException("Results must match configured rules");
        }

        int confirmedScore = 0;
        int maximumScore = 0;

        List<String> triggeredRules = new ArrayList<>();
        List<String> missingDataRules = new ArrayList<>();
        Set<String> processedRules = new HashSet<>();

        for (RuleResult result : results) {
            Objects.requireNonNull(result);

            if (!processedRules.add(result.code())) {
                throw new IllegalArgumentException("Duplicate rule result: " + result.code());
            }

            Integer weight = weights.get(result.code());

            if (weight == null) {
                throw new IllegalArgumentException("Unknown rule: " + result.code());
            }

            if (result.status() == RuleStatus.TRIGGERED) {
                confirmedScore = Math.min(100, confirmedScore + weight);
                maximumScore = Math.min(100, maximumScore + weight);
                triggeredRules.add(result.code());
            } else if (result.status() == RuleStatus.INSUFFICIENT_DATA) {
                maximumScore = Math.min(100, maximumScore + weight);
                missingDataRules.add(result.code());
            }
        }

        return new RuleScore(
                confirmedScore,
                maximumScore,
                triggeredRules,
                missingDataRules
        );
    }
}
