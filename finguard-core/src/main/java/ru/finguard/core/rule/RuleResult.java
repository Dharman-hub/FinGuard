package ru.finguard.core.rule;

import java.util.Objects;

public record RuleResult(
        String code,
        RuleStatus status,
        String description
) {
    public RuleResult {
        Objects.requireNonNull(code);
        Objects.requireNonNull(status);
        Objects.requireNonNull(description);

        if (code.isBlank()) {
            throw new IllegalArgumentException("Rule code cannot be blank");
        }

        if (status == RuleStatus.TRIGGERED && description.isBlank()) {
            throw new IllegalArgumentException("Triggered rule must have a description");
        }
    }

    public static RuleResult passed(String code) {
        return new RuleResult(code, RuleStatus.PASSED, "");
    }

    public static RuleResult triggered(String code, String description) {
        return new RuleResult(code, RuleStatus.TRIGGERED, description);
    }

    public static RuleResult insufficientData(String code, String description) {
        return new RuleResult(code, RuleStatus.INSUFFICIENT_DATA, description);
    }
}
