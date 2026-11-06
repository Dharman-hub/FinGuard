package ru.finguard.core.model;

public enum DecisionReason {
    LOW_RISK,
    RULE_REVIEW,
    NEURAL_REVIEW,
    NEURAL_REJECT,
    BANK_CONFIRMED_FRAUD,
    DEPENDENCY_UNAVAILABLE,
    INSUFFICIENT_NEURAL_DATA,
    UNSPECIFIED
}