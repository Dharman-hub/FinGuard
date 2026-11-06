package ru.finguard.core.rule.impl;

import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.rule.RiskRule;
import ru.finguard.core.rule.RuleResult;

import java.math.BigDecimal;
import java.util.Objects;
import java.util.Optional;

public final class UnusualAmountRule implements RiskRule {

    private static final String CODE = "UNUSUAL_AMOUNT";
    private static final BigDecimal DEFAULT_MULTIPLIER = new BigDecimal("3");

    private final BigDecimal multiplier;

    public UnusualAmountRule() {
        this(DEFAULT_MULTIPLIER);
    }

    public UnusualAmountRule(BigDecimal multiplier) {
        Objects.requireNonNull(multiplier);

        if (multiplier.compareTo(BigDecimal.ONE) <= 0) {
            throw new IllegalArgumentException("Multiplier must be greater than 1");
        }

        this.multiplier = multiplier;
    }

    @Override
    public RuleResult evaluate(TransferData transfer, TransactionHistorySnapshot history) {
        Objects.requireNonNull(transfer);
        Objects.requireNonNull(history);

        Optional<BigDecimal> average = history.amountStatistics().averageAmount30d();

        if (average.isEmpty() || average.get().signum() == 0) {
            return RuleResult.insufficientData(
                    CODE,
                    "Average transfer amount is unavailable"
            );
        }

        BigDecimal threshold = average.get().multiply(multiplier);

        if (transfer.amount().compareTo(threshold) > 0) {
            return RuleResult.triggered(
                    CODE,
                    "Transfer amount exceeds the configured threshold"
            );
        }

        return RuleResult.passed(CODE);
    }
}