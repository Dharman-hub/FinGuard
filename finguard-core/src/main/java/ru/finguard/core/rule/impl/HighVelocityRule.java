package ru.finguard.core.rule.impl;

import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.model.VelocityStatistics;
import ru.finguard.core.rule.RiskRule;
import ru.finguard.core.rule.RuleResult;

import java.util.Objects;
import java.util.OptionalLong;

public final class HighVelocityRule implements RiskRule {

    private static final String CODE = "HIGH_VELOCITY";
    private static final long DEFAULT_MAX_5_MINUTES = 5;
    private static final long DEFAULT_MAX_HOUR = 20;

    private final long maxTransfers5Minutes;
    private final long maxTransfersHour;

    public HighVelocityRule() {
        this(DEFAULT_MAX_5_MINUTES, DEFAULT_MAX_HOUR);
    }

    public HighVelocityRule(long maxTransfers5Minutes, long maxTransfersHour) {
        if (maxTransfers5Minutes <= 0 || maxTransfersHour <= 0) {
            throw new IllegalArgumentException("Transfer limits must be positive");
        }

        this.maxTransfers5Minutes = maxTransfers5Minutes;
        this.maxTransfersHour = maxTransfersHour;
    }

    @Override
    public RuleResult evaluate(TransferData transfer, TransactionHistorySnapshot history) {
        Objects.requireNonNull(transfer);
        Objects.requireNonNull(history);

        VelocityStatistics velocity = history.velocityStatistics();

        OptionalLong last5Minutes = velocity.transfersLast5Minutes();
        OptionalLong lastHour = velocity.transfersLastHour();

        if (last5Minutes.isPresent()
                && last5Minutes.getAsLong() >= maxTransfers5Minutes) {
            return RuleResult.triggered(
                    CODE,
                    "Transfer frequency exceeded the 5-minute limit"
            );
        }

        if (lastHour.isPresent()
                && lastHour.getAsLong() >= maxTransfersHour) {
            return RuleResult.triggered(
                    CODE,
                    "Transfer frequency exceeded the hourly limit"
            );
        }

        if (last5Minutes.isEmpty() || lastHour.isEmpty()) {
            return RuleResult.insufficientData(
                    CODE,
                    "Transfer frequency statistics are incomplete"
            );
        }

        return RuleResult.passed(CODE);
    }
}
