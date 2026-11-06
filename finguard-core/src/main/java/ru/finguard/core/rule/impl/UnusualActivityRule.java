package ru.finguard.core.rule.impl;

import ru.finguard.core.model.ActivityStatistics;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.rule.RiskRule;
import ru.finguard.core.rule.RuleResult;

import java.util.Objects;
import java.util.OptionalDouble;
import java.util.OptionalLong;

public final class UnusualActivityRule implements RiskRule {

    private static final String CODE = "UNUSUAL_ACTIVITY";
    private static final long DEFAULT_MIN_TRANSFERS_30D = 60;
    private static final long MIN_HISTORY_DAYS = 30;
    private static final double DEFAULT_MAX_HOUR_SHARE = 0.01;

    private final long minTransfers30d;
    private final double maxHourShare;

    public UnusualActivityRule() {
        this(DEFAULT_MIN_TRANSFERS_30D, DEFAULT_MAX_HOUR_SHARE);
    }

    public UnusualActivityRule(long minTransfers30d, double maxHourShare) {
        if (minTransfers30d <= 0) {
            throw new IllegalArgumentException("Minimum transfer count must be positive");
        }

        if (!Double.isFinite(maxHourShare) || maxHourShare <= 0 || maxHourShare >= 1) {
            throw new IllegalArgumentException("Hour share threshold must be between 0 and 1");
        }

        this.minTransfers30d = minTransfers30d;
        this.maxHourShare = maxHourShare;
    }

    @Override
    public RuleResult evaluate(TransferData transfer, TransactionHistorySnapshot history) {
        Objects.requireNonNull(transfer);
        Objects.requireNonNull(history);

        ActivityStatistics activity = history.activityStatistics();

        OptionalLong transfers30d = history.velocityStatistics().transfersLast30Days();
        OptionalLong historyDays = activity.historyDaysAvailable();
        OptionalDouble hourShare = activity.currentHourTransfersShare30d();

        if (transfers30d.isEmpty() || historyDays.isEmpty() || hourShare.isEmpty()) {
            return RuleResult.insufficientData(
                    CODE,
                    "Activity statistics are unavailable"
            );
        }

        if (historyDays.getAsLong() < MIN_HISTORY_DAYS
                || transfers30d.getAsLong() < minTransfers30d) {
            return RuleResult.insufficientData(
                    CODE,
                    "Insufficient transaction history for activity analysis"
            );
        }

        if (hourShare.getAsDouble() < maxHourShare) {
            return RuleResult.triggered(
                    CODE,
                    "Transfer occurred during an unusual activity hour"
            );
        }

        return RuleResult.passed(CODE);
    }
}
