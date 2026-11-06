package ru.finguard.core.feature;

import ru.finguard.core.model.ActivityStatistics;
import ru.finguard.core.model.AmountStatistics;
import ru.finguard.core.model.RecipientStatistics;
import ru.finguard.core.model.RiskHistoryStatistics;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.model.VelocityStatistics;

import java.math.BigDecimal;
import java.math.MathContext;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalLong;

public class FeatureExtractor {

    private static final int FEATURE_COUNT = FeatureName.values().length;

    public FeatureVector extract(TransferData transfer, TransactionHistorySnapshot history) {
        Objects.requireNonNull(transfer);
        Objects.requireNonNull(history);

        float[] values = new float[FEATURE_COUNT];
        float[] mask = new float[FEATURE_COUNT];

        set(values, mask, FeatureName.CURRENT_TRANSFER_AMOUNT, transfer.amount());

        AmountStatistics amount = history.amountStatistics();

        set(values, mask, FeatureName.AVERAGE_AMOUNT_30D, amount.averageAmount30d());
        set(values, mask, FeatureName.MEDIAN_AMOUNT_30D, amount.medianAmount30d());
        set(values, mask, FeatureName.MAX_AMOUNT_30D, amount.maxAmount30d());
        set(values, mask, FeatureName.PERCENTILE_95_AMOUNT_30D, amount.percentile95Amount30d());
        set(values, mask, FeatureName.STANDARD_DEVIATION_30D, amount.standardDeviation30d());
        set(values, mask, FeatureName.TOTAL_AMOUNT_24H, amount.totalAmount24h());
        set(values, mask, FeatureName.TOTAL_AMOUNT_7D, amount.totalAmount7d());

        amount.averageAmount30d()
                .filter(average -> average.signum() > 0)
                .ifPresent(average -> {
                    BigDecimal ratio = transfer.amount().divide(average, MathContext.DECIMAL64);
                    set(values, mask, FeatureName.AMOUNT_TO_AVERAGE_RATIO, ratio);
                });

        VelocityStatistics velocity = history.velocityStatistics();

        set(values, mask, FeatureName.TRANSFERS_LAST_5_MINUTES, velocity.transfersLast5Minutes());
        set(values, mask, FeatureName.TRANSFERS_LAST_HOUR, velocity.transfersLastHour());
        set(values, mask, FeatureName.TRANSFERS_LAST_24_HOURS, velocity.transfersLast24Hours());
        set(values, mask, FeatureName.TRANSFERS_LAST_7_DAYS, velocity.transfersLast7Days());
        set(values, mask, FeatureName.TRANSFERS_LAST_30_DAYS, velocity.transfersLast30Days());
        set(values, mask, FeatureName.TOTAL_AMOUNT_LAST_HOUR, velocity.totalAmountLastHour());
        set(values, mask, FeatureName.UNIQUE_RECIPIENTS_LAST_24_HOURS, velocity.uniqueRecipientsLast24Hours());
        set(values, mask, FeatureName.NEW_RECIPIENTS_LAST_24_HOURS, velocity.newRecipientsLast24Hours());
        set(values, mask, FeatureName.SECONDS_SINCE_LAST_TRANSFER, velocity.secondsSinceLastTransfer());
        set(values, mask, FeatureName.REPEATED_AMOUNTS_LAST_HOUR, velocity.repeatedAmountsLastHour());

        RecipientStatistics recipient = history.recipientStatistics();

        set(values, mask, FeatureName.COMPLETED_TRANSFERS_TO_RECIPIENT, recipient.completedTransfersToRecipient());
        set(values, mask, FeatureName.TOTAL_COMPLETED_AMOUNT_TO_RECIPIENT, recipient.totalCompletedAmountToRecipient());
        set(values, mask, FeatureName.SECONDS_SINCE_LAST_TRANSFER_TO_RECIPIENT, recipient.secondsSinceLastTransferToRecipient());
        set(values, mask, FeatureName.RECIPIENT_SHARE_30D, recipient.recipientShare30d());
        set(values, mask, FeatureName.UNIQUE_RECIPIENTS_30D, recipient.uniqueRecipients30d());

        ActivityStatistics activity = history.activityStatistics();

        set(values, mask, FeatureName.COMPLETED_TRANSFERS_TOTAL, activity.completedTransfersTotal());
        set(values, mask, FeatureName.HISTORY_DAYS_AVAILABLE, activity.historyDaysAvailable());
        set(values, mask, FeatureName.AVERAGE_DAILY_TRANSFERS_30D, activity.averageDailyTransfers30d());
        set(values, mask, FeatureName.NIGHT_TRANSFERS_SHARE_30D, activity.nightTransfersShare30d());
        set(values, mask, FeatureName.CURRENT_HOUR_TRANSFERS_SHARE_30D, activity.currentHourTransfersShare30d());
        set(values, mask, FeatureName.WEEKEND_TRANSFERS_SHARE_30D, activity.weekendTransfersShare30d());
        set(values, mask, FeatureName.SECONDS_SINCE_LAST_COMPLETED_TRANSFER, activity.secondsSinceLastCompletedTransfer());

        RiskHistoryStatistics risk = history.riskHistoryStatistics();

        set(values, mask, FeatureName.DECLINED_TRANSFERS_24H, risk.declinedTransfers24h());
        set(values, mask, FeatureName.FAILED_TRANSFERS_24H, risk.failedTransfers24h());
        set(values, mask, FeatureName.RISK_REVIEW_TRANSFERS_30D, risk.riskReviewTransfers30d());

        return new FeatureVector(values, mask);
    }

    private static void set(float[] values, float[] mask, FeatureName feature, Optional<BigDecimal> value) {
        value.ifPresent(number -> set(values, mask, feature, number));
    }

    private static void set(float[] values, float[] mask, FeatureName feature, OptionalLong value) {
        if (value.isPresent()) {
            set(values, mask, feature, (double) value.getAsLong());
        }
    }

    private static void set(float[] values, float[] mask, FeatureName feature, OptionalDouble value) {
        if (value.isPresent()) {
            set(values, mask, feature, value.getAsDouble());
        }
    }

    private static void set(float[] values, float[] mask, FeatureName feature, BigDecimal value) {
        set(values, mask, feature, value.doubleValue());
    }

    private static void set(float[] values, float[] mask, FeatureName feature, double value) {
        float converted = (float) value;

        if (!Float.isFinite(converted)) {
            throw new IllegalArgumentException("Feature value is out of range: " + feature);
        }

        int index = feature.ordinal();

        values[index] = converted;
        mask[index] = 1.0f;
    }
}
