package ru.finguard.training.bank;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public final class BankV3CandidateReleaseGate {

    private BankV3CandidateReleaseGate() {
    }

    public static Decision assess(
            BankV3ModelQualityEvaluator.Report report,
            Criteria criteria
    ) {
        Objects.requireNonNull(report);
        Objects.requireNonNull(criteria);

        List<String> issues = new ArrayList<>();

        if (report.fpBudgetPerThousand() != criteria.fpBudgetPerThousand()) {
            issues.add("Unexpected FP budget");
        }

        if (!Float.isFinite(report.thresholdLogit())) {
            issues.add("Invalid calibrated threshold");
        }

        var validation = report.validation();
        var test = report.test();

        checkPopulation(
                "VALIDATION",
                validation,
                criteria.minValidationLegitimate(),
                criteria.minValidationFraud(),
                issues
        );

        checkPopulation(
                "TEST",
                test,
                criteria.minTestLegitimate(),
                criteria.minTestFraud(),
                issues
        );

        if (validation == null || test == null) {
            return new Decision(
                    false,
                    report.thresholdLogit(),
                    issues
            );
        }

        if (validation.falsePositivesPerThousand()
                > criteria.fpBudgetPerThousand() + 1.0e-9) {
            issues.add("VALIDATION exceeds calibrated FP budget");
        }

        if (test.falsePositivesPerThousand()
                > criteria.maxTestFpPerThousand()) {
            issues.add("TEST exceeds allowed FP/1000");
        }

        if (test.recall() < criteria.minTestRecall()) {
            issues.add("TEST recall is too low");
        }

        if (test.averagePrecision()
                < criteria.minTestAveragePrecision()) {
            issues.add("TEST average precision is too low");
        }

        for (var scenario : BankV3ModelQualityEvaluator.Scenario.values()) {
            if (scenario == BankV3ModelQualityEvaluator.Scenario.ORIGINAL) {
                continue;
            }

            var metrics = report.missingHistory().get(scenario);

            if (metrics == null) {
                issues.add("Missing stress test: " + scenario);
                continue;
            }

            if (!Double.isFinite(metrics.recall())
                    || !Double.isFinite(metrics.precision())
                    || !Double.isFinite(metrics.averagePrecision())
                    || !Double.isFinite(metrics.falsePositivesPerThousand())) {
                issues.add("Non-finite stress metrics: " + scenario);
            }

            if (metrics.tp() + metrics.fn() != test.tp() + test.fn()
                    || metrics.fp() + metrics.tn()
                    != test.fp() + test.tn()) {
                issues.add("Stress test population differs: " + scenario);
            }
        }

        var noVelocity = report.missingHistory().get(
                BankV3ModelQualityEvaluator.Scenario.NO_VELOCITY_HISTORY
        );

        if (noVelocity != null
                && noVelocity.falsePositivesPerThousand()
                > criteria.maxNoVelocityFpPerThousand()) {
            issues.add("Too many false positives without velocity history");
        }

        var noRecipient = report.missingHistory().get(
                BankV3ModelQualityEvaluator.Scenario.NO_RECIPIENT_HISTORY_AND_IBAN
        );

        if (noRecipient != null
                && noRecipient.recall()
                < criteria.minNoRecipientRecall()) {
            issues.add("Recall is too low without recipient history");
        }

        return new Decision(
                issues.isEmpty(),
                report.thresholdLogit(),
                issues
        );
    }

    private static void checkPopulation(
            String name,
            BankV3ModelQualityEvaluator.Metrics metrics,
            long minLegitimate,
            long minFraud,
            List<String> issues
    ) {
        if (metrics == null) {
            issues.add(name + " metrics are missing");
            return;
        }

        long legitimate = metrics.fp() + metrics.tn();
        long fraud = metrics.tp() + metrics.fn();

        if (legitimate < minLegitimate) {
            issues.add(name + " has insufficient legitimate records");
        }

        if (fraud < minFraud) {
            issues.add(name + " has insufficient fraud records");
        }

        if (!Double.isFinite(metrics.recall())
                || !Double.isFinite(metrics.precision())
                || !Double.isFinite(metrics.averagePrecision())
                || !Double.isFinite(metrics.falsePositivesPerThousand())) {
            issues.add(name + " contains non-finite metrics");
        }
    }

    public record Criteria(
            int fpBudgetPerThousand,
            long minValidationLegitimate,
            long minValidationFraud,
            long minTestLegitimate,
            long minTestFraud,
            double maxTestFpPerThousand,
            double minTestRecall,
            double minTestAveragePrecision,
            double maxNoVelocityFpPerThousand,
            double minNoRecipientRecall
    ) {
        public Criteria {
            if (fpBudgetPerThousand < 1
                    || fpBudgetPerThousand >= 1000
                    || minValidationLegitimate <= 0
                    || minValidationFraud <= 0
                    || minTestLegitimate <= 0
                    || minTestFraud <= 0
                    || !validNonNegative(maxTestFpPerThousand)
                    || !validFraction(minTestRecall)
                    || !validFraction(minTestAveragePrecision)
                    || !validNonNegative(maxNoVelocityFpPerThousand)
                    || !validFraction(minNoRecipientRecall)) {
                throw new IllegalArgumentException(
                        "Invalid model acceptance criteria"
                );
            }
        }
    }

    public record Decision(
            boolean eligibleForManualReview,
            float calibratedReviewThresholdLogit,
            List<String> issues
    ) {
        public Decision {
            issues = List.copyOf(issues);
        }
    }

    private static boolean validNonNegative(double value) {
        return Double.isFinite(value) && value >= 0.0;
    }

    private static boolean validFraction(double value) {
        return Double.isFinite(value)
                && value >= 0.0
                && value <= 1.0;
    }
}