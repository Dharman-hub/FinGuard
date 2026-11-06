package ru.finguard.training.bank;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class BankV3CandidateReleaseGateTest {

    @Test
    void shouldAcceptCandidateForManualReview() {
        var decision = BankV3CandidateReleaseGate.assess(
                report(5),
                criteria()
        );

        assertTrue(decision.eligibleForManualReview());
        assertTrue(decision.issues().isEmpty());
        assertEquals(1.5f, decision.calibratedReviewThresholdLogit());
    }

    @Test
    void shouldRejectCandidateWithTooManyFalsePositives() {
        var decision = BankV3CandidateReleaseGate.assess(
                report(25),
                criteria()
        );

        assertFalse(decision.eligibleForManualReview());

        assertTrue(
                decision.issues().stream()
                        .anyMatch(issue -> issue.contains("TEST exceeds"))
        );
    }

    @Test
    void shouldRejectMissingStressTests() {
        var original = report(5);

        var incomplete = new BankV3ModelQualityEvaluator.Report(
                original.fpBudgetPerThousand(),
                original.thresholdLogit(),
                original.validation(),
                original.test(),
                Map.of()
        );

        var decision = BankV3CandidateReleaseGate.assess(
                incomplete,
                criteria()
        );

        assertFalse(decision.eligibleForManualReview());
        assertEquals(3, decision.issues().size());
    }

    private BankV3CandidateReleaseGate.Criteria criteria() {
        return new BankV3CandidateReleaseGate.Criteria(
                5,
                1000,
                100,
                1000,
                100,
                5.0,
                0.7,
                0.5,
                25.0,
                0.4
        );
    }

    @Test
    void shouldPreventReleaseWhenCandidateFailsQualityGate() {
        var original = report(5);

        var incomplete = new BankV3ModelQualityEvaluator.Report(
                original.fpBudgetPerThousand(),
                original.thresholdLogit(),
                original.validation(),
                original.test(),
                Map.of()
        );

        var failure = assertThrows(
                IllegalStateException.class,
                () -> BankV3ManualTrainingManager.requireReleaseEligibility(
                        incomplete,
                        criteria()
                )
        );

        assertTrue(
                failure.getMessage().contains("Missing stress test")
        );
    }

    @Test
    void shouldAllowCandidateWhenQualityGatePasses() {
        var decision =
                BankV3ManualTrainingManager.requireReleaseEligibility(
                        report(5),
                        criteria()
                );

        assertTrue(decision.eligibleForManualReview());
        assertEquals(1.5f, decision.calibratedReviewThresholdLogit());
    }

    @Test
    void shouldNotPublishRejectedCandidate(@TempDir Path directory) {
        Path release = directory.resolve("rejected-release");

        assertThrows(
                IllegalStateException.class,
                () -> BankV3OnnxRelease.publish(
                        directory.resolve("trained"),
                        release,
                        report(25),
                        criteria()
                )
        );

        assertFalse(Files.exists(release));
    }

    private BankV3ModelQualityEvaluator.Report report(int testFp) {
        var validation = metrics(80, 5, 995, 20, 0.8);

        var test = metrics(
                80,
                testFp,
                1000 - testFp,
                20,
                0.8
        );

        return new BankV3ModelQualityEvaluator.Report(
                5,
                1.5f,
                validation,
                test,
                Map.of(
                        BankV3ModelQualityEvaluator.Scenario.NO_VELOCITY_HISTORY,
                        metrics(70, 20, 980, 30, 0.7),

                        BankV3ModelQualityEvaluator.Scenario.NO_RECIPIENT_HISTORY_AND_IBAN,
                        metrics(50, 5, 995, 50, 0.5),

                        BankV3ModelQualityEvaluator.Scenario.CURRENT_AMOUNT_ONLY,
                        metrics(0, 0, 1000, 100, 0.1)
                )
        );
    }

    private BankV3ModelQualityEvaluator.Metrics metrics(
            long tp,
            long fp,
            long tn,
            long fn,
            double ap
    ) {
        double precision = tp + fp == 0
                ? 0.0
                : tp / (double) (tp + fp);

        double recall = tp + fn == 0
                ? 0.0
                : tp / (double) (tp + fn);

        double f1 = precision + recall == 0.0
                ? 0.0
                : 2.0 * precision * recall / (precision + recall);

        double fpPer1000 = 1000.0 * fp / (fp + tn);

        return new BankV3ModelQualityEvaluator.Metrics(
                tp, fp, tn, fn,
                precision,
                recall,
                f1,
                ap,
                fpPer1000
        );
    }
}