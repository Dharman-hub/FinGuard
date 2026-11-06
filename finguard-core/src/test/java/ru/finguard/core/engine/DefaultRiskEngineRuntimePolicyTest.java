package ru.finguard.core.engine;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import ru.finguard.core.api.RiskEngine;
import ru.finguard.core.api.RiskEngineBuilder;
import ru.finguard.core.decision.RuntimeThresholdPolicy;
import ru.finguard.core.feature.FeatureName;
import ru.finguard.core.feature.V3InputNormalizer;
import ru.finguard.core.model.DecisionReason;
import ru.finguard.core.model.RiskAssessment;
import ru.finguard.core.model.RiskDecision;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.neural.NeuralPredictor;
import ru.finguard.core.neural.NeuralReviewThreshold;
import ru.finguard.core.neural.V3NeuralRiskEvaluator;
import ru.finguard.core.rule.RuleResult;
import ru.finguard.core.decision.NeuralDataQualityPolicy;
import ru.finguard.core.model.AmountStatistics;
import ru.finguard.core.model.VelocityStatistics;
import ru.finguard.core.model.RecipientStatistics;
import ru.finguard.core.model.ActivityStatistics;
import ru.finguard.core.model.RiskHistoryStatistics;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DefaultRiskEngineRuntimePolicyTest {

    @TempDir
    Path directory;

    @Test
    void shouldUseOnePolicySnapshotDuringEvaluation() throws IOException {
        RuntimeThresholdPolicy policy = new RuntimeThresholdPolicy(
                0,
                20,
                80,
                1.0f
        );

        AtomicInteger calls = new AtomicInteger();

        NeuralPredictor predictor = input -> {
            if (calls.getAndIncrement() == 0) {
                policy.activate(
                        0,
                        1,
                        50,
                        90,
                        -1.0f
                );
            }

            return 0.0f;
        };

        V3NeuralRiskEvaluator evaluator = new V3NeuralRiskEvaluator(
                createNormalizer(),
                predictor
        );

        RiskEngine engine = new RiskEngineBuilder(
                transfer -> TransactionHistorySnapshot.empty(
                        transfer.timestamp()
                )
        )
                .removeRule("UNUSUAL_AMOUNT")
                .removeRule("HIGH_VELOCITY")
                .removeRule("NEW_RECIPIENT")
                .removeRule("UNUSUAL_ACTIVITY")
                .addRule(
                        "TEST_RULE",
                        (transfer, history) -> RuleResult.triggered(
                                "TEST_RULE",
                                "Test risk"
                        ),
                        30
                )
                .neuralReview(
                        evaluator,
                        new NeuralReviewThreshold(1.0f)
                )
                .runtimeThresholdPolicy(policy)
                .build();

        RiskAssessment first = engine.evaluate(createTransfer());

        var firstTrace = first.trace().orElseThrow();

        assertEquals(RiskDecision.REVIEW, first.decision());
        assertEquals(DecisionReason.RULE_REVIEW, first.reason());
        assertEquals(20, firstTrace.reviewThreshold());
        assertEquals(80, firstTrace.rejectThreshold());
        assertEquals(
                1.0f,
                firstTrace.neuralReviewThresholdLogit().orElseThrow()
        );
        assertFalse(first.factors().contains("NEURAL_REVIEW"));

        RiskAssessment second = engine.evaluate(createTransfer());

        var secondTrace = second.trace().orElseThrow();

        assertEquals(RiskDecision.REVIEW, second.decision());
        assertEquals(DecisionReason.NEURAL_REVIEW, second.reason());
        assertEquals(50, secondTrace.reviewThreshold());
        assertEquals(90, secondTrace.rejectThreshold());
        assertEquals(
                -1.0f,
                secondTrace.neuralReviewThresholdLogit().orElseThrow()
        );
        assertTrue(second.factors().contains("NEURAL_REVIEW"));

        assertEquals(1, policy.snapshot().version());

        assertEquals(
                20,
                first.trace().orElseThrow().reviewThreshold()
        );
    }

    @Test
    void shouldAutomaticallyRejectAndAllowDisablingRejection()
            throws IOException {
        RuntimeThresholdPolicy policy = new RuntimeThresholdPolicy(
                0L, 40, 80, 1.5145587f, 4.0f
        );

        RiskEngine engine = createNeuralEngine(policy, input -> 6.0f);

        RiskAssessment first = engine.evaluate(createTransfer());

        assertEquals(RiskDecision.REJECTED, first.decision());
        assertEquals(DecisionReason.NEURAL_REJECT, first.reason());
        assertTrue(first.factors().contains("NEURAL_REJECT"));
        assertEquals(0L, first.trace().orElseThrow().policyVersion());
        assertEquals(
                4.0f,
                first.trace().orElseThrow()
                        .neuralRejectThresholdLogit().orElseThrow()
        );

        policy.activate(
                0L, 1L, 40, 80, 1.5145587f, null
        );

        RiskAssessment second = engine.evaluate(createTransfer());

        assertEquals(RiskDecision.REVIEW, second.decision());
        assertEquals(DecisionReason.NEURAL_REVIEW, second.reason());
        assertTrue(second.factors().contains("NEURAL_REVIEW"));
        assertEquals(1L, second.trace().orElseThrow().policyVersion());
        assertTrue(
                second.trace().orElseThrow()
                        .neuralRejectThresholdLogit().isEmpty()
        );

        assertFalse(
                first.trace().orElseThrow().thresholdsFingerprint()
                        .equals(second.trace().orElseThrow()
                                .thresholdsFingerprint())
        );
    }

    @Test
    void shouldUseOriginalPolicyDuringConcurrentUpdate()
            throws IOException {
        RuntimeThresholdPolicy policy = new RuntimeThresholdPolicy(
                0L, 40, 80, 1.5145587f, 4.0f
        );

        AtomicInteger calls = new AtomicInteger();

        NeuralPredictor predictor = input -> {
            if (calls.getAndIncrement() == 0) {
                policy.activate(
                        0L, 1L, 40, 80, 1.5145587f, null
                );
            }
            return 6.0f;
        };

        RiskEngine engine = createNeuralEngine(policy, predictor);

        RiskAssessment first = engine.evaluate(createTransfer());

        assertEquals(RiskDecision.REJECTED, first.decision());
        assertEquals(DecisionReason.NEURAL_REJECT, first.reason());
        assertEquals(0L, first.trace().orElseThrow().policyVersion());
        assertEquals(
                4.0f,
                first.trace().orElseThrow()
                        .neuralRejectThresholdLogit().orElseThrow()
        );

        RiskAssessment second = engine.evaluate(createTransfer());

        assertEquals(RiskDecision.REVIEW, second.decision());
        assertEquals(1L, second.trace().orElseThrow().policyVersion());
        assertTrue(
                second.trace().orElseThrow()
                        .neuralRejectThresholdLogit().isEmpty()
        );
    }

    private V3InputNormalizer createNormalizer() throws IOException {
        StringBuilder content = new StringBuilder(
                "feature,mean,std,observed_train_count\n"
        );

        FeatureName[] names = FeatureName.values();

        for (int i = 0; i < 35; i++) {
            String name = i == 34
                    ? "NEW_SUPPLIER_IBAN_AFTER_HISTORY"
                    : names[i].name();

            content.append(name)
                    .append(",0,1,100\n");
        }

        Path path = directory.resolve("normalization.csv");
        Files.writeString(path, content.toString());

        return V3InputNormalizer.load(path);
    }

    @Test
    void shouldReviewIncompleteHistoryAndRejectWhenHistoryIsSufficient()
            throws IOException {
        NeuralDataQualityPolicy quality = strictQualityPolicy();

        RuntimeThresholdPolicy policy = new RuntimeThresholdPolicy(
                0L, 40, 80, 1.5145587f, 4.0f, quality
        );

        AtomicReference<TransactionHistorySnapshot> history =
                new AtomicReference<>(
                        TransactionHistorySnapshot.empty(
                                createTransfer().timestamp()
                        )
                );

        RiskEngine engine = createQualityEngine(
                policy,
                input -> 6.0f,
                history
        );

        RiskAssessment incomplete = engine.evaluate(createTransfer());

        assertEquals(RiskDecision.REVIEW, incomplete.decision());
        assertEquals(
                DecisionReason.INSUFFICIENT_NEURAL_DATA,
                incomplete.reason()
        );
        assertTrue(
                incomplete.factors().contains("NEURAL_DATA_INSUFFICIENT")
        );

        history.set(historyWithObservedZeros());

        RiskAssessment complete = engine.evaluate(createTransfer());

        assertEquals(RiskDecision.REJECTED, complete.decision());
        assertEquals(DecisionReason.NEURAL_REJECT, complete.reason());
        assertTrue(complete.factors().contains("NEURAL_REJECT"));
        assertEquals(0L, complete.trace().orElseThrow().policyVersion());
    }

    @Test
    void shouldRequireSufficientHistoryForAutomaticApproval()
            throws IOException {
        RuntimeThresholdPolicy policy = new RuntimeThresholdPolicy(
                0L, 40, 80, 1.5145587f, 4.0f,
                strictQualityPolicy()
        );

        AtomicReference<TransactionHistorySnapshot> history =
                new AtomicReference<>(
                        TransactionHistorySnapshot.empty(
                                createTransfer().timestamp()
                        )
                );

        RiskEngine engine = createQualityEngine(
                policy,
                input -> -3.0f,
                history
        );

        RiskAssessment incomplete = engine.evaluate(createTransfer());

        assertEquals(RiskDecision.REVIEW, incomplete.decision());
        assertEquals(
                DecisionReason.INSUFFICIENT_NEURAL_DATA,
                incomplete.reason()
        );

        history.set(historyWithObservedZeros());

        RiskAssessment complete = engine.evaluate(createTransfer());

        assertEquals(RiskDecision.APPROVED, complete.decision());
        assertEquals(DecisionReason.LOW_RISK, complete.reason());
    }

    @Test
    void shouldKeepOriginalDataQualityPolicyDuringEvaluation()
            throws IOException {
        RuntimeThresholdPolicy policy = new RuntimeThresholdPolicy(
                0L, 40, 80, 1.5145587f, 4.0f,
                strictQualityPolicy()
        );

        AtomicInteger calls = new AtomicInteger();

        NeuralPredictor predictor = input -> {
            if (calls.getAndIncrement() == 0) {
                policy.activate(
                        0L,
                        1L,
                        40,
                        80,
                        1.5145587f,
                        4.0f,
                        NeuralDataQualityPolicy.unrestricted()
                );
            }

            return 6.0f;
        };

        AtomicReference<TransactionHistorySnapshot> history =
                new AtomicReference<>(
                        TransactionHistorySnapshot.empty(
                                createTransfer().timestamp()
                        )
                );

        RiskEngine engine = createQualityEngine(
                policy,
                predictor,
                history
        );

        RiskAssessment first = engine.evaluate(createTransfer());

        assertEquals(RiskDecision.REVIEW, first.decision());
        assertEquals(0L, first.trace().orElseThrow().policyVersion());

        RiskAssessment second = engine.evaluate(createTransfer());

        assertEquals(RiskDecision.REJECTED, second.decision());
        assertEquals(DecisionReason.NEURAL_REJECT, second.reason());
        assertEquals(1L, second.trace().orElseThrow().policyVersion());
    }

    private TransferData createTransfer() {
        return new TransferData(
                1L,
                10L,
                20L,
                new BigDecimal("1000.00"),
                "EUR",
                Instant.parse("2026-10-30T12:00:00Z")
        );
    }

    private RiskEngine createNeuralEngine(
            RuntimeThresholdPolicy policy,
            NeuralPredictor predictor
    ) throws IOException {
        V3NeuralRiskEvaluator evaluator = new V3NeuralRiskEvaluator(
                createNormalizer(),
                predictor
        );

        return new RiskEngineBuilder(
                transfer -> TransactionHistorySnapshot.empty(
                        transfer.timestamp()
                )
        )
                .removeRule("UNUSUAL_AMOUNT")
                .removeRule("HIGH_VELOCITY")
                .removeRule("NEW_RECIPIENT")
                .removeRule("UNUSUAL_ACTIVITY")
                .addRule(
                        "ALWAYS_PASS",
                        (transfer, history) -> RuleResult.passed(
                                "ALWAYS_PASS"
                        ),
                        10
                )
                .neuralReview(evaluator, 1.5145587f)
                .runtimeThresholdPolicy(policy)
                .build();
    }

    private NeuralDataQualityPolicy strictQualityPolicy() {
        NeuralDataQualityPolicy.Requirements requirements =
                new NeuralDataQualityPolicy.Requirements(
                        3, 1, 1, false
                );

        return new NeuralDataQualityPolicy(
                requirements,
                requirements,
                NeuralDataQualityPolicy.MissingDataAction.REVIEW
        );
    }

    private RiskEngine createQualityEngine(
            RuntimeThresholdPolicy policy,
            NeuralPredictor predictor,
            AtomicReference<TransactionHistorySnapshot> history
    ) throws IOException {
        V3NeuralRiskEvaluator evaluator = new V3NeuralRiskEvaluator(
                createNormalizer(),
                predictor
        );

        return new RiskEngineBuilder(transfer -> history.get())
                .removeRule("UNUSUAL_AMOUNT")
                .removeRule("HIGH_VELOCITY")
                .removeRule("NEW_RECIPIENT")
                .removeRule("UNUSUAL_ACTIVITY")
                .addRule(
                        "ALWAYS_PASS",
                        (transfer, snapshot) -> RuleResult.passed(
                                "ALWAYS_PASS"
                        ),
                        10
                )
                .neuralReview(evaluator, 1.5145587f)
                .runtimeThresholdPolicy(policy)
                .build();
    }

    private TransactionHistorySnapshot historyWithObservedZeros() {
        VelocityStatistics velocity = new VelocityStatistics(
                OptionalLong.of(0),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                Optional.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty()
        );

        RecipientStatistics recipient = new RecipientStatistics(
                OptionalLong.of(0),
                Optional.empty(),
                OptionalLong.empty(),
                OptionalDouble.empty(),
                OptionalLong.empty()
        );

        return new TransactionHistorySnapshot(
                AmountStatistics.empty(),
                velocity,
                recipient,
                ActivityStatistics.empty(),
                RiskHistoryStatistics.empty(),
                createTransfer().timestamp()
        );
    }
}