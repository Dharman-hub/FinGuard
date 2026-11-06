package ru.finguard.core.engine;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import ru.finguard.core.api.RiskEngine;
import ru.finguard.core.api.RiskEngineBuilder;
import ru.finguard.core.feature.FeatureName;
import ru.finguard.core.feature.V3InputNormalizer;
import ru.finguard.core.model.RiskAssessment;
import ru.finguard.core.model.RiskDecision;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.neural.NeuralPredictor;
import ru.finguard.core.neural.NeuralReviewThreshold;
import ru.finguard.core.neural.V3NeuralRiskEvaluator;
import ru.finguard.core.rule.RuleResult;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DefaultRiskEngineNeuralTest {

    @TempDir
    Path directory;

    @Test
    void shouldSendNeuralRiskToReview() throws IOException {
        RiskEngine engine = createEngine(input -> 2.0f);

        RiskAssessment result = engine.evaluate(transfer());

        assertEquals(RiskDecision.REVIEW, result.decision());
        assertEquals(0, result.riskScore());
        assertTrue(result.factors().contains("NEURAL_REVIEW"));
    }

    @Test
    void shouldApproveWhenNeuralRiskIsLow() throws IOException {
        RiskEngine engine = createEngine(input -> -2.0f);

        RiskAssessment result = engine.evaluate(transfer());

        assertEquals(RiskDecision.APPROVED, result.decision());
        assertEquals(0, result.riskScore());
    }

    @Test
    void shouldReviewWhenNeuralInferenceFails() throws IOException {
        RiskEngine engine = createEngine(input -> {
            throw new IllegalStateException("Neural model unavailable");
        });

        RiskAssessment result = engine.evaluate(transfer());

        assertEquals(RiskDecision.REVIEW, result.decision());
        assertTrue(result.factors().contains("NEURAL_UNAVAILABLE"));
    }

    @Test
    void shouldApplyUpdatedThresholdWithoutRebuildingEngine()
            throws IOException {
        NeuralReviewThreshold threshold = new NeuralReviewThreshold(3.0f);

        RiskEngine engine = createEngine(
                input -> 2.0f,
                threshold
        );

        RiskAssessment first = engine.evaluate(transfer());

        assertEquals(RiskDecision.APPROVED, first.decision());
        assertEquals(
                3.0f,
                first.neural().orElseThrow().reviewThresholdLogit()
        );

        threshold.update(1.5145587f);

        RiskAssessment second = engine.evaluate(transfer());

        assertEquals(RiskDecision.REVIEW, second.decision());
        assertEquals(
                1.5145587f,
                second.neural().orElseThrow().reviewThresholdLogit()
        );
        assertTrue(second.factors().contains("NEURAL_REVIEW"));

        assertEquals(
                3.0f,
                first.neural().orElseThrow().reviewThresholdLogit()
        );
    }

    private RiskEngine createEngine(
            NeuralPredictor predictor
    ) throws IOException {
        return createEngine(
                predictor,
                new NeuralReviewThreshold(1.5145587f)
        );
    }

    private RiskEngine createEngine(
            NeuralPredictor predictor,
            NeuralReviewThreshold threshold
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
                .neuralReview(evaluator, threshold)
                .build();
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

    private TransferData transfer() {
        return new TransferData(
                1L,
                10L,
                20L,
                new BigDecimal("1000.00"),
                "EUR",
                Instant.parse("2026-10-30T12:00:00Z")
        );
    }
}