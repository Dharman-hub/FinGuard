package ru.finguard.core.model;

import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RiskAssessmentDetailsTest {

    @Test
    void shouldKeepOldConstructorWorking() {
        RiskAssessment result = new RiskAssessment(
                RiskDecision.APPROVED,
                10,
                List.of()
        );

        assertEquals(RiskDecision.APPROVED, result.decision());
        assertEquals(10, result.riskScore());
        assertEquals(DecisionReason.UNSPECIFIED, result.reason());
        assertTrue(result.neural().isEmpty());
    }

    @Test
    void shouldExposeNeuralRiskDetails() {
        NeuralRiskDetails neural = new NeuralRiskDetails(
                2.5f,
                1.5145587f
        );

        RiskAssessment result = new RiskAssessment(
                RiskDecision.REVIEW,
                20,
                List.of("NEURAL_REVIEW"),
                Optional.of(neural),
                DecisionReason.NEURAL_REVIEW
        );

        assertEquals(DecisionReason.NEURAL_REVIEW, result.reason());
        assertEquals(2.5f, result.neural().orElseThrow().logit());
        assertTrue(result.neural().orElseThrow().requiresReview());
    }

    @Test
    void shouldRejectInvalidNeuralValues() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new NeuralRiskDetails(Float.NaN, 1.5f)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> new NeuralRiskDetails(2.0f, Float.POSITIVE_INFINITY)
        );
    }

    @Test
    void shouldKeepFactorsImmutable() {
        List<String> factors = new ArrayList<>();
        factors.add("NEURAL_REVIEW");

        RiskAssessment result = new RiskAssessment(
                RiskDecision.REVIEW,
                50,
                factors,
                Optional.empty(),
                DecisionReason.NEURAL_REVIEW
        );

        factors.clear();

        assertEquals(1, result.factors().size());
        assertFalse(result.factors().isEmpty());

        assertThrows(
                UnsupportedOperationException.class,
                () -> result.factors().add("NEW_FACTOR")
        );
    }
}