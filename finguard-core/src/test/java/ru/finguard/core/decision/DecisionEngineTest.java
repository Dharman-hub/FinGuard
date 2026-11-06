package ru.finguard.core.decision;

import org.junit.jupiter.api.Test;
import ru.finguard.core.model.RiskDecision;
import ru.finguard.core.scoring.RuleScore;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class DecisionEngineTest {

    private final DecisionEngine engine = new DecisionEngine();

    @Test
    void shouldApproveWhenRiskIsZero() {
        assertEquals(
                RiskDecision.APPROVED,
                engine.decide(createScore(0, 0))
        );
    }

    @Test
    void shouldApproveLowRiskTransfer() {
        assertEquals(
                RiskDecision.APPROVED,
                engine.decide(createScore(25, 25))
        );
    }

    @Test
    void shouldReviewAtReviewThreshold() {
        assertEquals(
                RiskDecision.REVIEW,
                engine.decide(createScore(40, 40))
        );
    }

    @Test
    void shouldReviewWhenHistoryIsMissing() {
        assertEquals(
                RiskDecision.REVIEW,
                engine.decide(createScore(25, 80))
        );
    }

    @Test
    void shouldReviewHighRiskWithoutBankConfirmation() {
        assertEquals(
                RiskDecision.REVIEW,
                engine.decide(createScore(100, 100))
        );
    }

    @Test
    void shouldRejectConfirmedHighRiskFraud() {
        assertEquals(
                RiskDecision.REJECTED,
                engine.decide(createScore(90, 90), true)
        );
    }

    @Test
    void shouldReviewConfirmedFraudWithInsufficientRiskScore() {
        assertEquals(
                RiskDecision.REVIEW,
                engine.decide(createScore(30, 30), true)
        );
    }

    @Test
    void shouldReviewHighNeuralRisk() {
        assertEquals(
                RiskDecision.REVIEW,
                engine.decideWithNeural(
                        createScore(0, 0),
                        2.0f,
                        1.5145587f
                )
        );
    }

    @Test
    void shouldNotRejectBasedOnNeuralRiskAlone() {
        assertEquals(
                RiskDecision.REVIEW,
                engine.decideWithNeural(
                        createScore(100, 100),
                        10.0f,
                        1.5145587f
                )
        );
    }

    @Test
    void shouldReviewWhenNeuralIsUnavailable() {
        assertEquals(
                RiskDecision.REVIEW,
                engine.decideWithNeural(
                        createScore(0, 0),
                        null,
                        1.5145587f
                )
        );
    }

    @Test
    void shouldRejectConfirmedFraudWithHighRiskEvenWithoutNeural() {
        assertEquals(
                RiskDecision.REJECTED,
                engine.decideWithNeural(
                        createScore(90, 90),
                        null,
                        1.5145587f,
                        true
                )
        );
    }

    @Test
    void shouldSupportCustomThresholds() {
        DecisionEngine custom = new DecisionEngine(60, 90);

        assertEquals(
                RiskDecision.APPROVED,
                custom.decide(createScore(59, 59))
        );

        assertEquals(
                RiskDecision.REVIEW,
                custom.decide(createScore(60, 60))
        );

        assertEquals(
                RiskDecision.REVIEW,
                custom.decide(createScore(90, 90))
        );

        assertEquals(
                RiskDecision.REJECTED,
                custom.decide(createScore(90, 90), true)
        );
    }

    @Test
    void shouldRejectInvalidThresholds() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new DecisionEngine(0, 80)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> new DecisionEngine(80, 80)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> new DecisionEngine(90, 80)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> new DecisionEngine(40, 101)
        );
    }

    @Test
    void shouldRejectNullScore() {
        assertThrows(
                NullPointerException.class,
                () -> engine.decide(null)
        );
    }

    private RuleScore createScore(int confirmed, int maximum) {
        List<String> missingRules = maximum > confirmed
                ? List.of("MISSING_RULE")
                : List.of();

        return new RuleScore(
                confirmed,
                maximum,
                List.of(),
                missingRules
        );
    }

    @Test
    void shouldAutomaticallyRejectHighNeuralRisk() {
        assertEquals(
                RiskDecision.REJECTED,
                engine.decideWithNeural(
                        createScore(0, 0),
                        2.0f,
                        0.5f,
                        1.5145587f,
                        false,
                        engine.thresholdSnapshot()
                )
        );
    }

    @Test
    void shouldReviewIntermediateNeuralRisk() {
        assertEquals(
                RiskDecision.REVIEW,
                engine.decideWithNeural(
                        createScore(0, 0),
                        1.0f,
                        0.5f,
                        1.5145587f,
                        false,
                        engine.thresholdSnapshot()
                )
        );
    }

    @Test
    void shouldApproveLowNeuralRisk() {
        assertEquals(
                RiskDecision.APPROVED,
                engine.decideWithNeural(
                        createScore(0, 0),
                        0.1f,
                        0.5f,
                        1.5145587f,
                        false,
                        engine.thresholdSnapshot()
                )
        );
    }

    @Test
    void shouldAllowDisablingAutomaticRejection() {
        assertEquals(
                RiskDecision.REVIEW,
                engine.decideWithNeural(
                        createScore(0, 0),
                        5.0f,
                        0.5f,
                        null,
                        false,
                        engine.thresholdSnapshot()
                )
        );
    }

    @Test
    void shouldRejectInvalidNeuralThresholds() {
        assertThrows(
                IllegalArgumentException.class,
                () -> engine.decideWithNeural(
                        createScore(0, 0),
                        2.0f,
                        2.0f,
                        1.0f,
                        false,
                        engine.thresholdSnapshot()
                )
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> engine.decideWithNeural(
                        createScore(0, 0),
                        2.0f,
                        0.5f,
                        Float.NaN,
                        false,
                        engine.thresholdSnapshot()
                )
        );
    }
}