package ru.finguard.core.decision;

import org.junit.jupiter.api.Test;
import ru.finguard.core.model.RiskDecision;
import ru.finguard.core.scoring.RuleScore;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class RuleDecisionThresholdsTest {

    @Test
    void shouldUseInitialThresholds() {
        RuleDecisionThresholds thresholds =
                new RuleDecisionThresholds(40, 80);

        assertEquals(
                new RuleDecisionThresholds.Snapshot(40, 80),
                thresholds.snapshot()
        );
    }

    @Test
    void shouldUpdateBothThresholdsTogether() {
        RuleDecisionThresholds thresholds =
                new RuleDecisionThresholds(40, 80);

        RuleDecisionThresholds.Snapshot previous =
                thresholds.snapshot();

        thresholds.update(20, 95);

        assertEquals(
                new RuleDecisionThresholds.Snapshot(20, 95),
                thresholds.snapshot()
        );

        assertEquals(
                new RuleDecisionThresholds.Snapshot(40, 80),
                previous
        );
    }

    @Test
    void shouldRejectInvalidThresholdsWithoutChangingCurrent() {
        RuleDecisionThresholds thresholds =
                new RuleDecisionThresholds(40, 80);

        assertThrows(
                IllegalArgumentException.class,
                () -> thresholds.update(90, 80)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> thresholds.update(0, 80)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> thresholds.update(40, 101)
        );

        assertEquals(
                new RuleDecisionThresholds.Snapshot(40, 80),
                thresholds.snapshot()
        );
    }

    @Test
    void shouldApplyChangesToExistingDecisionEngine() {
        RuleDecisionThresholds thresholds =
                new RuleDecisionThresholds(40, 80);

        DecisionEngine engine = new DecisionEngine(thresholds);

        RuleScore lowRisk = new RuleScore(
                25,
                25,
                List.of(),
                List.of()
        );

        RuleScore highRisk = new RuleScore(
                90,
                90,
                List.of("TEST_RULE"),
                List.of()
        );

        assertEquals(
                RiskDecision.APPROVED,
                engine.decide(lowRisk)
        );

        thresholds.update(20, 80);

        assertEquals(
                RiskDecision.REVIEW,
                engine.decide(lowRisk)
        );

        assertEquals(
                RiskDecision.REJECTED,
                engine.decide(highRisk, true)
        );

        thresholds.update(20, 95);

        assertEquals(
                RiskDecision.REVIEW,
                engine.decide(highRisk, true)
        );

        assertEquals(
                RiskDecision.REVIEW,
                engine.decide(highRisk, false)
        );
    }
}