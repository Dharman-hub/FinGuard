package ru.finguard.core.scoring;

import org.junit.jupiter.api.Test;
import ru.finguard.core.rule.RuleResult;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RiskScorerTest {

    private final RiskScorer scorer = new RiskScorer();

    @Test
    void shouldCalculateTriggeredRuleScores() {
        RuleScore score = scorer.calculate(List.of(
                RuleResult.triggered("UNUSUAL_AMOUNT", "Unusual amount"),
                RuleResult.triggered("HIGH_VELOCITY", "High frequency"),
                RuleResult.passed("NEW_RECIPIENT"),
                RuleResult.passed("UNUSUAL_ACTIVITY")
        ));

        assertEquals(60, score.confirmedScore());
        assertEquals(60, score.maximumScore());
        assertEquals(2, score.triggeredRules().size());
        assertFalse(score.hasMissingData());
    }

    @Test
    void shouldAccountForMissingData() {
        RuleScore score = scorer.calculate(List.of(
                RuleResult.triggered("UNUSUAL_AMOUNT", "Unusual amount"),
                RuleResult.insufficientData("HIGH_VELOCITY", "Missing data"),
                RuleResult.passed("NEW_RECIPIENT"),
                RuleResult.insufficientData("UNUSUAL_ACTIVITY", "Missing data")
        ));

        assertEquals(25, score.confirmedScore());
        assertEquals(80, score.maximumScore());
        assertEquals(2, score.missingDataRules().size());
        assertTrue(score.hasMissingData());
    }

    @Test
    void shouldHandleAllRulesPassing() {
        RuleScore score = scorer.calculate(List.of(
                RuleResult.passed("UNUSUAL_AMOUNT"),
                RuleResult.passed("HIGH_VELOCITY"),
                RuleResult.passed("NEW_RECIPIENT"),
                RuleResult.passed("UNUSUAL_ACTIVITY")
        ));

        assertEquals(0, score.confirmedScore());
        assertEquals(0, score.maximumScore());
        assertFalse(score.hasMissingData());
    }

    @Test
    void shouldHandleAllRulesMissingData() {
        RuleScore score = scorer.calculate(List.of(
                RuleResult.insufficientData("UNUSUAL_AMOUNT", "Missing data"),
                RuleResult.insufficientData("HIGH_VELOCITY", "Missing data"),
                RuleResult.insufficientData("NEW_RECIPIENT", "Missing data"),
                RuleResult.insufficientData("UNUSUAL_ACTIVITY", "Missing data")
        ));

        assertEquals(0, score.confirmedScore());
        assertEquals(100, score.maximumScore());
        assertTrue(score.hasMissingData());
    }

    @Test
    void shouldSupportCustomWeights() {
        RiskScorer customScorer = new RiskScorer(Map.of(
                "CUSTOM_RULE", 75
        ));

        RuleScore score = customScorer.calculate(List.of(
                RuleResult.triggered("CUSTOM_RULE", "Custom risk")
        ));

        assertEquals(75, score.confirmedScore());
        assertEquals(75, score.maximumScore());
    }

    @Test
    void shouldLimitScoreToOneHundred() {
        RiskScorer customScorer = new RiskScorer(Map.of(
                "FIRST", 70,
                "SECOND", 70
        ));

        RuleScore score = customScorer.calculate(List.of(
                RuleResult.triggered("FIRST", "Risk"),
                RuleResult.triggered("SECOND", "Risk")
        ));

        assertEquals(100, score.confirmedScore());
        assertEquals(100, score.maximumScore());
    }

    @Test
    void shouldRejectMissingRuleResults() {
        assertThrows(
                IllegalArgumentException.class,
                () -> scorer.calculate(List.of(
                        RuleResult.passed("UNUSUAL_AMOUNT")
                ))
        );
    }

    @Test
    void shouldRejectDuplicateRuleResults() {
        assertThrows(
                IllegalArgumentException.class,
                () -> scorer.calculate(List.of(
                        RuleResult.passed("UNUSUAL_AMOUNT"),
                        RuleResult.passed("UNUSUAL_AMOUNT"),
                        RuleResult.passed("NEW_RECIPIENT"),
                        RuleResult.passed("UNUSUAL_ACTIVITY")
                ))
        );
    }

    @Test
    void shouldRejectUnknownRules() {
        assertThrows(
                IllegalArgumentException.class,
                () -> scorer.calculate(List.of(
                        RuleResult.passed("UNUSUAL_AMOUNT"),
                        RuleResult.passed("HIGH_VELOCITY"),
                        RuleResult.passed("NEW_RECIPIENT"),
                        RuleResult.passed("UNKNOWN")
                ))
        );
    }

    @Test
    void shouldRejectInvalidWeights() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new RiskScorer(Map.of())
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> new RiskScorer(Map.of("TEST", 0))
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> new RiskScorer(Map.of("TEST", 101))
        );
    }

    @Test
    void shouldReturnImmutableRuleLists() {
        RuleScore score = scorer.calculate(List.of(
                RuleResult.triggered("UNUSUAL_AMOUNT", "Risk"),
                RuleResult.passed("HIGH_VELOCITY"),
                RuleResult.passed("NEW_RECIPIENT"),
                RuleResult.passed("UNUSUAL_ACTIVITY")
        ));

        assertThrows(
                UnsupportedOperationException.class,
                () -> score.triggeredRules().clear()
        );
    }
}
