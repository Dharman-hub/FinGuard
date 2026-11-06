package ru.finguard.core.rule;

import org.junit.jupiter.api.Test;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.rule.impl.HighVelocityRule;
import ru.finguard.core.rule.impl.NewRecipientRule;
import ru.finguard.core.rule.impl.UnusualActivityRule;
import ru.finguard.core.rule.impl.UnusualAmountRule;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class RuleEngineTest {

    private final Instant timestamp = Instant.parse("2026-10-24T10:00:00Z");

    @Test
    void shouldEvaluateRulesInOrder() {
        RuleEngine engine = new RuleEngine(List.of(
                (transfer, history) -> RuleResult.passed("FIRST"),
                (transfer, history) -> RuleResult.triggered("SECOND", "Risk detected"),
                (transfer, history) -> RuleResult.insufficientData("THIRD", "Missing data")
        ));

        List<RuleResult> results = engine.evaluate(createTransfer(), createHistory());

        assertEquals(3, results.size());
        assertEquals("FIRST", results.get(0).code());
        assertEquals("SECOND", results.get(1).code());
        assertEquals("THIRD", results.get(2).code());
        assertEquals(RuleStatus.PASSED, results.get(0).status());
        assertEquals(RuleStatus.TRIGGERED, results.get(1).status());
        assertEquals(RuleStatus.INSUFFICIENT_DATA, results.get(2).status());
    }

    @Test
    void shouldEvaluateAllExistingRules() {
        RuleEngine engine = new RuleEngine(List.of(
                new UnusualAmountRule(),
                new HighVelocityRule(),
                new NewRecipientRule(),
                new UnusualActivityRule()
        ));

        List<RuleResult> results = engine.evaluate(createTransfer(), createHistory());

        assertEquals(4, results.size());

        for (RuleResult result : results) {
            assertEquals(RuleStatus.INSUFFICIENT_DATA, result.status());
        }
    }

    @Test
    void shouldReturnEmptyResultsWhenNoRulesConfigured() {
        RuleEngine engine = new RuleEngine(List.of());

        List<RuleResult> results = engine.evaluate(createTransfer(), createHistory());

        assertEquals(0, results.size());
    }

    @Test
    void shouldProtectRulesFromModification() {
        List<RiskRule> rules = new ArrayList<>();
        rules.add((transfer, history) -> RuleResult.passed("TEST"));

        RuleEngine engine = new RuleEngine(rules);

        rules.clear();

        List<RuleResult> results = engine.evaluate(createTransfer(), createHistory());

        assertEquals(1, results.size());
    }

    @Test
    void shouldReturnImmutableResults() {
        RuleEngine engine = new RuleEngine(List.of(
                (transfer, history) -> RuleResult.passed("TEST")
        ));

        List<RuleResult> results = engine.evaluate(createTransfer(), createHistory());

        assertThrows(
                UnsupportedOperationException.class,
                () -> results.clear()
        );
    }

    @Test
    void shouldPropagateRuleException() {
        RuleEngine engine = new RuleEngine(List.of(
                (transfer, history) -> {
                    throw new IllegalStateException("Rule execution failed");
                }
        ));

        assertThrows(
                IllegalStateException.class,
                () -> engine.evaluate(createTransfer(), createHistory())
        );
    }

    @Test
    void shouldRejectNullRuleResult() {
        RuleEngine engine = new RuleEngine(List.of(
                (transfer, history) -> null
        ));

        assertThrows(
                NullPointerException.class,
                () -> engine.evaluate(createTransfer(), createHistory())
        );
    }

    @Test
    void shouldRejectNullArguments() {
        RuleEngine engine = new RuleEngine(List.of());

        assertThrows(
                NullPointerException.class,
                () -> new RuleEngine(null)
        );

        assertThrows(
                NullPointerException.class,
                () -> engine.evaluate(null, createHistory())
        );

        assertThrows(
                NullPointerException.class,
                () -> engine.evaluate(createTransfer(), null)
        );
    }

    private TransferData createTransfer() {
        return new TransferData(
                1042L,
                125L,
                458L,
                new BigDecimal("10000.00"),
                "RUB",
                timestamp
        );
    }

    private TransactionHistorySnapshot createHistory() {
        return TransactionHistorySnapshot.empty(timestamp);
    }
}
