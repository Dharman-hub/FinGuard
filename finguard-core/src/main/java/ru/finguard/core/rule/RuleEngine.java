package ru.finguard.core.rule;

import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class RuleEngine {

    private final List<RiskRule> rules;

    public RuleEngine(List<RiskRule> rules) {
        this.rules = List.copyOf(Objects.requireNonNull(rules));
    }

    public List<RuleResult> evaluate(TransferData transfer, TransactionHistorySnapshot history) {
        Objects.requireNonNull(transfer);
        Objects.requireNonNull(history);

        List<RuleResult> results = new ArrayList<>(rules.size());

        for (RiskRule rule : rules) {
            RuleResult result = Objects.requireNonNull(
                    rule.evaluate(transfer, history),
                    "Risk rule must return a result"
            );

            results.add(result);
        }

        return List.copyOf(results);
    }
}
