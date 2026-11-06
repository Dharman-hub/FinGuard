package ru.finguard.core.rule;

import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;

public interface RiskRule {
    RuleResult evaluate(TransferData transfer, TransactionHistorySnapshot history);
}
