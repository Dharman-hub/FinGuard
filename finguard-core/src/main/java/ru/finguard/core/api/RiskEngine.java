package ru.finguard.core.api;

import ru.finguard.core.model.RiskAssessment;
import ru.finguard.core.model.TransferData;

public interface RiskEngine {
    RiskAssessment evaluate(TransferData transfer);
}