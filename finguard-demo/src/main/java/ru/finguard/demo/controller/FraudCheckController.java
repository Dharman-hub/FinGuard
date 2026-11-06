package ru.finguard.demo.controller;

import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;
import ru.finguard.core.api.RiskEngine;
import ru.finguard.core.model.RiskAssessment;
import ru.finguard.core.model.TransferData;

@RestController
@RequestMapping("/demo/risk-checks")
public class FraudCheckController {

    private final RiskEngine riskEngine;

    public FraudCheckController(RiskEngine riskEngine) {
        this.riskEngine = riskEngine;
    }

    @PostMapping
    public ResponseEntity<RiskAssessment> check(@RequestBody TransferData transfer) {
        RiskAssessment assessment = riskEngine.evaluate(transfer);

        return ResponseEntity.ok(assessment);
    }
}