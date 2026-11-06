package ru.finguard.core.engine;

import ru.finguard.core.api.RiskEngine;
import ru.finguard.core.decision.DecisionEngine;
import ru.finguard.core.decision.RuleDecisionThresholds;
import ru.finguard.core.decision.RuntimeThresholdPolicy;
import ru.finguard.core.model.DecisionReason;
import ru.finguard.core.model.NeuralRiskDetails;
import ru.finguard.core.model.RiskAssessment;
import ru.finguard.core.model.RiskDecision;
import ru.finguard.core.model.RiskEvaluationTrace;
import ru.finguard.core.model.SupplierIbanContext;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.neural.NeuralReviewThreshold;
import ru.finguard.core.neural.V3NeuralRiskEvaluator;
import ru.finguard.core.provider.ConfirmedFraudProvider;
import ru.finguard.core.provider.SupplierIbanDataProvider;
import ru.finguard.core.provider.TransactionDataProvider;
import ru.finguard.core.rule.RuleEngine;
import ru.finguard.core.rule.RuleResult;
import ru.finguard.core.scoring.RiskScorer;
import ru.finguard.core.scoring.RuleScore;
import ru.finguard.core.decision.NeuralDataQualityPolicy;
import ru.finguard.core.api.RiskCheckUnavailableException;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

public final class DefaultRiskEngine implements RiskEngine {

    private static final System.Logger LOGGER =
            System.getLogger(DefaultRiskEngine.class.getName());

    private static final NeuralDataQualityPolicy UNRESTRICTED_QUALITY =
            NeuralDataQualityPolicy.unrestricted();

    private final TransactionDataProvider transactionDataProvider;
    private final RuleEngine ruleEngine;
    private final RiskScorer riskScorer;
    private final DecisionEngine decisionEngine;
    private final V3NeuralRiskEvaluator neuralEvaluator;
    private final NeuralReviewThreshold neuralReviewThreshold;
    private final SupplierIbanDataProvider supplierIbanDataProvider;
    private final ConfirmedFraudProvider confirmedFraudProvider;
    private final String modelSha256;
    private final RuntimeThresholdPolicy runtimePolicy;

    public DefaultRiskEngine(
            TransactionDataProvider transactionDataProvider,
            RuleEngine ruleEngine,
            RiskScorer riskScorer,
            DecisionEngine decisionEngine,
            V3NeuralRiskEvaluator neuralEvaluator,
            NeuralReviewThreshold neuralReviewThreshold,
            SupplierIbanDataProvider supplierIbanDataProvider,
            ConfirmedFraudProvider confirmedFraudProvider,
            String modelSha256,
            RuntimeThresholdPolicy runtimePolicy
    ) {
        this.transactionDataProvider = Objects.requireNonNull(
                transactionDataProvider
        );
        this.ruleEngine = Objects.requireNonNull(ruleEngine);
        this.riskScorer = Objects.requireNonNull(riskScorer);
        this.decisionEngine = Objects.requireNonNull(decisionEngine);
        this.neuralReviewThreshold = Objects.requireNonNull(
                neuralReviewThreshold
        );

        if (supplierIbanDataProvider != null && neuralEvaluator == null) {
            throw new IllegalArgumentException(
                    "Supplier IBAN provider requires neural evaluation"
            );
        }

        if (modelSha256 != null
                && !modelSha256.matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException(
                    "Invalid model SHA-256"
            );
        }

        if (runtimePolicy != null) {
            boolean policyHasNeural = runtimePolicy.snapshot()
                    .neuralReviewThresholdLogit()
                    .isPresent();

            if (policyHasNeural != (neuralEvaluator != null)) {
                throw new IllegalArgumentException(
                        "Runtime policy does not match neural configuration"
                );
            }
        }

        this.neuralEvaluator = neuralEvaluator;
        this.supplierIbanDataProvider = supplierIbanDataProvider;
        this.confirmedFraudProvider = confirmedFraudProvider;
        this.modelSha256 = modelSha256;
        this.runtimePolicy = runtimePolicy;
    }

    @Override
    public RiskAssessment evaluate(TransferData transfer) {
        Objects.requireNonNull(transfer);

        RuntimeThresholdPolicy.Snapshot activePolicy =
                runtimePolicy == null
                        ? null
                        : runtimePolicy.snapshot();

        RuleDecisionThresholds.Snapshot appliedRules =
                activePolicy == null
                        ? decisionEngine.thresholdSnapshot()
                        : new RuleDecisionThresholds.Snapshot(
                        activePolicy.reviewThreshold(),
                        activePolicy.rejectThreshold()
                );

        float appliedThreshold = activePolicy == null
                ? neuralReviewThreshold.get()
                : activePolicy.neuralReviewThresholdLogit().orElse(0.0f);

        Float appliedRejectThreshold = activePolicy == null
                ? null
                : activePolicy.neuralRejectThresholdLogit().orElse(null);

        NeuralDataQualityPolicy qualityPolicy = activePolicy == null
                ? UNRESTRICTED_QUALITY
                : activePolicy.neuralDataQualityPolicy();

        TransactionHistorySnapshot history;

        try {
            history = Objects.requireNonNull(
                    transactionDataProvider.loadSnapshot(transfer),
                    "TransactionDataProvider returned null"
            );
        } catch (RuntimeException exception) {
            LOGGER.log(
                    System.Logger.Level.ERROR,
                    "Transaction history unavailable; fraud assessment aborted",
                    exception
            );

            throw new RiskCheckUnavailableException(
                    "Fraud assessment cannot continue without transaction history",
                    exception
            );
        }

        List<RuleResult> results = ruleEngine.evaluate(transfer, history);
        RuleScore score = riskScorer.calculate(results);

        List<String> factors = new ArrayList<>(
                score.triggeredRules()
        );

        for (String rule : score.missingDataRules()) {
            factors.add("INSUFFICIENT_DATA:" + rule);
        }

        boolean confirmedFraud = false;
        boolean confirmationFailed = false;

        if (confirmedFraudProvider != null) {
            try {
                confirmedFraud = confirmedFraudProvider
                        .isConfirmedFraud(transfer);

                if (confirmedFraud) {
                    factors.add("BANK_CONFIRMED_FRAUD");
                }
            } catch (RuntimeException exception) {
                LOGGER.log(
                        System.Logger.Level.ERROR,
                        "Fraud confirmation unavailable; transfer requires review",
                        exception
                );

                confirmationFailed = true;
                factors.add("FRAUD_CONFIRMATION_UNAVAILABLE");
            }
        }

        Float neuralLogit = null;
        NeuralDataQualityPolicy.Coverage coverage = null;
        boolean neuralFailed = false;
        boolean ibanProviderFailed = false;

        if (neuralEvaluator != null) {
            SupplierIbanContext ibanContext = SupplierIbanContext.unknown();

            if (supplierIbanDataProvider != null) {
                try {
                    ibanContext = Objects.requireNonNull(
                            supplierIbanDataProvider.loadContext(transfer),
                            "Supplier IBAN provider returned null"
                    );
                } catch (RuntimeException exception) {
                    LOGGER.log(
                            System.Logger.Level.ERROR,
                            "Supplier IBAN data unavailable; transfer requires review",
                            exception
                    );

                    ibanProviderFailed = true;
                    factors.add("SUPPLIER_IBAN_UNAVAILABLE");
                }
            }

            if (!ibanProviderFailed) {
                try {
                    V3NeuralRiskEvaluator.Prediction prediction =
                            neuralEvaluator.predictWithCoverage(
                                    transfer,
                                    history,
                                    ibanContext.toSignal(),
                                    qualityPolicy
                            );

                    neuralLogit = prediction.logit();
                    coverage = prediction.coverage();
                } catch (RuntimeException exception) {
                    LOGGER.log(
                            System.Logger.Level.ERROR,
                            "Neural inference failed; transfer requires review",
                            exception
                    );

                    neuralFailed = true;
                    factors.add("NEURAL_UNAVAILABLE");
                }
            }
        }

        RiskDecision rulesOnlyDecision = decisionEngine.decide(
                score,
                confirmedFraud,
                appliedRules
        );

        RiskDecision decision = rulesOnlyDecision;
        boolean blockedNeuralDecision = false;
        boolean allowedNeuralReject = false;

        if (neuralEvaluator != null) {
            decision = decisionEngine.decideWithNeural(
                    score,
                    neuralLogit,
                    appliedThreshold,
                    appliedRejectThreshold,
                    confirmedFraud,
                    appliedRules
            );

            if (neuralLogit != null && coverage != null) {
                boolean neuralReject =
                        decision == RiskDecision.REJECTED
                                && rulesOnlyDecision != RiskDecision.REJECTED;

                boolean neuralApprove =
                        decision == RiskDecision.APPROVED;

                boolean insufficientReject =
                        neuralReject && !coverage.neuralRejectionAllowed();

                boolean insufficientApprove =
                        neuralApprove && !coverage.neuralApprovalAllowed();

                blockedNeuralDecision =
                        insufficientReject || insufficientApprove;

                if (blockedNeuralDecision) {
                    factors.add("NEURAL_DATA_INSUFFICIENT");

                    decision = qualityPolicy.insufficientDataAction()
                            == NeuralDataQualityPolicy.MissingDataAction.REVIEW
                            ? RiskDecision.REVIEW
                            : rulesOnlyDecision;
                }

                allowedNeuralReject =
                        neuralReject && !insufficientReject;

                if (allowedNeuralReject) {
                    factors.add("NEURAL_REJECT");
                } else if (decision == RiskDecision.REVIEW
                        && neuralLogit > appliedThreshold) {
                    factors.add("NEURAL_REVIEW");
                }
            }
        }

        boolean dependencyFailed = confirmationFailed
                || ibanProviderFailed
                || neuralFailed;

        if (dependencyFailed && decision != RiskDecision.REJECTED) {
            decision = RiskDecision.REVIEW;
        }

        DecisionReason reason;

        if (decision == RiskDecision.REJECTED && confirmedFraud) {
            reason = DecisionReason.BANK_CONFIRMED_FRAUD;
        } else if (decision == RiskDecision.REJECTED
                && allowedNeuralReject) {
            reason = DecisionReason.NEURAL_REJECT;
        } else if (dependencyFailed) {
            reason = DecisionReason.DEPENDENCY_UNAVAILABLE;
        } else if (confirmedFraud) {
            reason = DecisionReason.BANK_CONFIRMED_FRAUD;
        } else if (blockedNeuralDecision
                && decision == RiskDecision.REVIEW) {
            reason = DecisionReason.INSUFFICIENT_NEURAL_DATA;
        } else if (decision == RiskDecision.REVIEW
                && neuralLogit != null
                && neuralLogit > appliedThreshold) {
            reason = DecisionReason.NEURAL_REVIEW;
        } else if (decision == RiskDecision.REVIEW) {
            reason = DecisionReason.RULE_REVIEW;
        } else {
            reason = DecisionReason.LOW_RISK;
        }

        Optional<NeuralRiskDetails> neuralDetails = neuralLogit == null
                ? Optional.empty()
                : Optional.of(new NeuralRiskDetails(
                neuralLogit,
                appliedThreshold
        ));

        RiskEvaluationTrace trace = RiskEvaluationTrace.create(
                neuralEvaluator == null ? null : modelSha256,
                appliedRules.reviewThreshold(),
                appliedRules.rejectThreshold(),
                neuralEvaluator == null ? null : appliedThreshold,
                activePolicy == null ? 0L : activePolicy.version(),
                neuralEvaluator == null ? null : appliedRejectThreshold
        );

        return new RiskAssessment(
                decision,
                score.confirmedScore(),
                factors,
                neuralDetails,
                reason,
                Optional.of(trace)
        );
    }
}