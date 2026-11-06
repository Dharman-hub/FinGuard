package ru.finguard.core.api;

import ru.finguard.core.decision.DecisionEngine;
import ru.finguard.core.engine.DefaultRiskEngine;
import ru.finguard.core.neural.NeuralReviewThreshold;
import ru.finguard.core.neural.V3NeuralRiskEvaluator;
import ru.finguard.core.provider.ConfirmedFraudProvider;
import ru.finguard.core.provider.SupplierIbanDataProvider;
import ru.finguard.core.provider.TransactionDataProvider;
import ru.finguard.core.rule.RiskRule;
import ru.finguard.core.rule.RuleEngine;
import ru.finguard.core.rule.impl.HighVelocityRule;
import ru.finguard.core.rule.impl.NewRecipientRule;
import ru.finguard.core.rule.impl.UnusualActivityRule;
import ru.finguard.core.rule.impl.UnusualAmountRule;
import ru.finguard.core.scoring.RiskScorer;
import ru.finguard.core.decision.RuleDecisionThresholds;
import ru.finguard.core.decision.RuntimeThresholdPolicy;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

public final class RiskEngineBuilder {

    private final TransactionDataProvider transactionDataProvider;
    private final Map<String, RiskRule> rules = new LinkedHashMap<>();
    private final Map<String, Integer> weights = new LinkedHashMap<>();

    private int reviewThreshold = 40;
    private int rejectThreshold = 80;

    private RuleDecisionThresholds dynamicDecisionThresholds;
    private RuntimeThresholdPolicy runtimePolicy;

    private V3NeuralRiskEvaluator neuralEvaluator;
    private NeuralReviewThreshold neuralReviewThreshold =
            new NeuralReviewThreshold(0.0f);

    private SupplierIbanDataProvider supplierIbanDataProvider;
    private ConfirmedFraudProvider confirmedFraudProvider;

    private String modelSha256;

    public RiskEngineBuilder(TransactionDataProvider transactionDataProvider) {
        this.transactionDataProvider = Objects.requireNonNull(
                transactionDataProvider
        );

        addRule("UNUSUAL_AMOUNT", new UnusualAmountRule(), 25);
        addRule("HIGH_VELOCITY", new HighVelocityRule(), 35);
        addRule("NEW_RECIPIENT", new NewRecipientRule(), 20);
        addRule("UNUSUAL_ACTIVITY", new UnusualActivityRule(), 20);
    }

    public RiskEngineBuilder addRule(
            String code,
            RiskRule rule,
            int weight
    ) {
        validateCode(code);
        Objects.requireNonNull(rule);
        validateWeight(weight);

        if (rules.containsKey(code)) {
            throw new IllegalArgumentException(
                    "Rule already exists: " + code
            );
        }

        rules.put(code, rule);
        weights.put(code, weight);

        return this;
    }

    public RiskEngineBuilder removeRule(String code) {
        validateCode(code);

        if (!rules.containsKey(code)) {
            throw new IllegalArgumentException(
                    "Unknown rule: " + code
            );
        }

        rules.remove(code);
        weights.remove(code);

        return this;
    }

    public RiskEngineBuilder weight(String code, int weight) {
        validateCode(code);
        validateWeight(weight);

        if (!rules.containsKey(code)) {
            throw new IllegalArgumentException(
                    "Unknown rule: " + code
            );
        }

        weights.put(code, weight);

        return this;
    }

    public RiskEngineBuilder reviewThreshold(int threshold) {
        reviewThreshold = threshold;
        return this;
    }

    public RiskEngineBuilder rejectThreshold(int threshold) {
        rejectThreshold = threshold;
        return this;
    }

    public RiskEngineBuilder decisionThresholds(
            RuleDecisionThresholds thresholds
    ) {
        this.dynamicDecisionThresholds = Objects.requireNonNull(
                thresholds,
                "Rule decision thresholds are required"
        );

        return this;
    }

    public RiskEngineBuilder neuralReview(
            V3NeuralRiskEvaluator evaluator,
            float thresholdLogit
    ) {
        return neuralReview(
                evaluator,
                new NeuralReviewThreshold(thresholdLogit)
        );
    }

    public RiskEngineBuilder neuralReview(
            V3NeuralRiskEvaluator evaluator,
            NeuralReviewThreshold threshold
    ) {
        neuralEvaluator = Objects.requireNonNull(
                evaluator,
                "Neural evaluator is required"
        );

        neuralReviewThreshold = Objects.requireNonNull(
                threshold,
                "Neural review threshold is required"
        );

        return this;
    }

    public RiskEngineBuilder supplierIbanDataProvider(
            SupplierIbanDataProvider provider
    ) {
        supplierIbanDataProvider = Objects.requireNonNull(
                provider,
                "Supplier IBAN provider is required"
        );

        return this;
    }

    public RiskEngineBuilder confirmedFraudProvider(
            ConfirmedFraudProvider provider
    ) {
        confirmedFraudProvider = Objects.requireNonNull(
                provider,
                "Confirmed fraud provider is required"
        );

        return this;
    }

    public RiskEngineBuilder modelSha256(String sha256) {
        if (sha256 == null || !sha256.matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException(
                    "Model SHA-256 must contain 64 lowercase hex characters"
            );
        }

        this.modelSha256 = sha256;
        return this;
    }

    public RiskEngineBuilder runtimeThresholdPolicy(
            RuntimeThresholdPolicy policy
    ) {
        this.runtimePolicy = Objects.requireNonNull(
                policy,
                "Runtime threshold policy is required"
        );

        return this;
    }

    public RiskEngine build() {
        if (rules.isEmpty()) {
            throw new IllegalStateException(
                    "At least one risk rule is required"
            );
        }

        if (supplierIbanDataProvider != null && neuralEvaluator == null) {
            throw new IllegalStateException(
                    "Supplier IBAN provider requires neural evaluation"
            );
        }

        RuleEngine ruleEngine = new RuleEngine(
                new ArrayList<>(rules.values())
        );

        RiskScorer riskScorer = new RiskScorer(weights);

        DecisionEngine decisionEngine = dynamicDecisionThresholds == null
                ? new DecisionEngine(reviewThreshold, rejectThreshold)
                : new DecisionEngine(dynamicDecisionThresholds);

        return new DefaultRiskEngine(
                transactionDataProvider,
                ruleEngine,
                riskScorer,
                decisionEngine,
                neuralEvaluator,
                neuralReviewThreshold,
                supplierIbanDataProvider,
                confirmedFraudProvider,
                modelSha256,
                runtimePolicy
        );
    }

    private static void validateCode(String code) {
        if (code == null || code.isBlank()) {
            throw new IllegalArgumentException(
                    "Rule code cannot be blank"
            );
        }
    }

    private static void validateWeight(int weight) {
        if (weight <= 0 || weight > 100) {
            throw new IllegalArgumentException(
                    "Rule weight must be between 1 and 100"
            );
        }
    }
}