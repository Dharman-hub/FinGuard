package ru.finguard.core.api;

import org.junit.jupiter.api.Test;
import ru.finguard.core.model.ActivityStatistics;
import ru.finguard.core.model.AmountStatistics;
import ru.finguard.core.model.RecipientStatistics;
import ru.finguard.core.model.RiskAssessment;
import ru.finguard.core.model.RiskDecision;
import ru.finguard.core.model.RiskHistoryStatistics;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.model.VelocityStatistics;
import ru.finguard.core.provider.TransactionDataProvider;
import ru.finguard.core.rule.RuleResult;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RiskEngineBuilderTest {

    private final Instant timestamp = Instant.parse("2026-10-24T10:00:00Z");

    @Test
    void shouldBuildEngineWithDefaultRules() {
        RiskEngine engine = new RiskEngineBuilder(createProvider()).build();

        RiskAssessment result = engine.evaluate(createTransfer("10000"));

        assertEquals(RiskDecision.APPROVED, result.decision());
        assertEquals(0, result.riskScore());
    }

    @Test
    void shouldApplyCustomRuleWeight() {
        RiskEngine engine = new RiskEngineBuilder(createProvider())
                .weight("UNUSUAL_AMOUNT", 50)
                .build();

        RiskAssessment result = engine.evaluate(createTransfer("30000"));

        assertEquals(RiskDecision.REVIEW, result.decision());
        assertEquals(50, result.riskScore());
    }

    @Test
    void shouldApplyCustomThresholds() {
        RiskEngine engine = new RiskEngineBuilder(createProvider())
                .reviewThreshold(20)
                .rejectThreshold(30)
                .build();

        RiskAssessment result = engine.evaluate(createTransfer("30000"));

        assertEquals(RiskDecision.REVIEW, result.decision());
        assertEquals(25, result.riskScore());
    }

    @Test
    void shouldAllowAddingCustomRules() {
        RiskEngine engine = new RiskEngineBuilder(createProvider())
                .addRule(
                        "CUSTOM_RULE",
                        (transfer, history) -> RuleResult.triggered(
                                "CUSTOM_RULE",
                                "Custom risk detected"
                        ),
                        90
                )
                .build();

        RiskAssessment result = engine.evaluate(createTransfer("10000"));

        assertEquals(RiskDecision.REVIEW, result.decision());
        assertEquals(90, result.riskScore());
        assertTrue(result.factors().contains("CUSTOM_RULE"));
    }

    @Test
    void shouldAllowRemovingRules() {
        RiskEngine engine = new RiskEngineBuilder(createProvider())
                .removeRule("UNUSUAL_AMOUNT")
                .build();

        RiskAssessment result = engine.evaluate(createTransfer("30000"));

        assertEquals(RiskDecision.APPROVED, result.decision());
        assertEquals(0, result.riskScore());
    }

    @Test
    void shouldRejectDuplicateRules() {
        RiskEngineBuilder builder = new RiskEngineBuilder(createProvider());

        assertThrows(
                IllegalArgumentException.class,
                () -> builder.addRule(
                        "UNUSUAL_AMOUNT",
                        (transfer, history) -> RuleResult.passed("UNUSUAL_AMOUNT"),
                        10
                )
        );
    }

    @Test
    void shouldRejectUnknownRules() {
        RiskEngineBuilder builder = new RiskEngineBuilder(createProvider());

        assertThrows(
                IllegalArgumentException.class,
                () -> builder.weight("UNKNOWN", 20)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> builder.removeRule("UNKNOWN")
        );
    }

    @Test
    void shouldRejectInvalidConfiguration() {
        RiskEngineBuilder builder = new RiskEngineBuilder(createProvider());

        assertThrows(
                IllegalArgumentException.class,
                () -> builder.weight("UNUSUAL_AMOUNT", 0)
        );

        assertThrows(
                IllegalArgumentException.class,
                () -> builder.reviewThreshold(90)
                        .rejectThreshold(80)
                        .build()
        );
    }

    @Test
    void shouldNotModifyPreviouslyBuiltEngine() {
        RiskEngineBuilder builder = new RiskEngineBuilder(createProvider());

        RiskEngine originalEngine = builder.build();

        builder.weight("UNUSUAL_AMOUNT", 90);

        RiskEngine modifiedEngine = builder.build();

        assertEquals(
                25,
                originalEngine.evaluate(createTransfer("30000")).riskScore()
        );

        assertEquals(
                90,
                modifiedEngine.evaluate(createTransfer("30000")).riskScore()
        );
    }

    private TransactionDataProvider createProvider() {
        return transfer -> createHistory();
    }

    private TransferData createTransfer(String amount) {
        return new TransferData(
                1042L,
                125L,
                458L,
                new BigDecimal(amount),
                "RUB",
                timestamp
        );
    }

    private TransactionHistorySnapshot createHistory() {
        AmountStatistics amount = new AmountStatistics(
                Optional.of(new BigDecimal("5000")),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty()
        );

        VelocityStatistics velocity = new VelocityStatistics(
                OptionalLong.of(1),
                OptionalLong.of(2),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.of(120),
                Optional.empty(),
                OptionalLong.empty(),
                OptionalLong.of(0),
                OptionalLong.empty(),
                OptionalLong.empty()
        );

        RecipientStatistics recipient = new RecipientStatistics(
                OptionalLong.of(2),
                Optional.empty(),
                OptionalLong.empty(),
                OptionalDouble.empty(),
                OptionalLong.empty()
        );

        ActivityStatistics activity = new ActivityStatistics(
                OptionalLong.empty(),
                OptionalLong.of(30),
                OptionalDouble.empty(),
                OptionalDouble.empty(),
                OptionalDouble.of(0.2),
                OptionalDouble.empty(),
                OptionalLong.empty()
        );

        return new TransactionHistorySnapshot(
                amount,
                velocity,
                recipient,
                activity,
                RiskHistoryStatistics.empty(),
                timestamp
        );
    }
}