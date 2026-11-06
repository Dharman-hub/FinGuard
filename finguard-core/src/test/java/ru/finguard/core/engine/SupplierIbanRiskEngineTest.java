package ru.finguard.core.engine;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import ru.finguard.core.api.RiskEngine;
import ru.finguard.core.api.RiskEngineBuilder;
import ru.finguard.core.feature.FeatureName;
import ru.finguard.core.feature.V3InputNormalizer;
import ru.finguard.core.model.RiskDecision;
import ru.finguard.core.model.SupplierIbanContext;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.neural.NeuralPredictor;
import ru.finguard.core.neural.V3NeuralRiskEvaluator;
import ru.finguard.core.provider.SupplierIbanDataProvider;
import ru.finguard.core.rule.RuleResult;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SupplierIbanRiskEngineTest {

    @TempDir
    Path directory;

    @Test
    void shouldUseSupplierIbanHistory() throws IOException {
        AtomicReference<float[]> captured = new AtomicReference<>();

        NeuralPredictor predictor = input -> {
            captured.set(input.clone());
            return -2.0f;
        };

        RiskEngine engine = createEngine(
                predictor,
                transfer -> new SupplierIbanContext(
                        true,
                        "DE89370400440532013000",
                        true,
                        false
                )
        );

        var result = engine.evaluate(transfer());

        assertEquals(RiskDecision.APPROVED, result.decision());
        assertEquals(1.0f, captured.get()[34]);
        assertEquals(1.0f, captured.get()[69]);
    }

    @Test
    void shouldSupportMissingIbanProvider() throws IOException {
        AtomicReference<float[]> captured = new AtomicReference<>();

        RiskEngine engine = createEngine(
                input -> {
                    captured.set(input.clone());
                    return -2.0f;
                },
                null
        );

        var result = engine.evaluate(transfer());

        assertEquals(RiskDecision.APPROVED, result.decision());
        assertEquals(0.0f, captured.get()[34]);
        assertEquals(0.0f, captured.get()[69]);
    }

    @Test
    void shouldReviewWhenIbanProviderFails() throws IOException {
        RiskEngine engine = createEngine(
                input -> -2.0f,
                transfer -> {
                    throw new IllegalStateException("IBAN history unavailable");
                }
        );

        var result = engine.evaluate(transfer());

        assertEquals(RiskDecision.REVIEW, result.decision());
        assertTrue(
                result.factors().contains("SUPPLIER_IBAN_UNAVAILABLE")
        );
    }

    private RiskEngine createEngine(
            NeuralPredictor predictor,
            SupplierIbanDataProvider ibanProvider
    ) throws IOException {
        V3NeuralRiskEvaluator evaluator = new V3NeuralRiskEvaluator(
                createNormalizer(),
                predictor
        );

        RiskEngineBuilder builder = new RiskEngineBuilder(
                transfer -> TransactionHistorySnapshot.empty(
                        transfer.timestamp()
                )
        )
                .removeRule("UNUSUAL_AMOUNT")
                .removeRule("HIGH_VELOCITY")
                .removeRule("NEW_RECIPIENT")
                .removeRule("UNUSUAL_ACTIVITY")
                .addRule(
                        "ALWAYS_PASS",
                        (transfer, history) -> RuleResult.passed("ALWAYS_PASS"),
                        10
                )
                .neuralReview(evaluator, 1.5145587f);

        if (ibanProvider != null) {
            builder.supplierIbanDataProvider(ibanProvider);
        }

        return builder.build();
    }

    private V3InputNormalizer createNormalizer() throws IOException {
        StringBuilder content = new StringBuilder(
                "feature,mean,std,observed_train_count\n"
        );

        FeatureName[] names = FeatureName.values();

        for (int i = 0; i < 35; i++) {
            String name = i == 34
                    ? "NEW_SUPPLIER_IBAN_AFTER_HISTORY"
                    : names[i].name();

            content.append(name)
                    .append(",0,1,100\n");
        }

        Path path = directory.resolve("normalization.csv");

        Files.writeString(path, content.toString());

        return V3InputNormalizer.load(path);
    }

    private TransferData transfer() {
        return new TransferData(
                1L,
                10L,
                20L,
                new BigDecimal("1000.00"),
                "EUR",
                Instant.parse("2026-10-30T12:00:00Z")
        );
    }
}