package ru.finguard.core.neural;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import ru.finguard.core.feature.FeatureName;
import ru.finguard.core.feature.NewSupplierIbanDetector;
import ru.finguard.core.feature.V3InputNormalizer;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.model.ActivityStatistics;
import ru.finguard.core.model.AmountStatistics;
import ru.finguard.core.model.RecipientStatistics;
import ru.finguard.core.model.RiskHistoryStatistics;
import ru.finguard.core.model.VelocityStatistics;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.OptionalLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class V3NeuralRiskEvaluatorTest {

    @TempDir
    Path directory;

    @Test
    void shouldPrepareInputsAndPredictLogit() throws IOException {
        NeuralPredictor predictor = input -> {
            assertEquals(70, input.length);
            assertEquals(2.0f, input[0]);
            assertEquals(1.0f, input[34]);
            assertEquals(1.0f, input[35]);
            assertEquals(1.0f, input[69]);
            assertEquals(0.0f, input[36]);

            return 2.5f;
        };

        V3NeuralRiskEvaluator evaluator =
                new V3NeuralRiskEvaluator(
                        createNormalizer(),
                        predictor
                );

        float result = evaluator.predictLogit(
                transfer(),
                history(),
                new NewSupplierIbanDetector.Result(1.0f, 1.0f)
        );

        assertEquals(2.5f, result);
    }

    @Test
    void shouldSupportMissingIbanHistory() throws IOException {
        NeuralPredictor predictor = input -> {
            assertEquals(0.0f, input[34]);
            assertEquals(0.0f, input[69]);

            return -0.5f;
        };

        V3NeuralRiskEvaluator evaluator =
                new V3NeuralRiskEvaluator(
                        createNormalizer(),
                        predictor
                );

        float result = evaluator.predictLogit(
                transfer(),
                history()
        );

        assertEquals(-0.5f, result);
    }

    @Test
    void shouldRejectInvalidNeuralOutput() throws IOException {
        NeuralPredictor predictor = input -> Float.NaN;

        V3NeuralRiskEvaluator evaluator =
                new V3NeuralRiskEvaluator(
                        createNormalizer(),
                        predictor
                );

        assertThrows(
                IllegalStateException.class,
                () -> evaluator.predictLogit(
                        transfer(),
                        history()
                )
        );
    }

    @Test
    void shouldPreserveMissingAndKnownZeroAtNeuralInput()
            throws IOException {
        List<float[]> inputs = new ArrayList<>();

        NeuralPredictor predictor = input -> {
            inputs.add(input.clone());
            return 0.0f;
        };

        V3NeuralRiskEvaluator evaluator =
                new V3NeuralRiskEvaluator(
                        createNormalizer(),
                        predictor
                );

        VelocityStatistics velocity = new VelocityStatistics(
                OptionalLong.of(0),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                Optional.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.empty()
        );

        TransactionHistorySnapshot partialHistory =
                new TransactionHistorySnapshot(
                        AmountStatistics.empty(),
                        velocity,
                        RecipientStatistics.empty(),
                        ActivityStatistics.empty(),
                        RiskHistoryStatistics.empty(),
                        transfer().timestamp()
                );

        evaluator.predictLogit(transfer(), history());
        evaluator.predictLogit(transfer(), partialHistory);

        assertEquals(2, inputs.size());

        int index = FeatureName.TRANSFERS_LAST_5_MINUTES.ordinal();
        int maskIndex = 35 + index;

        float[] missing = inputs.get(0);
        float[] knownZero = inputs.get(1);

        assertEquals(70, missing.length);
        assertEquals(70, knownZero.length);

        assertEquals(0.0f, missing[index]);
        assertEquals(0.0f, knownZero[index]);

        assertEquals(0.0f, missing[maskIndex]);
        assertEquals(1.0f, knownZero[maskIndex]);

        assertEquals(1.0f, missing[35]);
        assertEquals(1.0f, knownZero[35]);

        assertEquals(0.0f, missing[69]);
        assertEquals(0.0f, knownZero[69]);
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

            double mean = i == 0 ? 10.0 : 0.0;
            double std = i == 0 ? 2.0 : 1.0;

            content.append(name)
                    .append(",")
                    .append(mean)
                    .append(",")
                    .append(std)
                    .append(",100\n");
        }

        Path file = directory.resolve("normalization.csv");

        Files.writeString(file, content.toString());

        return V3InputNormalizer.load(file);
    }

    private TransferData transfer() {
        return new TransferData(
                1L,
                10L,
                20L,
                new BigDecimal("14.00"),
                "EUR",
                Instant.parse("2026-10-01T12:00:00Z")
        );
    }

    private TransactionHistorySnapshot history() {
        return TransactionHistorySnapshot.empty(
                Instant.parse("2026-10-01T11:59:00Z")
        );
    }
}