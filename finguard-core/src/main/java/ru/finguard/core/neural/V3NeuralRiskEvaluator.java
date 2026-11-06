package ru.finguard.core.neural;

import ru.finguard.core.decision.NeuralDataQualityPolicy;
import ru.finguard.core.feature.FeatureExtractor;
import ru.finguard.core.feature.NewSupplierIbanDetector;
import ru.finguard.core.feature.RecipientAccountInputEncoder;
import ru.finguard.core.feature.V3InputNormalizer;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;

import java.util.Objects;

public final class V3NeuralRiskEvaluator {

    private static final NewSupplierIbanDetector.Result UNKNOWN_IBAN =
            new NewSupplierIbanDetector.Result(0.0f, 0.0f);

    private final FeatureExtractor featureExtractor;
    private final V3InputNormalizer normalizer;
    private final NeuralPredictor predictor;

    public V3NeuralRiskEvaluator(
            V3InputNormalizer normalizer,
            NeuralPredictor predictor
    ) {
        this.featureExtractor = new FeatureExtractor();
        this.normalizer = Objects.requireNonNull(normalizer);
        this.predictor = Objects.requireNonNull(predictor);
    }

    public float predictLogit(
            TransferData transfer,
            TransactionHistorySnapshot history
    ) {
        return predictLogit(transfer, history, UNKNOWN_IBAN);
    }

    public float predictLogit(
            TransferData transfer,
            TransactionHistorySnapshot history,
            NewSupplierIbanDetector.Result ibanSignal
    ) {
        return infer(rawInput(transfer, history, ibanSignal));
    }

    public Prediction predictWithCoverage(
            TransferData transfer,
            TransactionHistorySnapshot history,
            NewSupplierIbanDetector.Result ibanSignal,
            NeuralDataQualityPolicy qualityPolicy
    ) {
        Objects.requireNonNull(qualityPolicy);

        float[] raw = rawInput(transfer, history, ibanSignal);

        NeuralDataQualityPolicy.Coverage coverage =
                qualityPolicy.inspect(raw);

        return new Prediction(infer(raw), coverage);
    }

    private float[] rawInput(
            TransferData transfer,
            TransactionHistorySnapshot history,
            NewSupplierIbanDetector.Result ibanSignal
    ) {
        Objects.requireNonNull(transfer);
        Objects.requireNonNull(history);
        Objects.requireNonNull(ibanSignal);

        float[] base = featureExtractor
                .extract(transfer, history)
                .toModelInput();

        return RecipientAccountInputEncoder.encode(
                base,
                ibanSignal
        );
    }

    private float infer(float[] raw) {
        float logit = predictor.predictLogit(
                normalizer.normalize(raw)
        );

        if (!Float.isFinite(logit)) {
            throw new IllegalStateException(
                    "Neural network returned non-finite logit"
            );
        }

        return logit;
    }

    public record Prediction(
            float logit,
            NeuralDataQualityPolicy.Coverage coverage
    ) {
        public Prediction {
            Objects.requireNonNull(coverage);
        }
    }
}