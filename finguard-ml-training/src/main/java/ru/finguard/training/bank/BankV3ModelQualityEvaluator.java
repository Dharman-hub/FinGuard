package ru.finguard.training.bank;

import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import ru.finguard.core.feature.FeatureName;
import ru.finguard.core.feature.V3InputNormalizer;
import ru.finguard.training.evaluation.FraudMetrics;
import ru.finguard.training.model.FraudMlp;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

public final class BankV3ModelQualityEvaluator {

    private static final int INPUT_SIZE = V3InputNormalizer.INPUT_SIZE;
    private static final int FEATURE_COUNT = V3InputNormalizer.FEATURE_COUNT;

    private BankV3ModelQualityEvaluator() {
    }

    public static Report evaluate(
            Path datasetDirectory,
            Path checkpointDirectory,
            String checkpointName,
            Path normalizationFile,
            int falsePositivesPerThousand,
            int batchSize
    ) throws IOException {
        Objects.requireNonNull(datasetDirectory);
        Objects.requireNonNull(checkpointDirectory);
        Objects.requireNonNull(normalizationFile);

        if (checkpointName == null || checkpointName.isBlank() || batchSize <= 0) {
            throw new IllegalArgumentException("Invalid checkpoint or batch size");
        }

        validateBudget(falsePositivesPerThousand);
        V3InputNormalizer normalizer = V3InputNormalizer.load(normalizationFile);

        try (Model model = FraudMlp.create()) {
            try {
                model.load(checkpointDirectory, checkpointName);
            } catch (MalformedModelException exception) {
                throw new IOException("Cannot load DJL checkpoint", exception);
            }

            try (Predictor<NDList, NDList> predictor =
                         model.newPredictor(new NoopTranslator())) {
                List<Score> validation = scoreFile(
                        datasetDirectory.resolve("validation.fgbin"),
                        model, predictor, normalizer, Scenario.ORIGINAL, batchSize
                );

                float threshold = calibrate(validation, falsePositivesPerThousand);
                Metrics validationMetrics = calculate(validation, threshold);

                List<Score> test = scoreFile(
                        datasetDirectory.resolve("test.fgbin"),
                        model, predictor, normalizer, Scenario.ORIGINAL, batchSize
                );

                Metrics testMetrics = calculate(test, threshold);

                EnumMap<Scenario, Metrics> stress = new EnumMap<>(Scenario.class);

                for (Scenario scenario : Scenario.values()) {
                    if (scenario == Scenario.ORIGINAL) {
                        continue;
                    }

                    List<Score> modified = scoreFile(
                            datasetDirectory.resolve("test.fgbin"),
                            model, predictor, normalizer, scenario, batchSize
                    );

                    stress.put(scenario, calculate(modified, threshold));
                }

                return new Report(
                        falsePositivesPerThousand,
                        threshold,
                        validationMetrics,
                        testMetrics,
                        stress
                );
            }
        }
    }

    public static Report evaluateScores(
            List<Score> validation,
            List<Score> test,
            int falsePositivesPerThousand
    ) {
        float threshold = calibrate(validation, falsePositivesPerThousand);

        return new Report(
                falsePositivesPerThousand,
                threshold,
                calculate(validation, threshold),
                calculate(test, threshold),
                Map.of()
        );
    }

    public static float calibrate(
            List<Score> validation,
            int falsePositivesPerThousand
    ) {
        Objects.requireNonNull(validation);
        validateBudget(falsePositivesPerThousand);
        requireBothClasses(validation);

        float[] legitimate = new float[Math.toIntExact(
                validation.stream().filter(sample -> sample.label() == 0).count()
        )];

        int position = 0;

        for (Score sample : validation) {
            if (sample.label() == 0) {
                legitimate[position++] = sample.logit();
            }
        }

        Arrays.sort(legitimate);

        long allowed = (long) legitimate.length
                * falsePositivesPerThousand / 1000;

        return legitimate[
                Math.toIntExact(legitimate.length - allowed - 1)
                ];
    }

    public static Metrics calculate(List<Score> samples, float threshold) {
        Objects.requireNonNull(samples);

        if (!Float.isFinite(threshold)) {
            throw new IllegalArgumentException("Invalid threshold");
        }

        requireBothClasses(samples);

        long tp = 0;
        long fp = 0;
        long tn = 0;
        long fn = 0;

        List<FraudMetrics.Prediction> probabilities =
                new ArrayList<>(samples.size());

        for (Score sample : samples) {
            boolean detected = sample.logit() > threshold;

            if (sample.label() == 1) {
                if (detected) {
                    tp++;
                } else {
                    fn++;
                }
            } else if (detected) {
                fp++;
            } else {
                tn++;
            }

            probabilities.add(
                    new FraudMetrics.Prediction(
                            sample.label(),
                            sigmoid(sample.logit())
                    )
            );
        }

        double ap = FraudMetrics.evaluate(
                probabilities, 0.5
        ).averagePrecision();

        double precision = ratio(tp, tp + fp);
        double recall = ratio(tp, tp + fn);
        double f1 = ratio(2.0 * tp, 2.0 * tp + fp + fn);
        double fpPer1000 = 1000.0 * ratio(fp, fp + tn);

        return new Metrics(
                tp, fp, tn, fn,
                precision, recall, f1, ap, fpPer1000
        );
    }

    private static List<Score> scoreFile(
            Path file,
            Model model,
            Predictor<NDList, NDList> predictor,
            V3InputNormalizer normalizer,
            Scenario scenario,
            int batchSize
    ) throws IOException {
        List<Score> scores = new ArrayList<>();

        try (BankV3DatasetBinary.Reader reader =
                     BankV3DatasetBinary.openReader(file)) {

            while (true) {
                List<BankV3TrainingSample> batch = new ArrayList<>(batchSize);

                for (int i = 0; i < batchSize; i++) {
                    BankV3TrainingSample sample = reader.readNext();

                    if (sample == null) {
                        break;
                    }

                    batch.add(sample);
                }

                if (batch.isEmpty()) {
                    break;
                }

                float[] flat = new float[batch.size() * INPUT_SIZE];

                for (int i = 0; i < batch.size(); i++) {
                    float[] input = batch.get(i).modelInput();
                    applyScenario(input, scenario);

                    float[] normalized = normalizer.normalize(input);

                    System.arraycopy(
                            normalized, 0,
                            flat, i * INPUT_SIZE, INPUT_SIZE
                    );
                }

                try (NDManager manager = model.getNDManager().newSubManager()) {
                    NDArray tensor = manager.create(flat).reshape(
                            new Shape(batch.size(), INPUT_SIZE)
                    );

                    try (NDList result =
                                 predictor.predict(new NDList(tensor))) {
                        float[] logits =
                                result.singletonOrThrow().toFloatArray();

                        if (logits.length != batch.size()) {
                            throw new IOException(
                                    "Unexpected number of predictions"
                            );
                        }

                        for (int i = 0; i < logits.length; i++) {
                            scores.add(new Score(
                                    batch.get(i).label(),
                                    logits[i]
                            ));
                        }
                    } catch (TranslateException exception) {
                        throw new IOException(
                                "DJL inference failed", exception
                        );
                    }
                }
            }
        }

        return scores;
    }

    static void applyScenario(float[] input, Scenario scenario) {
        switch (scenario) {
            case ORIGINAL -> {
            }
            case NO_VELOCITY_HISTORY -> hideRange(
                    input,
                    FeatureName.TRANSFERS_LAST_5_MINUTES.ordinal(),
                    FeatureName.REPEATED_AMOUNTS_LAST_HOUR.ordinal()
            );
            case NO_RECIPIENT_HISTORY_AND_IBAN -> {
                hideRange(
                        input,
                        FeatureName.COMPLETED_TRANSFERS_TO_RECIPIENT.ordinal(),
                        FeatureName.UNIQUE_RECIPIENTS_30D.ordinal()
                );
                hideRange(input, FEATURE_COUNT - 1, FEATURE_COUNT - 1);
            }
            case CURRENT_AMOUNT_ONLY -> hideRange(
                    input, 1, FEATURE_COUNT - 1
            );
        }
    }

    private static void hideRange(float[] input, int from, int to) {
        for (int index = from; index <= to; index++) {
            input[index] = 0.0f;
            input[FEATURE_COUNT + index] = 0.0f;
        }
    }

    private static void validateBudget(int budget) {
        if (budget < 1 || budget >= 1000) {
            throw new IllegalArgumentException(
                    "FP budget must be from 1 to 999 per 1000"
            );
        }
    }

    private static void requireBothClasses(List<Score> scores) {
        if (scores.isEmpty()) {
            throw new IllegalArgumentException(
                    "Empty evaluation dataset"
            );
        }

        long fraud = scores.stream()
                .filter(sample -> sample.label() == 1)
                .count();

        if (fraud == 0 || fraud == scores.size()) {
            throw new IllegalArgumentException(
                    "Both labels must be present"
            );
        }
    }

    private static double sigmoid(float logit) {
        if (logit >= 0) {
            return 1.0 / (1.0 + Math.exp(-logit));
        }

        double exp = Math.exp(logit);
        return exp / (1.0 + exp);
    }

    private static double ratio(double numerator, double denominator) {
        return denominator == 0.0 ? 0.0 : numerator / denominator;
    }

    public record Score(int label, float logit) {
        public Score {
            if ((label != 0 && label != 1) || !Float.isFinite(logit)) {
                throw new IllegalArgumentException("Invalid score");
            }
        }
    }

    public record Metrics(
            long tp,
            long fp,
            long tn,
            long fn,
            double precision,
            double recall,
            double f1,
            double averagePrecision,
            double falsePositivesPerThousand
    ) {
    }

    public record Report(
            int fpBudgetPerThousand,
            float thresholdLogit,
            Metrics validation,
            Metrics test,
            Map<Scenario, Metrics> missingHistory
    ) {
        public Report {
            missingHistory = Map.copyOf(missingHistory);
        }
    }

    public enum Scenario {
        ORIGINAL,
        NO_VELOCITY_HISTORY,
        NO_RECIPIENT_HISTORY_AND_IBAN,
        CURRENT_AMOUNT_ONLY
    }
}