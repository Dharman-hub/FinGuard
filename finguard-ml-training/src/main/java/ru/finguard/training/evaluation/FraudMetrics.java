package ru.finguard.training.evaluation;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

public final class FraudMetrics {

    private FraudMetrics() {
    }

    public record Prediction(int label, double score) {

        public Prediction {
            if (label != 0 && label != 1) {
                throw new IllegalArgumentException("Label must be 0 or 1");
            }

            if (!Double.isFinite(score) || score < 0 || score > 1) {
                throw new IllegalArgumentException("Score must be between 0 and 1");
            }
        }
    }

    public record Result(
            int total,
            long positives,
            long truePositives,
            long falsePositives,
            long trueNegatives,
            long falseNegatives,
            double precision,
            double recall,
            double f1,
            double falsePositiveRate,
            double averagePrecision
    ) {
    }

    public static Result evaluate(
            List<Prediction> predictions,
            double threshold
    ) {
        validate(predictions);

        if (!Double.isFinite(threshold) || threshold < 0 || threshold > 1) {
            throw new IllegalArgumentException("Invalid threshold");
        }

        long tp = 0;
        long fp = 0;
        long tn = 0;
        long fn = 0;

        for (Prediction prediction : predictions) {
            boolean detected = prediction.score() >= threshold;

            if (prediction.label() == 1 && detected) {
                tp++;
            } else if (prediction.label() == 0 && detected) {
                fp++;
            } else if (prediction.label() == 0) {
                tn++;
            } else {
                fn++;
            }
        }

        double precision = divide(tp, tp + fp);
        double recall = divide(tp, tp + fn);
        double f1 = divide(2.0 * tp, 2.0 * tp + fp + fn);
        double falsePositiveRate = divide(fp, fp + tn);

        return new Result(
                predictions.size(),
                tp + fn,
                tp,
                fp,
                tn,
                fn,
                precision,
                recall,
                f1,
                falsePositiveRate,
                averagePrecision(predictions)
        );
    }

    public static double findBestThreshold(List<Prediction> predictions) {
        validate(predictions);

        List<Prediction> sorted = sortByScore(predictions);

        long positives = sorted.stream()
                .filter(prediction -> prediction.label() == 1)
                .count();

        long tp = 0;
        long fp = 0;

        double bestThreshold = 1.0;
        double bestF1 = -1.0;

        int index = 0;

        while (index < sorted.size()) {
            double threshold = sorted.get(index).score();

            while (index < sorted.size()
                    && Double.compare(sorted.get(index).score(), threshold) == 0) {
                if (sorted.get(index).label() == 1) {
                    tp++;
                } else {
                    fp++;
                }

                index++;
            }

            long fn = positives - tp;
            double f1 = divide(2.0 * tp, 2.0 * tp + fp + fn);

            if (f1 > bestF1) {
                bestF1 = f1;
                bestThreshold = threshold;
            }
        }

        return bestThreshold;
    }

    private static double averagePrecision(List<Prediction> predictions) {
        List<Prediction> sorted = sortByScore(predictions);

        long totalPositives = sorted.stream()
                .filter(prediction -> prediction.label() == 1)
                .count();

        long detectedPositives = 0;
        double result = 0.0;

        int index = 0;

        while (index < sorted.size()) {
            double score = sorted.get(index).score();
            long groupPositives = 0;

            while (index < sorted.size()
                    && Double.compare(sorted.get(index).score(), score) == 0) {
                if (sorted.get(index).label() == 1) {
                    groupPositives++;
                }

                index++;
            }

            detectedPositives += groupPositives;

            double precision = (double) detectedPositives / index;
            double recallIncrease = (double) groupPositives / totalPositives;

            result += precision * recallIncrease;
        }

        return result;
    }

    private static List<Prediction> sortByScore(List<Prediction> predictions) {
        List<Prediction> sorted = new ArrayList<>(predictions);

        sorted.sort(
                Comparator.comparingDouble(Prediction::score).reversed()
        );

        return sorted;
    }

    private static void validate(List<Prediction> predictions) {
        Objects.requireNonNull(predictions);

        if (predictions.isEmpty()) {
            throw new IllegalArgumentException("Predictions cannot be empty");
        }

        long positives = 0;

        for (Prediction prediction : predictions) {
            Objects.requireNonNull(prediction);

            if (prediction.label() == 1) {
                positives++;
            }
        }

        if (positives == 0 || positives == predictions.size()) {
            throw new IllegalArgumentException(
                    "Evaluation requires both legitimate and fraudulent transactions"
            );
        }
    }

    private static double divide(double numerator, double denominator) {
        return denominator == 0 ? 0.0 : numerator / denominator;
    }
}