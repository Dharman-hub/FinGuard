package ru.finguard.core.model;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Objects;
import java.util.Optional;

public record RiskEvaluationTrace(
        Optional<String> modelSha256,
        String thresholdsFingerprint,
        int reviewThreshold,
        int rejectThreshold,
        Optional<Float> neuralReviewThresholdLogit,
        long policyVersion,
        Optional<Float> neuralRejectThresholdLogit
) {
    public RiskEvaluationTrace(
            Optional<String> modelSha256,
            String thresholdsFingerprint,
            int reviewThreshold,
            int rejectThreshold,
            Optional<Float> neuralReviewThresholdLogit,
            long policyVersion
    ) {
        this(
                modelSha256,
                thresholdsFingerprint,
                reviewThreshold,
                rejectThreshold,
                neuralReviewThresholdLogit,
                policyVersion,
                Optional.empty()
        );
    }

    public RiskEvaluationTrace {
        Objects.requireNonNull(modelSha256);
        Objects.requireNonNull(thresholdsFingerprint);
        Objects.requireNonNull(neuralReviewThresholdLogit);
        Objects.requireNonNull(neuralRejectThresholdLogit);

        if (policyVersion < 0) {
            throw new IllegalArgumentException(
                    "Invalid policy version"
            );
        }

        if (reviewThreshold <= 0
                || reviewThreshold >= rejectThreshold
                || rejectThreshold > 100) {
            throw new IllegalArgumentException(
                    "Invalid rule thresholds"
            );
        }

        if (modelSha256.isPresent()
                && !modelSha256.orElseThrow()
                .matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException(
                    "Invalid model SHA-256"
            );
        }

        if (neuralReviewThresholdLogit.isPresent()
                && !Float.isFinite(
                neuralReviewThresholdLogit.orElseThrow())) {
            throw new IllegalArgumentException(
                    "Invalid neural review threshold"
            );
        }

        if (neuralRejectThresholdLogit.isPresent()) {
            if (neuralReviewThresholdLogit.isEmpty()) {
                throw new IllegalArgumentException(
                        "Neural rejection requires neural evaluation"
                );
            }

            float reject = neuralRejectThresholdLogit.orElseThrow();
            float review = neuralReviewThresholdLogit.orElseThrow();

            if (!Float.isFinite(reject) || reject <= review) {
                throw new IllegalArgumentException(
                        "Invalid neural reject threshold"
                );
            }
        }

        if (!thresholdsFingerprint.matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException(
                    "Invalid thresholds fingerprint"
            );
        }
    }

    public static RiskEvaluationTrace create(
            String modelSha256,
            int reviewThreshold,
            int rejectThreshold,
            Float neuralThreshold
    ) {
        return create(
                modelSha256,
                reviewThreshold,
                rejectThreshold,
                neuralThreshold,
                0L,
                null
        );
    }

    public static RiskEvaluationTrace create(
            String modelSha256,
            int reviewThreshold,
            int rejectThreshold,
            Float neuralThreshold,
            long policyVersion
    ) {
        return create(
                modelSha256,
                reviewThreshold,
                rejectThreshold,
                neuralThreshold,
                policyVersion,
                null
        );
    }

    public static RiskEvaluationTrace create(
            String modelSha256,
            int reviewThreshold,
            int rejectThreshold,
            Float neuralReviewThreshold,
            long policyVersion,
            Float neuralRejectThreshold
    ) {
        String canonical = reviewThreshold
                + ":"
                + rejectThreshold
                + ":"
                + (neuralReviewThreshold == null
                ? "DISABLED"
                : Float.floatToRawIntBits(neuralReviewThreshold));

        if (neuralRejectThreshold != null) {
            canonical += ":"
                    + Float.floatToRawIntBits(neuralRejectThreshold);
        }

        String fingerprint;

        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");

            fingerprint = HexFormat.of().formatHex(
                    digest.digest(
                            canonical.getBytes(StandardCharsets.UTF_8)
                    )
            );
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException(
                    "SHA-256 is unavailable",
                    exception
            );
        }

        return new RiskEvaluationTrace(
                Optional.ofNullable(modelSha256),
                fingerprint,
                reviewThreshold,
                rejectThreshold,
                Optional.ofNullable(neuralReviewThreshold),
                policyVersion,
                Optional.ofNullable(neuralRejectThreshold)
        );
    }
}