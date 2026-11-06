package ru.finguard.core.decision;

import ru.finguard.core.feature.FeatureName;
import ru.finguard.core.feature.V3InputNormalizer;

import java.util.Objects;

public record NeuralDataQualityPolicy(
        Requirements approvalRequirements,
        Requirements rejectionRequirements,
        MissingDataAction insufficientDataAction
) {
    private static final int FEATURE_COUNT = V3InputNormalizer.FEATURE_COUNT;
    private static final int INPUT_SIZE = V3InputNormalizer.INPUT_SIZE;
    private static final int VELOCITY_START = FeatureName.TRANSFERS_LAST_5_MINUTES.ordinal();
    private static final int VELOCITY_END = FeatureName.REPEATED_AMOUNTS_LAST_HOUR.ordinal();
    private static final int RECIPIENT_START = FeatureName.COMPLETED_TRANSFERS_TO_RECIPIENT.ordinal();
    private static final int RECIPIENT_END = FeatureName.UNIQUE_RECIPIENTS_30D.ordinal();
    private static final int IBAN_INDEX = FEATURE_COUNT - 1;

    public NeuralDataQualityPolicy {
        Objects.requireNonNull(approvalRequirements);
        Objects.requireNonNull(rejectionRequirements);
        Objects.requireNonNull(insufficientDataAction);
    }

    public static NeuralDataQualityPolicy unrestricted() {
        Requirements noMinimums = new Requirements(0, 0, 0, false);

        return new NeuralDataQualityPolicy(
                noMinimums,
                noMinimums,
                MissingDataAction.RULES_ONLY
        );
    }

    public Coverage inspect(float[] rawModelInput) {
        Objects.requireNonNull(rawModelInput);

        if (rawModelInput.length != INPUT_SIZE) {
            throw new IllegalArgumentException("Expected 70 model inputs");
        }

        int total = 0;
        int velocity = 0;
        int recipient = 0;

        for (int index = 0; index < FEATURE_COUNT; index++) {
            float present = rawModelInput[FEATURE_COUNT + index];

            if (present != 0.0f && present != 1.0f) {
                throw new IllegalArgumentException(
                        "Invalid feature mask at index " + index
                );
            }

            if (present == 0.0f) {
                continue;
            }

            total++;

            if (index >= VELOCITY_START && index <= VELOCITY_END) {
                velocity++;
            }

            if (index >= RECIPIENT_START && index <= RECIPIENT_END) {
                recipient++;
            }
        }

        boolean ibanAvailable =
                rawModelInput[FEATURE_COUNT + IBAN_INDEX] == 1.0f;

        return new Coverage(
                total,
                velocity,
                recipient,
                ibanAvailable,
                meets(
                        approvalRequirements,
                        total,
                        velocity,
                        recipient,
                        ibanAvailable
                ),
                meets(
                        rejectionRequirements,
                        total,
                        velocity,
                        recipient,
                        ibanAvailable
                )
        );
    }

    private static boolean meets(
            Requirements requirements,
            int total,
            int velocity,
            int recipient,
            boolean ibanAvailable
    ) {
        return total >= requirements.minimumObservedFeatures()
                && velocity >= requirements.minimumVelocityFeatures()
                && recipient >= requirements.minimumRecipientFeatures()
                && (!requirements.requireIban() || ibanAvailable);
    }

    public record Requirements(
            int minimumObservedFeatures,
            int minimumVelocityFeatures,
            int minimumRecipientFeatures,
            boolean requireIban
    ) {
        public Requirements {
            if (minimumObservedFeatures < 0
                    || minimumObservedFeatures > FEATURE_COUNT
                    || minimumVelocityFeatures < 0
                    || minimumVelocityFeatures > VELOCITY_END - VELOCITY_START + 1
                    || minimumRecipientFeatures < 0
                    || minimumRecipientFeatures > RECIPIENT_END - RECIPIENT_START + 1) {
                throw new IllegalArgumentException(
                        "Invalid feature availability requirements"
                );
            }
        }
    }

    public record Coverage(
            int observedFeatures,
            int observedVelocityFeatures,
            int observedRecipientFeatures,
            boolean ibanAvailable,
            boolean neuralApprovalAllowed,
            boolean neuralRejectionAllowed
    ) {
    }

    public enum MissingDataAction {
        RULES_ONLY,
        REVIEW
    }
}