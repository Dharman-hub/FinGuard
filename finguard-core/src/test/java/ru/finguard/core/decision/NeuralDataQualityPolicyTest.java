package ru.finguard.core.decision;

import org.junit.jupiter.api.Test;
import ru.finguard.core.feature.FeatureName;
import ru.finguard.core.feature.V3InputNormalizer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NeuralDataQualityPolicyTest {

    @Test
    void shouldCountObservedZeroAsAvailable() {
        NeuralDataQualityPolicy policy = new NeuralDataQualityPolicy(
                new NeuralDataQualityPolicy.Requirements(3, 1, 1, false),
                new NeuralDataQualityPolicy.Requirements(4, 1, 1, true),
                NeuralDataQualityPolicy.MissingDataAction.RULES_ONLY
        );

        float[] input = inputWith(
                FeatureName.CURRENT_TRANSFER_AMOUNT,
                FeatureName.TRANSFERS_LAST_5_MINUTES,
                FeatureName.COMPLETED_TRANSFERS_TO_RECIPIENT
        );

        NeuralDataQualityPolicy.Coverage coverage = policy.inspect(input);

        assertEquals(3, coverage.observedFeatures());
        assertEquals(1, coverage.observedVelocityFeatures());
        assertEquals(1, coverage.observedRecipientFeatures());
        assertTrue(coverage.neuralApprovalAllowed());
        assertFalse(coverage.neuralRejectionAllowed());
    }

    @Test
    void shouldBlockNeuralRejectionWithoutVelocityHistory() {
        NeuralDataQualityPolicy policy = new NeuralDataQualityPolicy(
                new NeuralDataQualityPolicy.Requirements(1, 0, 0, false),
                new NeuralDataQualityPolicy.Requirements(3, 1, 1, false),
                NeuralDataQualityPolicy.MissingDataAction.RULES_ONLY
        );

        float[] input = inputWith(
                FeatureName.CURRENT_TRANSFER_AMOUNT,
                FeatureName.COMPLETED_TRANSFERS_TO_RECIPIENT,
                FeatureName.UNIQUE_RECIPIENTS_30D
        );

        NeuralDataQualityPolicy.Coverage coverage = policy.inspect(input);

        assertTrue(coverage.neuralApprovalAllowed());
        assertFalse(coverage.neuralRejectionAllowed());
        assertEquals(0, coverage.observedVelocityFeatures());
    }

    @Test
    void shouldDetectAmountOnlyHistory() {
        NeuralDataQualityPolicy policy = new NeuralDataQualityPolicy(
                new NeuralDataQualityPolicy.Requirements(3, 1, 1, false),
                new NeuralDataQualityPolicy.Requirements(5, 2, 1, false),
                NeuralDataQualityPolicy.MissingDataAction.REVIEW
        );

        float[] input = inputWith(
                FeatureName.CURRENT_TRANSFER_AMOUNT
        );

        NeuralDataQualityPolicy.Coverage coverage = policy.inspect(input);

        assertEquals(1, coverage.observedFeatures());
        assertFalse(coverage.neuralApprovalAllowed());
        assertFalse(coverage.neuralRejectionAllowed());
    }

    @Test
    void shouldAllowBankToRequireIban() {
        NeuralDataQualityPolicy policy = new NeuralDataQualityPolicy(
                new NeuralDataQualityPolicy.Requirements(0, 0, 0, false),
                new NeuralDataQualityPolicy.Requirements(1, 0, 0, true),
                NeuralDataQualityPolicy.MissingDataAction.RULES_ONLY
        );

        float[] input = inputWith(
                FeatureName.CURRENT_TRANSFER_AMOUNT
        );

        assertFalse(policy.inspect(input).neuralRejectionAllowed());

        input[69] = 1.0f;

        assertTrue(policy.inspect(input).neuralRejectionAllowed());
    }

    @Test
    void shouldAllowUnrestrictedPolicy() {
        NeuralDataQualityPolicy policy =
                NeuralDataQualityPolicy.unrestricted();

        NeuralDataQualityPolicy.Coverage coverage =
                policy.inspect(new float[70]);

        assertTrue(coverage.neuralApprovalAllowed());
        assertTrue(coverage.neuralRejectionAllowed());
    }

    @Test
    void shouldRejectInvalidRequirementsAndMasks() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new NeuralDataQualityPolicy.Requirements(
                        36, 0, 0, false
                )
        );

        NeuralDataQualityPolicy policy =
                NeuralDataQualityPolicy.unrestricted();

        float[] input = new float[70];
        input[35] = 0.5f;

        assertThrows(
                IllegalArgumentException.class,
                () -> policy.inspect(input)
        );
    }

    private float[] inputWith(FeatureName... features) {
        float[] input = new float[V3InputNormalizer.INPUT_SIZE];

        for (FeatureName feature : features) {
            input[
                    V3InputNormalizer.FEATURE_COUNT + feature.ordinal()
                    ] = 1.0f;
        }

        return input;
    }
}