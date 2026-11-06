package ru.finguard.core.feature;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class V3InputNormalizerTest {

    @TempDir
    Path directory;

    @Test
    void shouldNormalizeAvailableFeatures() throws IOException {
        V3InputNormalizer normalizer =
                V3InputNormalizer.load(createProfile());

        float[] input = new float[70];
        input[0] = 14.0f;
        input[35] = 1.0f;
        input[34] = 1.0f;
        input[69] = 1.0f;

        float[] result = normalizer.normalize(input);

        assertEquals(2.0f, result[0]);
        assertEquals(1.0f, result[35]);
        assertEquals(1.0f, result[34]);
        assertEquals(1.0f, result[69]);
    }

    @Test
    void shouldKeepMissingFeaturesAtZero() throws IOException {
        V3InputNormalizer normalizer =
                V3InputNormalizer.load(createProfile());

        float[] input = new float[70];
        input[0] = 14.0f;

        float[] result = normalizer.normalize(input);

        assertEquals(0.0f, result[0]);
        assertEquals(0.0f, result[35]);
    }

    @Test
    void shouldNotModifyOriginalInput() throws IOException {
        V3InputNormalizer normalizer =
                V3InputNormalizer.load(createProfile());

        float[] input = new float[70];
        input[0] = 14.0f;
        input[35] = 1.0f;

        float[] original = input.clone();

        normalizer.normalize(input);

        assertArrayEquals(original, input);
    }

    @Test
    void shouldRejectIncorrectInputSize() throws IOException {
        V3InputNormalizer normalizer =
                V3InputNormalizer.load(createProfile());

        assertThrows(
                IllegalArgumentException.class,
                () -> normalizer.normalize(new float[68])
        );
    }

    @Test
    void shouldRejectInvalidIbanFeature() throws IOException {
        V3InputNormalizer normalizer =
                V3InputNormalizer.load(createProfile());

        float[] input = new float[70];
        input[34] = 0.5f;
        input[69] = 1.0f;

        assertThrows(
                IllegalArgumentException.class,
                () -> normalizer.normalize(input)
        );
    }

    @Test
    void shouldRejectIncorrectFeatureOrder() throws IOException {
        Path path = createProfile();

        String content = Files.readString(path);
        Files.writeString(
                path,
                content.replace(
                        "CURRENT_TRANSFER_AMOUNT",
                        "WRONG_FEATURE"
                )
        );

        assertThrows(
                IOException.class,
                () -> V3InputNormalizer.load(path)
        );
    }

    @Test
    void shouldDistinguishMissingValueFromObservedZeroAfterNormalization()
            throws IOException {
        V3InputNormalizer normalizer =
                V3InputNormalizer.load(createProfile());

        float[] missing = new float[70];
        float[] observedZero = new float[70];

        observedZero[35] = 1.0f;

        float[] missingResult = normalizer.normalize(missing);
        float[] observedResult = normalizer.normalize(observedZero);

        assertEquals(0.0f, missingResult[0]);
        assertEquals(0.0f, missingResult[35]);

        assertEquals(-5.0f, observedResult[0]);
        assertEquals(1.0f, observedResult[35]);
    }

    private Path createProfile() throws IOException {
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

        Path path = directory.resolve("normalization.csv");
        Files.writeString(path, content.toString());

        return path;
    }
}