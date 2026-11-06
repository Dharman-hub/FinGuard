package ru.finguard.starter.config;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import ru.finguard.core.neural.OnnxModelRegistry;

import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ApprovedReleaseEvidenceTest {

    @TempDir
    Path directory;

    @Test
    void shouldVerifyApprovedArtifactsAndThreshold()
            throws Exception {
        Path model = directory.resolve("finguard-model.onnx");
        Path normalization = directory.resolve("normalization.csv");
        Path metadata = directory.resolve("quality.properties");

        Files.writeString(model, "model-weights");
        Files.writeString(normalization, "normalization-data");

        var version = new OnnxModelRegistry.ModelVersion(
                "bank-v1",
                model,
                normalization,
                sha256(model),
                sha256(normalization),
                70,
                35
        );

        Files.writeString(
                metadata,
                """
                        format=bank-v3-candidate-1
                        eligibleForManualReview=true
                        automaticActivation=false
                        fpBudgetPerThousand=5
                        reviewThresholdLogit=2.25
                        modelSha256=%s
                        normalizationSha256=%s
                        """.formatted(
                        version.modelSha256(),
                        version.normalizationSha256()
                )
        );

        String approvedSha = sha256(metadata);

        var evidence = new ApprovedReleaseEvidence(
                metadata,
                approvedSha
        );

        var verified = evidence.verify(version);

        assertEquals(
                2.25f,
                verified.reviewThresholdLogit()
        );

        assertEquals(5, verified.fpBudgetPerThousand());

        Files.writeString(
                metadata,
                Files.readString(metadata)
                        .replace("2.25", "1.0")
        );

        assertThrows(
                SecurityException.class,
                () -> evidence.verify(version)
        );
    }

    private static String sha256(Path file) throws Exception {
        return HexFormat.of().formatHex(
                MessageDigest.getInstance("SHA-256")
                        .digest(Files.readAllBytes(file))
        );
    }
}