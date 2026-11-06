package ru.finguard.starter.config;

import ru.finguard.core.neural.OnnxModelRegistry;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Objects;
import java.util.Properties;

public final class ApprovedReleaseEvidence {

    private static final long MAX_METADATA_BYTES = 65536;

    private final Path metadataPath;
    private final String approvedSha256;

    public ApprovedReleaseEvidence(
            Path metadataPath,
            String approvedSha256
    ) {
        this.metadataPath = Objects.requireNonNull(
                metadataPath
        ).toAbsolutePath().normalize();

        if (approvedSha256 == null
                || !approvedSha256.matches("[0-9a-f]{64}")) {
            throw new IllegalArgumentException(
                    "Invalid approved metadata SHA-256"
            );
        }

        this.approvedSha256 = approvedSha256;
    }

    public Verified verify(
            OnnxModelRegistry.ModelVersion model
    ) throws IOException {
        Objects.requireNonNull(model);

        Path metadata = metadataPath.toRealPath();

        if (!Files.isRegularFile(metadata)
                || Files.size(metadata) > MAX_METADATA_BYTES) {
            throw new IOException(
                    "Invalid approved release metadata"
            );
        }

        byte[] bytes = Files.readAllBytes(metadata);

        if (bytes.length > MAX_METADATA_BYTES
                || !sha256(bytes).equals(approvedSha256)) {
            throw new SecurityException(
                    "Approved release metadata SHA-256 mismatch"
            );
        }

        Path directory = metadata.getParent();

        if (!model.modelPath().toRealPath().equals(
                directory.resolve("finguard-model.onnx").toRealPath()
        ) || !model.normalizationPath().toRealPath().equals(
                directory.resolve("normalization.csv").toRealPath()
        )) {
            throw new SecurityException(
                    "Model artifacts do not belong to approved release"
            );
        }

        Properties properties = new Properties();

        try (var input = new java.io.ByteArrayInputStream(bytes)) {
            properties.load(input);
        }

        if (!"bank-v3-candidate-1".equals(
                properties.getProperty("format")
        ) || !"true".equals(
                properties.getProperty("eligibleForManualReview")
        ) || !"false".equals(
                properties.getProperty("automaticActivation")
        )) {
            throw new SecurityException(
                    "Release has not passed the required approval"
            );
        }

        if (!model.modelSha256().equals(
                properties.getProperty("modelSha256")
        ) || !model.normalizationSha256().equals(
                properties.getProperty("normalizationSha256")
        )) {
            throw new SecurityException(
                    "Approved hashes do not match registered artifacts"
            );
        }

        float threshold;
        int budget;

        try {
            threshold = Float.parseFloat(
                    required(properties, "reviewThresholdLogit")
            );

            budget = Integer.parseInt(
                    required(properties, "fpBudgetPerThousand")
            );
        } catch (NumberFormatException exception) {
            throw new SecurityException(
                    "Invalid approved release thresholds",
                    exception
            );
        }

        if (!Float.isFinite(threshold)
                || budget < 1
                || budget >= 1000) {
            throw new SecurityException(
                    "Invalid approved release parameters"
            );
        }

        return new Verified(
                threshold,
                budget
        );
    }

    private static String required(
            Properties properties,
            String name
    ) {
        String value = properties.getProperty(name);

        if (value == null || value.isBlank()) {
            throw new SecurityException(
                    "Missing approved release field: " + name
            );
        }

        return value;
    }

    private static String sha256(byte[] bytes) {
        try {
            MessageDigest digest =
                    MessageDigest.getInstance("SHA-256");

            return HexFormat.of().formatHex(
                    digest.digest(bytes)
            );
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException(
                    "SHA-256 is unavailable",
                    exception
            );
        }
    }

    public record Verified(
            float reviewThresholdLogit,
            int fpBudgetPerThousand
    ) {
    }
}