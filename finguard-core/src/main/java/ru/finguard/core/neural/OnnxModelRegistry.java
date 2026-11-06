package ru.finguard.core.neural;

import ai.onnxruntime.OrtException;
import ru.finguard.core.feature.V3InputNormalizer;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;

public final class OnnxModelRegistry {

    private final Map<String, ModelVersion> models = new LinkedHashMap<>();

    @FunctionalInterface
    public interface RegistrationCommit {
        void commit(ModelVersion candidate) throws IOException;
    }

    public synchronized ModelVersion register(
            String version,
            Path modelPath,
            Path normalizationPath
    ) throws IOException, OrtException {
        return register(version, modelPath, normalizationPath, candidate -> {
        });
    }

    public synchronized ModelVersion register(
            String version,
            Path modelPath,
            Path normalizationPath,
            RegistrationCommit commit
    ) throws IOException, OrtException {
        Objects.requireNonNull(commit);
        validateVersion(version);

        if (models.containsKey(version)) {
            throw new IllegalArgumentException(
                    "Model version already registered: " + version
            );
        }

        Path model = readablePath(modelPath, "ONNX model");
        Path normalization = readablePath(
                normalizationPath, "Normalization profile"
        );

        ModelVersion candidate = inspect(version, model, normalization);
        commit.commit(candidate);
        models.put(version, candidate);
        return candidate;
    }

    public synchronized Optional<ModelVersion> find(String version) {
        return Optional.ofNullable(models.get(version));
    }

    public synchronized List<ModelVersion> versions() {
        return List.copyOf(models.values());
    }

    public synchronized ModelVersion validateForActivation(
            String version
    ) throws IOException, OrtException {
        ModelVersion registered = models.get(version);

        if (registered == null) {
            throw new IllegalArgumentException(
                    "Unknown model version: " + version
            );
        }

        if (!registered.modelSha256().equals(sha256(registered.modelPath()))
                || !registered.normalizationSha256().equals(
                sha256(registered.normalizationPath()))) {
            throw new IllegalStateException(
                    "Model artifacts changed after registration: " + version
            );
        }

        ModelVersion checked = inspect(
                version,
                registered.modelPath(),
                registered.normalizationPath()
        );

        if (!registered.equals(checked)) {
            throw new IllegalStateException(
                    "Model artifacts changed during validation: " + version
            );
        }

        return registered;
    }

    private static ModelVersion inspect(
            String version,
            Path modelPath,
            Path normalizationPath
    ) throws IOException, OrtException {
        if (OnnxNeuralPredictor.INPUT_SIZE != V3InputNormalizer.INPUT_SIZE
                || OnnxNeuralPredictor.FEATURE_COUNT
                != V3InputNormalizer.FEATURE_COUNT) {
            throw new IllegalStateException(
                    "Neural predictor and normalizer are incompatible"
            );
        }

        String modelHash = sha256(modelPath);
        String normalizationHash = sha256(normalizationPath);

        V3InputNormalizer.load(normalizationPath);

        try (OnnxNeuralPredictor predictor =
                     new OnnxNeuralPredictor(modelPath)) {
            predictor.predictLogit(
                    new float[OnnxNeuralPredictor.INPUT_SIZE]
            );
        }

        if (!modelHash.equals(sha256(modelPath))
                || !normalizationHash.equals(sha256(normalizationPath))) {
            throw new IllegalStateException(
                    "Model artifacts changed during validation: " + version
            );
        }

        return new ModelVersion(
                version,
                modelPath,
                normalizationPath,
                modelHash,
                normalizationHash,
                OnnxNeuralPredictor.INPUT_SIZE,
                OnnxNeuralPredictor.FEATURE_COUNT
        );
    }

    private static Path readablePath(Path path, String name)
            throws IOException {
        Objects.requireNonNull(path, name + " path is required");
        Path resolved = path.toRealPath();

        if (!Files.isRegularFile(resolved) || !Files.isReadable(resolved)) {
            throw new IOException(name + " is not readable: " + resolved);
        }

        return resolved;
    }

    private static String sha256(Path path) throws IOException {
        MessageDigest digest;

        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException(
                    "SHA-256 is unavailable", exception
            );
        }

        try (InputStream input = Files.newInputStream(path)) {
            byte[] buffer = new byte[8192];
            int count;

            while ((count = input.read(buffer)) != -1) {
                digest.update(buffer, 0, count);
            }
        }

        return HexFormat.of().formatHex(digest.digest());
    }

    private static void validateVersion(String version) {
        if (version == null
                || !version.matches("[A-Za-z0-9][A-Za-z0-9._-]{0,63}")) {
            throw new IllegalArgumentException(
                    "Invalid model version: " + version
            );
        }
    }

    public record ModelVersion(
            String version,
            Path modelPath,
            Path normalizationPath,
            String modelSha256,
            String normalizationSha256,
            int inputSize,
            int featureCount
    ) {
    }
}