package ru.finguard.core.neural;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.Objects;

public final class OnnxNeuralPredictor
        implements NeuralPredictor, AutoCloseable {

    public static final int INPUT_SIZE = 70;
    public static final int FEATURE_COUNT = 35;

    private final OrtEnvironment environment;
    private final OrtSession session;

    public OnnxNeuralPredictor(Path modelPath) throws OrtException {
        Objects.requireNonNull(modelPath, "Model path is required");

        if (!Files.isRegularFile(modelPath)
                || !Files.isReadable(modelPath)) {
            throw new IllegalArgumentException(
                    "ONNX model is not readable: " + modelPath
            );
        }

        environment = OrtEnvironment.getEnvironment();

        try (OrtSession.SessionOptions options =
                     new OrtSession.SessionOptions()) {

            OrtSession loaded = environment.createSession(
                    modelPath.toAbsolutePath().toString(),
                    options
            );

            if (!loaded.getInputNames().contains("features")
                    || !loaded.getOutputNames().contains("risk_logit")) {
                loaded.close();

                throw new IllegalArgumentException(
                        "Unexpected FinGuard ONNX model interface"
                );
            }

            session = loaded;
        }
    }

    @Override
    public float predictLogit(float[] normalizedInput) {
        validateInput(normalizedInput);

        ByteBuffer bytes = ByteBuffer.allocateDirect(
                INPUT_SIZE * Float.BYTES
        ).order(ByteOrder.nativeOrder());

        var buffer = bytes.asFloatBuffer();
        buffer.put(normalizedInput);
        buffer.flip();

        try (OnnxTensor tensor = OnnxTensor.createTensor(
                environment,
                buffer,
                new long[]{1, INPUT_SIZE}
        );
             OrtSession.Result result = session.run(
                     Map.of("features", tensor)
             )) {

            float[][] output = (float[][]) result
                    .get(0)
                    .getValue();

            if (output.length != 1
                    || output[0].length != 1) {
                throw new IllegalStateException(
                        "Unexpected neural network output shape"
                );
            }

            float logit = output[0][0];

            if (!Float.isFinite(logit)) {
                throw new IllegalStateException(
                        "Neural network returned non-finite logit"
                );
            }

            return logit;

        } catch (OrtException exception) {
            throw new IllegalStateException(
                    "FinGuard ONNX inference failed",
                    exception
            );
        }
    }

    private static void validateInput(float[] input) {
        Objects.requireNonNull(
                input,
                "Normalized input is required"
        );

        if (input.length != INPUT_SIZE) {
            throw new IllegalArgumentException(
                    "Expected 70 inputs, got " + input.length
            );
        }

        for (int i = 0; i < INPUT_SIZE; i++) {
            if (!Float.isFinite(input[i])) {
                throw new IllegalArgumentException(
                        "Non-finite input at index " + i
                );
            }
        }

        if (input[34] != 0.0f && input[34] != 1.0f) {
            throw new IllegalArgumentException(
                    "Invalid supplier IBAN feature"
            );
        }

        for (int i = 0; i < FEATURE_COUNT; i++) {
            float mask = input[FEATURE_COUNT + i];

            if (mask != 0.0f && mask != 1.0f) {
                throw new IllegalArgumentException(
                        "Invalid feature mask at index " + i
                );
            }

            if (mask == 0.0f && input[i] != 0.0f) {
                throw new IllegalArgumentException(
                        "Missing feature must have zero value"
                );
            }
        }
    }

    @Override
    public void close() throws OrtException {
        session.close();
    }
}
