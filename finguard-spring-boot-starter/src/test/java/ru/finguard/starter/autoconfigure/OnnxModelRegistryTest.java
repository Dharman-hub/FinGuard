package ru.finguard.starter.autoconfigure;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import ru.finguard.core.neural.OnnxModelRegistry;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.Objects;

import static org.junit.jupiter.api.Assertions.*;

class OnnxModelRegistryTest {

    @TempDir
    Path temporaryDirectory;

    @Test
    void shouldRegisterAndValidateExistingModel() throws Exception {
        OnnxModelRegistry registry = new OnnxModelRegistry();

        var registered = registry.register(
                "v3.1",
                resourcePath("finguard-v3/finguard-model.onnx"),
                resourcePath("finguard-v3/normalization.csv")
        );

        assertEquals("v3.1", registered.version());
        assertEquals(70, registered.inputSize());
        assertEquals(35, registered.featureCount());
        assertTrue(registered.modelSha256().matches("[0-9a-f]{64}"));
        assertTrue(registered.normalizationSha256().matches("[0-9a-f]{64}"));
        assertEquals(registered, registry.validateForActivation("v3.1"));
        assertEquals(registered, registry.find("v3.1").orElseThrow());
        assertEquals(1, registry.versions().size());
    }

    @Test
    void shouldRejectRepeatedVersion() throws Exception {
        OnnxModelRegistry registry = new OnnxModelRegistry();
        Path model = resourcePath("finguard-v3/finguard-model.onnx");
        Path normalization = resourcePath("finguard-v3/normalization.csv");

        registry.register("v3.1", model, normalization);

        assertThrows(IllegalArgumentException.class,
                () -> registry.register("v3.1", model, normalization));
        assertEquals(1, registry.versions().size());
    }

    @Test
    void shouldRejectChangedModelBeforeActivation() throws Exception {
        OnnxModelRegistry registry = new OnnxModelRegistry();
        Path model = copyResource("finguard-v3/finguard-model.onnx", "model.onnx");
        Path normalization = copyResource("finguard-v3/normalization.csv", "normalization.csv");

        registry.register("v3.2", model, normalization);
        Files.write(model, new byte[]{1}, StandardOpenOption.APPEND);

        assertThrows(IllegalStateException.class,
                () -> registry.validateForActivation("v3.2"));
    }

    @Test
    void shouldRejectChangedNormalizationBeforeActivation() throws Exception {
        OnnxModelRegistry registry = new OnnxModelRegistry();
        Path model = copyResource("finguard-v3/finguard-model.onnx", "model.onnx");
        Path normalization = copyResource("finguard-v3/normalization.csv", "normalization.csv");

        registry.register("v3.3", model, normalization);
        Files.writeString(normalization, "invalid", StandardOpenOption.APPEND);

        assertThrows(IllegalStateException.class,
                () -> registry.validateForActivation("v3.3"));
    }

    @Test
    void shouldNotRegisterInvalidNormalization() throws Exception {
        OnnxModelRegistry registry = new OnnxModelRegistry();
        Path normalization = temporaryDirectory.resolve("invalid.csv");
        Files.writeString(normalization, "invalid profile");

        assertThrows(IOException.class,
                () -> registry.register(
                        "invalid",
                        resourcePath("finguard-v3/finguard-model.onnx"),
                        normalization
                ));

        assertTrue(registry.versions().isEmpty());
    }

    private Path copyResource(String resource, String targetName)
            throws IOException, URISyntaxException {
        return Files.copy(
                resourcePath(resource),
                temporaryDirectory.resolve(targetName),
                StandardCopyOption.REPLACE_EXISTING
        );
    }

    private Path resourcePath(String resource) throws URISyntaxException {
        return Path.of(Objects.requireNonNull(
                getClass().getClassLoader().getResource(resource),
                "Missing test resource: " + resource
        ).toURI());
    }
}