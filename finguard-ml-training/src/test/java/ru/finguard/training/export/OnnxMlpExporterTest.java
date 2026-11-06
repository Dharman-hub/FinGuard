package ru.finguard.training.export;

import ai.djl.Model;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.Trainer;
import ai.djl.training.loss.Loss;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import ru.finguard.training.model.FraudMlp;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class OnnxMlpExporterTest {

    @TempDir
    Path directory;

    @Test
    void shouldExport70InputModelAndMatchDjlLogit() throws Exception {
        Path destination = directory.resolve("finguard-model.onnx");

        try (Model model = FraudMlp.create();
             Trainer trainer = model.newTrainer(
                     new DefaultTrainingConfig(
                             Loss.sigmoidBinaryCrossEntropyLoss()
                     )
             )) {

            trainer.initialize(new Shape(1, FraudMlp.INPUT_SIZE));

            float[] features = new float[FraudMlp.INPUT_SIZE];
            features[0] = 0.75f;
            features[35] = 1.0f;
            features[34] = 1.0f;
            features[69] = 1.0f;

            var input = trainer.getManager().create(
                    features,
                    new Shape(1, FraudMlp.INPUT_SIZE)
            );

            float expected = trainer.evaluate(new NDList(input))
                    .singletonOrThrow()
                    .toFloatArray()[0];

            OnnxMlpExporter.exportLogits(model, destination);

            assertTrue(Files.isRegularFile(destination));
            assertTrue(Files.size(destination) > 1000);

            OrtEnvironment environment = OrtEnvironment.getEnvironment();

            try (var session = environment.createSession(
                    destination.toString()
            );
                 OnnxTensor tensor = OnnxTensor.createTensor(
                         environment,
                         new float[][]{features}
                 );
                 var result = session.run(Map.of("features", tensor))) {

                float[][] output = (float[][]) result.get(0).getValue();

                assertEquals(
                        expected,
                        output[0][0],
                        0.0001f
                );
            }
        }
    }

    @Test
    void shouldRejectLegacy68InputModel() throws Exception {
        Path destination = directory.resolve("legacy.onnx");

        try (Model model = FraudMlp.create();
             Trainer trainer = model.newTrainer(
                     new DefaultTrainingConfig(
                             Loss.sigmoidBinaryCrossEntropyLoss()
                     )
             )) {

            trainer.initialize(new Shape(1, 68));

            assertThrows(
                    IllegalArgumentException.class,
                    () -> OnnxMlpExporter.exportLogits(
                            model,
                            destination
                    )
            );
        }
    }
}