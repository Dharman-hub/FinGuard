package ru.finguard.training.dataset.synsepa;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.Model;
import ai.djl.nn.Activation;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ru.finguard.training.export.OnnxMlpExporter;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;

public final class SynSepV3OnnxExportApplication {

    private static final String MODEL_NAME = "synsepa-v3-70";

    private static final Path MODEL_DIRECTORY = Path.of(
            "models",
            "synsepa-v3",
            "comparison",
            MODEL_NAME
    );

    private static final Path OUTPUT = Path.of(
            "models",
            "synsepa-v3",
            "finguard-model.onnx"
    );

    private SynSepV3OnnxExportApplication() {
    }

    public static void main(String[] args) throws IOException {
        System.out.println(
                "========== FINGUARD V3 ONNX EXPORT =========="
        );

        System.out.println("Model: " + MODEL_NAME);
        System.out.println("Architecture: 70 -> 32 -> 16 -> 1");
        System.out.println("Output: risk_logit");
        System.out.println("Sigmoid: DISABLED");
        System.out.println("Retraining: NOT REQUIRED");

        try (Model model = Model.newInstance(
                MODEL_NAME,
                Device.cpu(),
                "PyTorch"
        )) {
            model.setBlock(
                    new SequentialBlock()
                            .add(
                                    Linear.builder()
                                            .setUnits(32)
                                            .build()
                            )
                            .add(Activation::relu)
                            .add(
                                    Linear.builder()
                                            .setUnits(16)
                                            .build()
                            )
                            .add(Activation::relu)
                            .add(
                                    Linear.builder()
                                            .setUnits(1)
                                            .build()
                            )
            );

            try {
                model.load(MODEL_DIRECTORY, MODEL_NAME);
            } catch (MalformedModelException exception) {
                throw new IOException(
                        "Unable to load FinGuard v3 checkpoint",
                        exception
                );
            }

            OnnxMlpExporter.exportLogits(model, OUTPUT);
        }

        if (!Files.isRegularFile(OUTPUT)
                || Files.size(OUTPUT) == 0) {
            throw new IOException(
                    "ONNX model was not created"
            );
        }

        System.out.println();
        System.out.println("ONNX: " + OUTPUT.toAbsolutePath());

        System.out.printf(
                Locale.US,
                "Size: %.2f KB%n",
                Files.size(OUTPUT) / 1024.0
        );

        System.out.println(
                "FINGUARD V3 ONNX EXPORT COMPLETED"
        );
    }
}