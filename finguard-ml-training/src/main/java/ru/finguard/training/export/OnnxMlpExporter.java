package ru.finguard.training.export;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ru.finguard.training.model.FraudMlp;
import ru.finguard.training.onnx.proto.AttributeProto;
import ru.finguard.training.onnx.proto.GraphProto;
import ru.finguard.training.onnx.proto.ModelProto;
import ru.finguard.training.onnx.proto.NodeProto;
import ru.finguard.training.onnx.proto.OperatorSetIdProto;
import ru.finguard.training.onnx.proto.TensorProto;
import ru.finguard.training.onnx.proto.TensorShapeProto;
import ru.finguard.training.onnx.proto.TypeProto;
import ru.finguard.training.onnx.proto.ValueInfoProto;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

public final class OnnxMlpExporter {

    private static final int FLOAT_TYPE = 1;
    private static final int INTEGER_ATTRIBUTE = 2;
    private static final long IR_VERSION = 8;
    private static final long OPSET_VERSION = 13;

    private OnnxMlpExporter() {
    }

    public static void exportLogits(
            Model model,
            Path destination
    ) throws IOException {
        Objects.requireNonNull(model);
        Objects.requireNonNull(destination);

        if (!(model.getBlock() instanceof SequentialBlock network)) {
            throw new IllegalArgumentException(
                    "Expected sequential MLP"
            );
        }

        var layers = network.getChildren();

        if (layers.size() != 5) {
            throw new IllegalArgumentException(
                    "Expected Linear, ReLU, Linear, ReLU, Linear"
            );
        }

        Block first = layers.get(0).getValue();
        Block second = layers.get(2).getValue();
        Block third = layers.get(4).getValue();

        NDArray firstWeights = parameter(first, "weight");
        Shape firstShape = firstWeights.getShape();

        if (firstShape.dimension() != 2
                || firstShape.get(0) != 32
                || firstShape.get(1) != FraudMlp.INPUT_SIZE) {
            throw new IllegalArgumentException(
                    "Expected first layer with 70 inputs and 32 units"
            );
        }

        GraphProto.Builder graph = GraphProto.newBuilder()
                .setName("finguard-fraud-mlp")
                .addInput(valueInfo("features", FraudMlp.INPUT_SIZE))
                .addOutput(valueInfo("risk_logit", 1))
                .addInitializer(tensor(
                        "w1",
                        firstWeights,
                        32,
                        FraudMlp.INPUT_SIZE
                ))
                .addInitializer(tensor(
                        "b1",
                        parameter(first, "bias"),
                        32
                ))
                .addInitializer(tensor(
                        "w2",
                        parameter(second, "weight"),
                        16,
                        32
                ))
                .addInitializer(tensor(
                        "b2",
                        parameter(second, "bias"),
                        16
                ))
                .addInitializer(tensor(
                        "w3",
                        parameter(third, "weight"),
                        1,
                        16
                ))
                .addInitializer(tensor(
                        "b3",
                        parameter(third, "bias"),
                        1
                ))
                .addNode(gemm(
                        "features",
                        "w1",
                        "b1",
                        "dense1"
                ))
                .addNode(activation(
                        "Relu",
                        "dense1",
                        "relu1"
                ))
                .addNode(gemm(
                        "relu1",
                        "w2",
                        "b2",
                        "dense2"
                ))
                .addNode(activation(
                        "Relu",
                        "dense2",
                        "relu2"
                ))
                .addNode(gemm(
                        "relu2",
                        "w3",
                        "b3",
                        "risk_logit"
                ));

        ModelProto onnxModel = ModelProto.newBuilder()
                .setIrVersion(IR_VERSION)
                .setProducerName("FinGuard")
                .addOpsetImport(
                        OperatorSetIdProto.newBuilder()
                                .setVersion(OPSET_VERSION)
                                .build()
                )
                .setGraph(graph.build())
                .build();

        Path absolute = destination.toAbsolutePath();
        Files.createDirectories(absolute.getParent());

        try (var output = Files.newOutputStream(absolute)) {
            onnxModel.writeTo(output);
        }
    }

    private static NDArray parameter(
            Block block,
            String name
    ) {
        var parameter = block.getParameters().get(name);

        if (parameter == null || parameter.getArray() == null) {
            throw new IllegalStateException(
                    "Model parameter is not initialized: " + name
            );
        }

        return parameter.getArray();
    }

    private static TensorProto tensor(
            String name,
            NDArray array,
            long... dimensions
    ) {
        if (!array.getShape().equals(new Shape(dimensions))) {
            throw new IllegalArgumentException(
                    "Unexpected tensor shape: " + name
            );
        }

        TensorProto.Builder builder = TensorProto.newBuilder()
                .setName(name)
                .setDataType(FLOAT_TYPE);

        for (long dimension : dimensions) {
            builder.addDims(dimension);
        }

        for (float value : array.toFloatArray()) {
            if (!Float.isFinite(value)) {
                throw new IllegalArgumentException(
                        "Non-finite model parameter: " + name
                );
            }

            builder.addFloatData(value);
        }

        return builder.build();
    }

    private static NodeProto gemm(
            String input,
            String weight,
            String bias,
            String output
    ) {
        return NodeProto.newBuilder()
                .setOpType("Gemm")
                .addInput(input)
                .addInput(weight)
                .addInput(bias)
                .addOutput(output)
                .addAttribute(
                        AttributeProto.newBuilder()
                                .setName("transB")
                                .setType(INTEGER_ATTRIBUTE)
                                .setI(1)
                                .build()
                )
                .build();
    }

    private static NodeProto activation(
            String operation,
            String input,
            String output
    ) {
        return NodeProto.newBuilder()
                .setOpType(operation)
                .addInput(input)
                .addOutput(output)
                .build();
    }

    private static ValueInfoProto valueInfo(
            String name,
            int features
    ) {
        TensorShapeProto shape = TensorShapeProto.newBuilder()
                .addDim(
                        TensorShapeProto.Dimension.newBuilder()
                                .setDimParam("batch")
                                .build()
                )
                .addDim(
                        TensorShapeProto.Dimension.newBuilder()
                                .setDimValue(features)
                                .build()
                )
                .build();

        TypeProto type = TypeProto.newBuilder()
                .setTensorType(
                        TypeProto.Tensor.newBuilder()
                                .setElemType(FLOAT_TYPE)
                                .setShape(shape)
                                .build()
                )
                .build();

        return ValueInfoProto.newBuilder()
                .setName(name)
                .setType(type)
                .build();
    }
}