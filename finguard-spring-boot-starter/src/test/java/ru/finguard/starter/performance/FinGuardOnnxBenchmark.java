package ru.finguard.starter.performance;

import ru.finguard.core.api.RiskEngine;
import ru.finguard.core.api.RiskEngineBuilder;
import ru.finguard.core.feature.FeatureExtractor;
import ru.finguard.core.feature.NewSupplierIbanDetector;
import ru.finguard.core.feature.RecipientAccountInputEncoder;
import ru.finguard.core.feature.V3InputNormalizer;
import ru.finguard.core.model.RiskAssessment;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.monitoring.MonitoringRiskEngine;
import ru.finguard.core.monitoring.RiskMetrics;
import ru.finguard.core.neural.OnnxNeuralPredictor;
import ru.finguard.core.neural.V3NeuralRiskEvaluator;
import ru.finguard.core.provider.TransactionDataProvider;

import java.lang.management.ManagementFactory;
import java.math.BigDecimal;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import java.nio.file.Files;

public final class FinGuardOnnxBenchmark {

    private static final int WARMUP = 1500;
    private static final int SAMPLES = 8000;
    private static final int SAMPLES_PER_THREAD = 2000;

    private static volatile long checksumSink;

    public static void main(String[] args) throws Exception {
        Path modelPath = resource("finguard-v3/finguard-model.onnx");
        Path normalizationPath = resource("finguard-v3/normalization.csv");

        TransferData transfer = new TransferData(
                1001L,
                2001L,
                3001L,
                BigDecimal.valueOf(1000),
                "RUB",
                Instant.parse("2026-11-01T00:00:00Z")
        );

        TransactionHistorySnapshot history =
                TransactionHistorySnapshot.empty(transfer.timestamp());

        V3InputNormalizer normalizer =
                V3InputNormalizer.load(normalizationPath);

        float[] baseInput = new FeatureExtractor()
                .extract(transfer, history)
                .toModelInput();

        float[] modelInput = normalizer.normalize(
                RecipientAccountInputEncoder.encode(
                        baseInput,
                        new NewSupplierIbanDetector.Result(0.0f, 0.0f)
                )
        );

        try (OnnxNeuralPredictor predictor =
                     new OnnxNeuralPredictor(modelPath)) {

            V3NeuralRiskEvaluator evaluator =
                    new V3NeuralRiskEvaluator(normalizer, predictor);

            TransactionDataProvider provider = ignored -> history;

            RiskEngine engine = new RiskEngineBuilder(provider)
                    .neuralReview(evaluator, 0.0f)
                    .build();

            RiskMetrics metrics = new RiskMetrics();

            RiskEngine monitored =
                    new MonitoringRiskEngine(engine, metrics);

            Operation directOnnx = () ->
                    Float.floatToIntBits(predictor.predictLogit(modelInput));

            Operation withoutMonitoring = () ->
                    neuralBits(engine.evaluate(transfer));

            Operation withMonitoring = () ->
                    neuralBits(monitored.evaluate(transfer));

            long warmupChecksum = 0L;

            for (int i = 0; i < WARMUP; i++) {
                warmupChecksum += directOnnx.execute();
                warmupChecksum += withoutMonitoring.execute();
                warmupChecksum += withMonitoring.execute();
            }

            checksumSink = warmupChecksum;

            long heapBefore = heapUsed();

            Statistics direct = measure(directOnnx, SAMPLES);
            Statistics off = measure(withoutMonitoring, SAMPLES);
            Statistics on = measure(withMonitoring, SAMPLES);

            int threads = Math.min(
                    4,
                    Runtime.getRuntime().availableProcessors()
            );

            Statistics parallel = measureParallel(
                    withMonitoring,
                    threads,
                    SAMPLES_PER_THREAD
            );

            long heapAfter = heapUsed();

            System.out.println();
            System.out.println("========== FINGUARD ONNX BENCHMARK ==========");
            System.out.println("Java: " + System.getProperty("java.version"));
            System.out.println("Processors: "
                    + Runtime.getRuntime().availableProcessors());

            print("ONNX only", direct);
            print("RiskEngine OFF", off);
            print("RiskEngine ON", on);
            print("Parallel " + threads + " threads", parallel);

            double overhead = on.averageMillis()
                    - off.averageMillis();

            System.out.printf(
                    Locale.ROOT,
                    "%nMonitoring average difference: %+.4f ms%n",
                    overhead
            );

            System.out.printf(
                    Locale.ROOT,
                    "Heap before: %.2f MB%n",
                    heapBefore / 1048576.0
            );

            System.out.printf(
                    Locale.ROOT,
                    "Heap after:  %.2f MB%n",
                    heapAfter / 1048576.0
            );

            RiskMetrics.Snapshot snapshot = metrics.snapshot();

            System.out.println("Recorded checks: " + snapshot.total());
            System.out.println("Recorded failures: " + snapshot.failed());
            System.out.println("Checksum: " + checksumSink);
            System.out.println("============================================");
        }
    }

    private static long neuralBits(RiskAssessment assessment) {
        float logit = assessment.neural()
                .orElseThrow(() -> new IllegalStateException(
                        "ONNX inference did not complete successfully: "
                                + assessment.reason()
                ))
                .logit();

        return Float.floatToIntBits(logit);
    }

    private static Statistics measure(
            Operation operation,
            int samples
    ) {
        long[] durations = new long[samples];
        long checksum = 0L;

        long wallStarted = System.nanoTime();

        for (int i = 0; i < samples; i++) {
            long started = System.nanoTime();

            checksum += operation.execute();

            durations[i] = System.nanoTime() - started;
        }

        long wallDuration = System.nanoTime() - wallStarted;
        checksumSink = checksum;

        return summarize(durations, wallDuration);
    }

    private static Statistics measureParallel(
            Operation operation,
            int threads,
            int samplesPerThread
    ) throws Exception {
        ExecutorService executor =
                Executors.newFixedThreadPool(threads);

        CountDownLatch ready = new CountDownLatch(threads);
        CountDownLatch start = new CountDownLatch(1);

        List<Future<WorkerResult>> futures = new ArrayList<>();

        try {
            for (int i = 0; i < threads; i++) {
                futures.add(executor.submit(() -> {
                    ready.countDown();
                    start.await();

                    long[] durations = new long[samplesPerThread];
                    long checksum = 0L;

                    for (int j = 0; j < samplesPerThread; j++) {
                        long started = System.nanoTime();

                        checksum += operation.execute();

                        durations[j] = System.nanoTime() - started;
                    }

                    return new WorkerResult(durations, checksum);
                }));
            }

            if (!ready.await(20, TimeUnit.SECONDS)) {
                throw new IllegalStateException(
                        "Workers did not become ready"
                );
            }

            long wallStarted = System.nanoTime();
            start.countDown();

            long[] allDurations = new long[threads * samplesPerThread];
            int offset = 0;
            long checksum = 0L;

            for (Future<WorkerResult> future : futures) {
                WorkerResult result =
                        future.get(120, TimeUnit.SECONDS);

                System.arraycopy(
                        result.durations(),
                        0,
                        allDurations,
                        offset,
                        result.durations().length
                );

                offset += result.durations().length;
                checksum += result.checksum();
            }

            long wallDuration = System.nanoTime() - wallStarted;
            checksumSink = checksum;

            return summarize(allDurations, wallDuration);
        } finally {
            start.countDown();
            executor.shutdownNow();
        }
    }

    private static Statistics summarize(
            long[] durations,
            long wallDuration
    ) {
        double total = 0.0;

        for (long duration : durations) {
            total += duration;
        }

        Arrays.sort(durations);

        int count = durations.length;

        return new Statistics(
                count,
                total / count / 1_000_000.0,
                percentile(durations, 0.50),
                percentile(durations, 0.95),
                percentile(durations, 0.99),
                count * 1_000_000_000.0 / wallDuration
        );
    }

    private static double percentile(
            long[] sortedDurations,
            double percentile
    ) {
        int index = (int) Math.ceil(
                sortedDurations.length * percentile
        ) - 1;

        return sortedDurations[index] / 1_000_000.0;
    }

    private static long heapUsed() {
        return ManagementFactory.getMemoryMXBean()
                .getHeapMemoryUsage()
                .getUsed();
    }

    private static Path resource(String name) throws Exception {
        var url = FinGuardOnnxBenchmark.class
                .getClassLoader()
                .getResource(name);

        if (url != null) {
            return Path.of(url.toURI());
        }

        Path fromProject = Path.of(
                "finguard-spring-boot-starter",
                "src",
                "test",
                "resources",
                name
        );

        if (Files.isRegularFile(fromProject)) {
            return fromProject.toAbsolutePath();
        }

        Path fromModule = Path.of(
                "src",
                "test",
                "resources",
                name
        );

        if (Files.isRegularFile(fromModule)) {
            return fromModule.toAbsolutePath();
        }

        throw new IllegalStateException(
                "Resource not found: " + name
                        + ". Working directory: "
                        + System.getProperty("user.dir")
        );
    }

    private static void print(
            String name,
            Statistics statistics
    ) {
        System.out.printf(
                Locale.ROOT,
                "%n%s (%d checks)%n"
                        + "  Average: %.4f ms%n"
                        + "  p50:     %.4f ms%n"
                        + "  p95:     %.4f ms%n"
                        + "  p99:     %.4f ms%n"
                        + "  Speed:   %.0f checks/sec%n",
                name,
                statistics.samples(),
                statistics.averageMillis(),
                statistics.p50Millis(),
                statistics.p95Millis(),
                statistics.p99Millis(),
                statistics.throughput()
        );
    }

    @FunctionalInterface
    private interface Operation {
        long execute();
    }

    private record WorkerResult(
            long[] durations,
            long checksum
    ) {
    }

    private record Statistics(
            int samples,
            double averageMillis,
            double p50Millis,
            double p95Millis,
            double p99Millis,
            double throughput
    ) {
    }
}