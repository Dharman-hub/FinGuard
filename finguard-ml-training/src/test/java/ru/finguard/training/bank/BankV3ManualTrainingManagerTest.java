package ru.finguard.training.bank;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class BankV3ManualTrainingManagerTest {

    @Test
    @Timeout(5)
    void shouldRejectConcurrentTrainingAndReturnResult() throws Exception {
        CountDownLatch started = new CountDownLatch(1);
        CountDownLatch release = new CountDownLatch(1);

        BankV3ManualTrainingManager.Pipeline pipeline =
                (command, progress) -> {
                    started.countDown();

                    if (!release.await(3, TimeUnit.SECONDS)) {
                        throw new IllegalStateException("Test timeout");
                    }

                    progress.accept(
                            BankV3ManualTrainingManager.Stage.EXPORTING
                    );

                    return outcome();
                };

        try (var manager = new BankV3ManualTrainingManager(
                limits(),
                pipeline
        )) {
            long jobId = manager.startFromScratch(request());

            assertTrue(started.await(3, TimeUnit.SECONDS));

            assertEquals(
                    BankV3ManualTrainingManager.State.RUNNING,
                    manager.status().orElseThrow().state()
            );

            assertThrows(
                    IllegalStateException.class,
                    () -> manager.startFromScratch(request())
            );

            release.countDown();

            var status = manager.awaitCompletion(jobId);

            assertEquals(
                    BankV3ManualTrainingManager.State.SUCCEEDED,
                    status.state()
            );

            assertEquals(
                    BankV3ManualTrainingManager.Stage.COMPLETED,
                    status.stage()
            );

            assertTrue(manager.result(jobId).isPresent());
            assertNotNull(status.finishedAt());
            assertNull(status.error());
        } finally {
            release.countDown();
        }
    }

    @Test
    @Timeout(5)
    void shouldReportFailureAndAllowAnotherManualStart() throws Exception {
        AtomicInteger attempts = new AtomicInteger();

        BankV3ManualTrainingManager.Pipeline pipeline =
                (command, progress) -> {
                    if (attempts.getAndIncrement() == 0) {
                        throw new IllegalArgumentException(
                                "Invalid training dataset"
                        );
                    }

                    return outcome();
                };

        try (var manager = new BankV3ManualTrainingManager(
                limits(),
                pipeline
        )) {
            long firstJob = manager.startFromScratch(request());

            var failed = manager.awaitCompletion(firstJob);

            assertEquals(
                    BankV3ManualTrainingManager.State.FAILED,
                    failed.state()
            );

            assertTrue(
                    failed.error().contains("Invalid training dataset")
            );

            assertTrue(manager.result(firstJob).isEmpty());

            long secondJob = manager.startFromScratch(request());

            assertNotEquals(firstJob, secondJob);

            var succeeded = manager.awaitCompletion(secondJob);

            assertEquals(
                    BankV3ManualTrainingManager.State.SUCCEEDED,
                    succeeded.state()
            );
        }
    }

    @Test
    @Timeout(5)
    void shouldSupportFineTuningBankModel() throws Exception {
        BankV3ManualTrainingManager.Pipeline pipeline =
                (command, progress) -> {
                    assertEquals(
                            BankV3ManualTrainingManager.Mode.FINE_TUNE,
                            command.mode()
                    );

                    assertEquals("best", command.source().modelName());

                    return outcome();
                };

        try (var manager = new BankV3ManualTrainingManager(
                limits(),
                pipeline
        )) {
            long jobId = manager.startFineTune(
                    request(),
                    new BankV3ManualTrainingManager.SourceCheckpoint(
                            Path.of("models", "previous", "checkpoints"),
                            "best",
                            Path.of("models", "previous", "normalization.csv")
                    )
            );

            assertEquals(
                    BankV3ManualTrainingManager.State.SUCCEEDED,
                    manager.awaitCompletion(jobId).state()
            );
        }
    }

    @Test
    void shouldEnforceBankResourceLimits() {
        try (var manager = new BankV3ManualTrainingManager(
                limits()
        )) {
            var excessive = new BankV3ManualTrainingManager.Request(
                    Path.of("data", "bank"),
                    Path.of("models", "trained"),
                    Path.of("models", "released"),
                    new BankV3FromScratchTrainer.Settings(
                            20, 8, 16, 0.001f, 20.0f
                    ),
                    5,
                    16,
                    criteria()
            );

            assertThrows(
                    IllegalArgumentException.class,
                    () -> manager.startFromScratch(excessive)
            );

            assertTrue(manager.status().isEmpty());
        }
    }

    private BankV3ManualTrainingManager.Request request() {
        return new BankV3ManualTrainingManager.Request(
                Path.of("data", "bank"),
                Path.of("models", "trained"),
                Path.of("models", "released"),
                new BankV3FromScratchTrainer.Settings(
                        1, 8, 16, 0.001f, 20.0f
                ),
                5,
                16,
                criteria()
        );
    }

    private BankV3ManualTrainingManager.ResourceLimits limits() {
        return new BankV3ManualTrainingManager.ResourceLimits(
                2, 16, 32, 32
        );
    }

    private BankV3ManualTrainingManager.Outcome outcome() {
        var metrics = new BankV3ModelQualityEvaluator.Metrics(
                1, 0, 1, 0,
                1.0, 1.0, 1.0, 1.0, 0.0
        );

        var training = new BankV3FromScratchTrainer.Report(
                Path.of("models", "trained"),
                new BankV3FromScratchTrainer.Counts(1, 1),
                new BankV3FromScratchTrainer.Counts(1, 1),
                1.0f,
                1,
                0.1,
                List.of()
        );

        var quality = new BankV3ModelQualityEvaluator.Report(
                5,
                1.5f,
                metrics,
                metrics,
                Map.of()
        );

        var published = new BankV3OnnxRelease.Published(
                Path.of("models", "released", "finguard-model.onnx"),
                Path.of("models", "released", "normalization.csv"),
                Path.of("models", "released", "training.properties")
        );

        return new BankV3ManualTrainingManager.Outcome(
                training,
                quality,
                published
        );
    }

    private BankV3CandidateReleaseGate.Criteria criteria() {
        return new BankV3CandidateReleaseGate.Criteria(
                5,
                1,
                1,
                1,
                1,
                1000.0,
                0.0,
                0.0,
                1000.0,
                0.0
        );
    }
}