package ru.finguard.training.bank;

import java.nio.file.Path;
import java.time.Instant;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.Consumer;

public final class BankV3ManualTrainingManager implements AutoCloseable {

    private final ExecutorService executor;
    private final Pipeline pipeline;
    private final ResourceLimits limits;

    private boolean closed;
    private long lastJobId;
    private Status status;
    private Outcome outcome;

    public BankV3ManualTrainingManager(ResourceLimits limits) {
        this(limits, BankV3ManualTrainingManager::execute);
    }

    BankV3ManualTrainingManager(ResourceLimits limits, Pipeline pipeline) {
        this.limits = Objects.requireNonNull(limits);
        this.pipeline = Objects.requireNonNull(pipeline);
        this.executor = Executors.newSingleThreadExecutor(
                task -> new Thread(task, "finguard-manual-training")
        );
    }

    public synchronized long startFromScratch(Request request) {
        return start(new Command(Mode.FROM_SCRATCH, request, null));
    }

    public synchronized long startFineTune(
            Request request,
            SourceCheckpoint source
    ) {
        return start(new Command(Mode.FINE_TUNE, request, source));
    }

    private long start(Command command) {
        Request request = command.request();

        if (request.settings().epochs() > limits.maxEpochs()
                || request.settings().batchSize() > limits.maxBatchSize()
                || request.settings().shuffleBlockSize()
                > limits.maxShuffleBlockSize()
                || request.evaluationBatchSize()
                > limits.maxEvaluationBatchSize()) {
            throw new IllegalArgumentException(
                    "Training resource limits exceeded"
            );
        }

        if (closed) {
            throw new IllegalStateException(
                    "Training manager is closed"
            );
        }

        if (status != null && status.state() == State.RUNNING) {
            throw new IllegalStateException(
                    "A training job is already running"
            );
        }

        long jobId = Math.incrementExact(lastJobId);
        lastJobId = jobId;

        status = new Status(
                jobId,
                command.mode(),
                State.RUNNING,
                Stage.TRAINING,
                Instant.now(),
                null,
                null
        );

        outcome = null;

        try {
            executor.execute(() -> runJob(jobId, command));
        } catch (RuntimeException exception) {
            status = new Status(
                    jobId,
                    command.mode(),
                    State.FAILED,
                    Stage.FAILED,
                    status.startedAt(),
                    Instant.now(),
                    exception.toString()
            );

            notifyAll();
            throw exception;
        }

        return jobId;
    }

    private void runJob(long jobId, Command command) {
        try {
            Outcome completed = Objects.requireNonNull(
                    pipeline.run(
                            command,
                            stage -> updateStage(jobId, stage)
                    ),
                    "Training outcome is required"
            );

            synchronized (this) {
                outcome = completed;

                status = new Status(
                        jobId,
                        command.mode(),
                        State.SUCCEEDED,
                        Stage.COMPLETED,
                        status.startedAt(),
                        Instant.now(),
                        null
                );

                notifyAll();
            }
        } catch (Throwable failure) {
            synchronized (this) {
                status = new Status(
                        jobId,
                        command.mode(),
                        State.FAILED,
                        Stage.FAILED,
                        status.startedAt(),
                        Instant.now(),
                        failure.toString()
                );

                notifyAll();
            }

            if (failure instanceof Error error) {
                throw error;
            }
        }
    }

    private synchronized void updateStage(long jobId, Stage stage) {
        if (status == null
                || status.jobId() != jobId
                || status.state() != State.RUNNING) {
            throw new IllegalStateException(
                    "Training job is no longer active"
            );
        }

        status = new Status(
                jobId,
                status.mode(),
                State.RUNNING,
                Objects.requireNonNull(stage),
                status.startedAt(),
                null,
                null
        );
    }

    public synchronized Optional<Status> status() {
        return Optional.ofNullable(status);
    }

    public synchronized Optional<Outcome> result(long jobId) {
        requireJob(jobId);
        return Optional.ofNullable(outcome);
    }

    public synchronized Status awaitCompletion(long jobId)
            throws InterruptedException {
        requireJob(jobId);

        while (status.state() == State.RUNNING) {
            wait();
        }

        return status;
    }

    private void requireJob(long jobId) {
        if (status == null || status.jobId() != jobId) {
            throw new IllegalArgumentException(
                    "Unknown or expired job ID"
            );
        }
    }

    @Override
    public synchronized void close() {
        closed = true;
        executor.shutdown();
    }

    private static Outcome execute(
            Command command,
            Consumer<Stage> progress
    ) throws Exception {
        Request request = command.request();
        BankV3FromScratchTrainer.Report training;

        progress.accept(Stage.TRAINING);

        if (command.mode() == Mode.FROM_SCRATCH) {
            training = BankV3FromScratchTrainer.train(
                    request.datasetDirectory(),
                    request.trainingDirectory(),
                    request.settings()
            );
        } else {
            SourceCheckpoint source = command.source();

            training = BankV3FromScratchTrainer.fineTune(
                    request.datasetDirectory(),
                    request.trainingDirectory(),
                    request.settings(),
                    source.directory(),
                    source.modelName(),
                    source.normalization()
            );
        }

        progress.accept(Stage.EVALUATING);

        BankV3ModelQualityEvaluator.Report quality =
                BankV3ModelQualityEvaluator.evaluate(
                        request.datasetDirectory(),
                        request.trainingDirectory().resolve("checkpoints"),
                        "best",
                        request.trainingDirectory().resolve("normalization.csv"),
                        request.fpBudgetPerThousand(),
                        request.evaluationBatchSize()
                );

        progress.accept(Stage.VALIDATING);

        requireReleaseEligibility(
                quality,
                request.releaseCriteria()
        );

        progress.accept(Stage.EXPORTING);

        BankV3OnnxRelease.Published published =
                BankV3OnnxRelease.publish(
                        request.trainingDirectory(),
                        request.releaseDirectory(),
                        quality,
                        request.releaseCriteria()
                );

        return new Outcome(training, quality, published);
    }

    static BankV3CandidateReleaseGate.Decision requireReleaseEligibility(
            BankV3ModelQualityEvaluator.Report quality,
            BankV3CandidateReleaseGate.Criteria criteria
    ) {
        BankV3CandidateReleaseGate.Decision decision =
                BankV3CandidateReleaseGate.assess(
                        quality,
                        criteria
                );

        if (!decision.eligibleForManualReview()) {
            throw new IllegalStateException(
                    "Model candidate rejected: "
                            + String.join("; ", decision.issues())
            );
        }

        return decision;
    }

    @FunctionalInterface
    interface Pipeline {
        Outcome run(
                Command command,
                Consumer<Stage> progress
        ) throws Exception;
    }

    record Command(
            Mode mode,
            Request request,
            SourceCheckpoint source
    ) {
        Command {
            Objects.requireNonNull(mode);
            Objects.requireNonNull(request);

            if (mode == Mode.FINE_TUNE) {
                Objects.requireNonNull(
                        source,
                        "Fine-tuning requires checkpoint"
                );
            } else if (source != null) {
                throw new IllegalArgumentException(
                        "From-scratch training cannot use a checkpoint"
                );
            }
        }
    }

    public record Request(
            Path datasetDirectory,
            Path trainingDirectory,
            Path releaseDirectory,
            BankV3FromScratchTrainer.Settings settings,
            int fpBudgetPerThousand,
            int evaluationBatchSize,
            BankV3CandidateReleaseGate.Criteria releaseCriteria
    ) {
        public Request {
            Objects.requireNonNull(datasetDirectory);
            Objects.requireNonNull(trainingDirectory);
            Objects.requireNonNull(releaseDirectory);
            Objects.requireNonNull(settings);
            Objects.requireNonNull(releaseCriteria);

            if (fpBudgetPerThousand < 1
                    || fpBudgetPerThousand >= 1000
                    || evaluationBatchSize <= 0) {
                throw new IllegalArgumentException(
                        "Invalid evaluation settings"
                );
            }

            if (releaseCriteria.fpBudgetPerThousand()
                    != fpBudgetPerThousand) {
                throw new IllegalArgumentException(
                        "Release criteria and evaluation FP budgets differ"
                );
            }

            datasetDirectory =
                    datasetDirectory.toAbsolutePath().normalize();

            trainingDirectory =
                    trainingDirectory.toAbsolutePath().normalize();

            releaseDirectory =
                    releaseDirectory.toAbsolutePath().normalize();

            if (trainingDirectory.startsWith(datasetDirectory)
                    || datasetDirectory.startsWith(trainingDirectory)
                    || releaseDirectory.startsWith(trainingDirectory)
                    || trainingDirectory.startsWith(releaseDirectory)
                    || releaseDirectory.startsWith(datasetDirectory)
                    || datasetDirectory.startsWith(releaseDirectory)) {
                throw new IllegalArgumentException(
                        "Training paths must be separate"
                );
            }
        }
    }

    public record ResourceLimits(
            int maxEpochs,
            int maxBatchSize,
            int maxShuffleBlockSize,
            int maxEvaluationBatchSize
    ) {
        public ResourceLimits {
            if (maxEpochs <= 0
                    || maxBatchSize <= 0
                    || maxShuffleBlockSize <= 0
                    || maxEvaluationBatchSize <= 0) {
                throw new IllegalArgumentException(
                        "Invalid resource limits"
                );
            }
        }
    }

    public record SourceCheckpoint(
            Path directory,
            String modelName,
            Path normalization
    ) {
        public SourceCheckpoint {
            Objects.requireNonNull(directory);
            Objects.requireNonNull(normalization);

            if (modelName == null || modelName.isBlank()) {
                throw new IllegalArgumentException(
                        "Checkpoint name is required"
                );
            }
        }
    }

    public record Outcome(
            BankV3FromScratchTrainer.Report training,
            BankV3ModelQualityEvaluator.Report quality,
            BankV3OnnxRelease.Published release
    ) {
        public Outcome {
            Objects.requireNonNull(training);
            Objects.requireNonNull(quality);
            Objects.requireNonNull(release);
        }
    }

    public record Status(
            long jobId,
            Mode mode,
            State state,
            Stage stage,
            Instant startedAt,
            Instant finishedAt,
            String error
    ) {
    }

    public enum Mode {
        FROM_SCRATCH,
        FINE_TUNE
    }

    public enum State {
        RUNNING,
        SUCCEEDED,
        FAILED
    }

    public enum Stage {
        TRAINING,
        EVALUATING,
        VALIDATING,
        EXPORTING,
        COMPLETED,
        FAILED
    }
}