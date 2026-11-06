package ru.finguard.core.neural;

import ai.onnxruntime.OrtException;
import ru.finguard.core.api.RiskEngine;
import ru.finguard.core.feature.V3InputNormalizer;
import ru.finguard.core.model.RiskAssessment;
import ru.finguard.core.model.TransferData;

import java.io.IOException;
import java.util.ConcurrentModificationException;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public final class VersionedOnnxRiskEngine implements RiskEngine, AutoCloseable {

    @FunctionalInterface
    public interface EngineFactory {
        RiskEngine create(V3NeuralRiskEvaluator evaluator, String modelSha256);
    }

    @FunctionalInterface
    public interface ActivationCommit {
        void commit() throws IOException;
    }

    private static final System.Logger LOGGER =
            System.getLogger(VersionedOnnxRiskEngine.class.getName());

    private final OnnxModelRegistry registry;
    private final EngineFactory factory;
    private final ReentrantReadWriteLock lock = new ReentrantReadWriteLock(true);

    private LoadedModel active;
    private String previousVersion;
    private boolean closed;
    private boolean failedClosed;

    public VersionedOnnxRiskEngine(
            OnnxModelRegistry registry,
            String initialVersion,
            EngineFactory factory
    ) throws IOException, OrtException {
        this(registry, initialVersion, null, factory);
    }

    public VersionedOnnxRiskEngine(
            OnnxModelRegistry registry,
            String initialVersion,
            String restoredPreviousVersion,
            EngineFactory factory
    ) throws IOException, OrtException {
        this.registry = Objects.requireNonNull(registry);
        this.factory = Objects.requireNonNull(factory);

        if (restoredPreviousVersion != null) {
            if (restoredPreviousVersion.equals(initialVersion)
                    || registry.find(restoredPreviousVersion).isEmpty()) {
                throw new IllegalArgumentException("Invalid previous model version");
            }
        }

        this.active = prepare(initialVersion);
        this.previousVersion = restoredPreviousVersion;
    }

    @Override
    public RiskAssessment evaluate(TransferData transfer) {
        lock.readLock().lock();
        try {
            ensureOpen();
            return active.engine().evaluate(transfer);
        } finally {
            lock.readLock().unlock();
        }
    }

    public String activeVersion() {
        lock.readLock().lock();
        try {
            ensureOpen();
            return active.version();
        } finally {
            lock.readLock().unlock();
        }
    }

    public Optional<String> previousVersion() {
        lock.readLock().lock();
        try {
            ensureOpen();
            return Optional.ofNullable(previousVersion);
        } finally {
            lock.readLock().unlock();
        }
    }

    public void activate(String nextVersion, String expectedCurrentVersion)
            throws IOException, OrtException {
        activate(nextVersion, expectedCurrentVersion, () -> {
        });
    }

    public void activate(
            String nextVersion,
            String expectedCurrentVersion,
            ActivationCommit commit
    ) throws IOException, OrtException {
        activate(nextVersion, expectedCurrentVersion, commit, new Object());
    }

    public void activate(
            String nextVersion,
            String expectedCurrentVersion,
            ActivationCommit commit,
            Object activationMutex
    ) throws IOException, OrtException {
        Objects.requireNonNull(nextVersion);
        Objects.requireNonNull(expectedCurrentVersion);
        Objects.requireNonNull(commit);
        Objects.requireNonNull(activationMutex);

        LoadedModel candidate = prepare(nextVersion);
        LoadedModel retired = null;
        boolean installed = false;

        try {
            lock.writeLock().lock();
            try {
                synchronized (activationMutex) {
                    ensureOpen();

                    if (!active.version().equals(expectedCurrentVersion)) {
                        throw new ConcurrentModificationException(
                                "Active model version changed"
                        );
                    }

                    if (active.version().equals(nextVersion)) {
                        throw new IllegalArgumentException(
                                "Model version is already active"
                        );
                    }

                    commit.commit();

                    retired = active;
                    active = candidate;
                    previousVersion = retired.version();
                    installed = true;
                }
            } finally {
                lock.writeLock().unlock();
            }
        } finally {
            if (!installed) {
                closeRetired(candidate);
            }
        }

        closeRetired(retired);
    }

    public void rollback(String expectedCurrentVersion)
            throws IOException, OrtException {
        String target;

        lock.readLock().lock();
        try {
            ensureOpen();

            if (!active.version().equals(expectedCurrentVersion)) {
                throw new ConcurrentModificationException(
                        "Active model version changed"
                );
            }

            if (previousVersion == null) {
                throw new IllegalStateException(
                        "No previous model version available"
                );
            }

            target = previousVersion;
        } finally {
            lock.readLock().unlock();
        }

        activate(target, expectedCurrentVersion);
    }

    @Override
    public void close() {
        LoadedModel retired;

        lock.writeLock().lock();
        try {
            if (closed) {
                return;
            }

            closed = true;
            retired = active;
            active = null;
            previousVersion = null;
        } finally {
            lock.writeLock().unlock();
        }

        try {
            retired.predictor().close();
        } catch (OrtException exception) {
            throw new IllegalStateException(
                    "Unable to close ONNX session", exception
            );
        }
    }

    private LoadedModel prepare(String version) throws IOException, OrtException {
        OnnxModelRegistry.ModelVersion model =
                registry.validateForActivation(version);

        V3InputNormalizer normalizer =
                V3InputNormalizer.load(model.normalizationPath());
        OnnxNeuralPredictor predictor =
                new OnnxNeuralPredictor(model.modelPath());

        try {
            if (!model.equals(registry.validateForActivation(version))) {
                throw new IllegalStateException(
                        "Model artifacts changed while preparing: " + version
                );
            }

            RiskEngine engine = Objects.requireNonNull(
                    factory.create(
                            new V3NeuralRiskEvaluator(normalizer, predictor),
                            model.modelSha256()
                    ),
                    "EngineFactory returned null"
            );

            return new LoadedModel(version, engine, predictor);
        } catch (IOException | OrtException | RuntimeException exception) {
            try {
                predictor.close();
            } catch (OrtException closeError) {
                exception.addSuppressed(closeError);
            }
            throw exception;
        }
    }

    public void failClosed() {
        lock.writeLock().lock();
        try {
            failedClosed = true;
        } finally {
            lock.writeLock().unlock();
        }
    }

    private void ensureOpen() {
        if (closed) {
            throw new IllegalStateException(
                    "Versioned ONNX engine is closed"
            );
        }

        if (failedClosed) {
            throw new IllegalStateException(
                    "Model-policy state may be inconsistent; restart FinGuard"
            );
        }
    }

    private static void closeRetired(LoadedModel model) {
        try {
            model.predictor().close();
        } catch (OrtException exception) {
            LOGGER.log(System.Logger.Level.WARNING,
                    "Unable to close retired ONNX session", exception);
        }
    }

    private record LoadedModel(
            String version,
            RiskEngine engine,
            OnnxNeuralPredictor predictor
    ) {
    }
}