package ru.finguard.starter.config;

import ai.onnxruntime.OrtException;
import ru.finguard.core.api.RiskEngine;
import ru.finguard.core.neural.OnnxModelRegistry;
import ru.finguard.core.neural.VersionedOnnxRiskEngine;
import ru.finguard.core.decision.RuntimeThresholdPolicy;

import java.io.IOException;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.ConcurrentModificationException;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.HashMap;
import java.util.Map;

public final class FinGuardModelManager implements AutoCloseable {

    private static final int MAX_AUDIT_ENTRIES = 1000;
    private static final String LEGACY_ACTOR = "LEGACY_API";

    private final OnnxModelRegistry registry;
    private final VersionedOnnxRiskEngine engine;
    private final ModelStateStore store;
    private final RiskEngine riskEngineView;
    private ModelStateStore.State state;

    private final FinGuardSettingsManager settingsManager;
    private RuntimeThresholdPolicy.Snapshot previousPairedPolicy;
    private RuntimeThresholdPolicy.Snapshot activePairedPolicy;

    private final Map<ModelPolicyPreview, ApprovedReleaseEvidence>
            approvedPreviews = new HashMap<>();

    private boolean approvedReleasesRequired;

    public FinGuardModelManager(
            OnnxModelRegistry registry,
            VersionedOnnxRiskEngine engine
    ) {
        this(registry, engine, null);
    }

    public FinGuardModelManager(
            OnnxModelRegistry registry,
            VersionedOnnxRiskEngine engine,
            ModelStateStore store
    ) {
        this(registry, engine, store, null);
    }

    public FinGuardModelManager(
            OnnxModelRegistry registry,
            VersionedOnnxRiskEngine engine,
            ModelStateStore store,
            FinGuardSettingsManager settingsManager
    ) {
        this.registry = Objects.requireNonNull(registry);
        this.engine = Objects.requireNonNull(engine);
        this.store = store;
        this.settingsManager = settingsManager;
        this.riskEngineView = engine::evaluate;

        try {
            Optional<ModelStateStore.State> saved = store == null
                    ? Optional.empty()
                    : Objects.requireNonNull(store.load());

            if (saved.isEmpty()) {
                state = new ModelStateStore.State(
                        0L,
                        0L,
                        engine.activeVersion(),
                        engine.previousVersion().orElse(null),
                        registry.versions(),
                        List.of()
                );

                if (store != null) {
                    store.saveIfRevisionMatches(state, -1L);
                }
            } else {
                ModelStateStore.State restored = saved.get();

                if (!restored.activeVersion().equals(engine.activeVersion())
                        || !Objects.equals(
                        restored.previousVersion(),
                        engine.previousVersion().orElse(null)
                )
                        || !restored.models().equals(registry.versions())) {
                    throw new IllegalStateException(
                            "Model state differs from loaded engine"
                    );
                }

                state = restored;
            }
        } catch (IOException exception) {
            throw new IllegalStateException(
                    "Cannot restore model state",
                    exception
            );
        }
    }

    public synchronized OnnxModelRegistry.ModelVersion register(
            String version,
            Path modelPath,
            Path normalizationPath
    ) throws IOException, OrtException {
        ModelStateStore.State before = state;

        return registry.register(
                version,
                modelPath,
                normalizationPath,
                candidate -> {
                    List<OnnxModelRegistry.ModelVersion> updated =
                            new ArrayList<>(before.models());
                    updated.add(candidate);

                    ModelStateStore.State next = new ModelStateStore.State(
                            Math.addExact(before.revision(), 1L),
                            before.auditVersion(),
                            before.activeVersion(),
                            before.previousVersion(),
                            updated,
                            before.auditHistory()
                    );

                    commit(next, before.revision());
                }
        );
    }

    public synchronized List<OnnxModelRegistry.ModelVersion> versions() {
        return registry.versions();
    }

    public synchronized Optional<OnnxModelRegistry.ModelVersion> find(
            String version
    ) {
        return registry.find(version);
    }

    public synchronized String activeVersion() {
        return engine.activeVersion();
    }

    public synchronized Optional<String> previousVersion() {
        return engine.previousVersion();
    }

    public synchronized long auditVersion() {
        return state.auditVersion();
    }

    public synchronized List<ModelAuditEntry> auditHistory() {
        return state.auditHistory();
    }

    public synchronized void activate(
            String nextVersion,
            String expectedCurrentVersion,
            String actorId
    ) throws IOException, OrtException {

        if (approvedReleasesRequired) {
            throw new IllegalStateException(
                    "Legacy activation is disabled; use an approved release"
            );
        }

        if (store instanceof PairedModelStateStoreAdapter
                || activePairedPolicy != null) {
            throw new IllegalStateException(
                    "Use activateWithPolicy"
            );
        }

        applyChange(
                nextVersion,
                expectedCurrentVersion,
                actorId,
                ChangeType.ACTIVATE
        );
    }

    public synchronized void rollback(
            String expectedCurrentVersion,
            String actorId
    ) throws IOException, OrtException {

        if (approvedReleasesRequired) {
            throw new IllegalStateException(
                    "Use rollbackWithPolicy in approved-release mode"
            );
        }

        if (store instanceof PairedModelStateStoreAdapter
                || activePairedPolicy != null) {
            throw new IllegalStateException(
                    "Use rollbackWithPolicy"
            );
        }

        String target = engine.previousVersion().orElseThrow(
                () -> new IllegalStateException(
                        "No previous model version available"
                )
        );

        applyChange(
                target,
                expectedCurrentVersion,
                actorId,
                ChangeType.ROLLBACK
        );
    }

    public synchronized void requireApprovedReleases() {
        approvedReleasesRequired = true;
    }

    public synchronized ModelPolicyPreview previewApprovedReleaseActivation(
            String targetVersion,
            RuntimeThresholdPolicy runtimePolicy,
            Path qualityFile,
            String bankApprovedSha256
    ) throws IOException, OrtException {
        ApprovedReleaseEvidence evidence =
                new ApprovedReleaseEvidence(
                        qualityFile,
                        bankApprovedSha256
                );

        OnnxModelRegistry.ModelVersion model =
                registry.validateForActivation(targetVersion);

        ApprovedReleaseEvidence.Verified approved =
                evidence.verify(model);

        ModelPolicyPreview preview =
                previewActivationWithPolicy(
                        targetVersion,
                        runtimePolicy,
                        approved.reviewThresholdLogit(),
                        null,
                        approved.fpBudgetPerThousand()
                );

        approvedPreviews.put(preview, evidence);

        return preview;
    }

    public synchronized ModelPolicyPreview previewActivationWithPolicy(
            String targetVersion,
            RuntimeThresholdPolicy runtimePolicy,
            float calibratedReviewLogit,
            Float approvedRejectLogit,
            int fpBudgetPerThousand
    ) throws IOException, OrtException {
        Objects.requireNonNull(runtimePolicy);

        if (fpBudgetPerThousand < 1 || fpBudgetPerThousand >= 1000) {
            throw new IllegalArgumentException("Invalid FP budget");
        }

        String currentVersion = engine.activeVersion();

        if (currentVersion.equals(targetVersion)) {
            throw new IllegalArgumentException("Model is already active");
        }

        OnnxModelRegistry.ModelVersion target =
                registry.validateForActivation(targetVersion);

        RuntimeThresholdPolicy.Snapshot before = runtimePolicy.snapshot();

        if (before.neuralReviewThresholdLogit().isEmpty()) {
            throw new IllegalStateException("Neural policy is disabled");
        }

        RuntimeThresholdPolicy.Snapshot after =
                new RuntimeThresholdPolicy.Snapshot(
                        Math.addExact(before.version(), 1L),
                        before.reviewThreshold(),
                        before.rejectThreshold(),
                        Optional.of(calibratedReviewLogit),
                        Optional.ofNullable(approvedRejectLogit),
                        before.neuralDataQualityPolicy()
                );

        ModelPolicyPreview preview = new ModelPolicyPreview(
                currentVersion,
                target,
                before,
                after,
                fpBudgetPerThousand
        );

        verifyActivationPreview(preview, runtimePolicy);
        return preview;
    }

    public synchronized void verifyActivationPreview(
            ModelPolicyPreview preview,
            RuntimeThresholdPolicy runtimePolicy
    ) throws IOException, OrtException {
        Objects.requireNonNull(preview);
        Objects.requireNonNull(runtimePolicy);

        if (!engine.activeVersion().equals(preview.currentModelVersion())
                || !runtimePolicy.snapshot().equals(preview.before())) {
            throw new ConcurrentModificationException(
                    "Active model or policy changed since preview"
            );
        }

        OnnxModelRegistry.ModelVersion validated =
                registry.validateForActivation(preview.target().version());

        if (!validated.equals(preview.target())) {
            throw new IllegalStateException(
                    "Candidate model artifacts changed"
            );
        }

        RuntimeThresholdPolicy.Snapshot before = preview.before();
        RuntimeThresholdPolicy.Snapshot after = preview.after();

        if (preview.currentModelVersion().equals(preview.target().version())
                || preview.fpBudgetPerThousand() < 1
                || preview.fpBudgetPerThousand() >= 1000
                || after.version() != Math.addExact(before.version(), 1L)
                || after.reviewThreshold() != before.reviewThreshold()
                || after.rejectThreshold() != before.rejectThreshold()
                || after.neuralReviewThresholdLogit().isEmpty()
                || !after.neuralDataQualityPolicy().equals(
                before.neuralDataQualityPolicy())) {
            throw new IllegalArgumentException(
                    "Invalid model-policy activation preview"
            );
        }

        if (!engine.activeVersion().equals(preview.currentModelVersion())
                || !runtimePolicy.snapshot().equals(before)) {
            throw new ConcurrentModificationException(
                    "Active model or policy changed during validation"
            );
        }
    }

    public record ModelPolicyPreview(
            String currentModelVersion,
            OnnxModelRegistry.ModelVersion target,
            RuntimeThresholdPolicy.Snapshot before,
            RuntimeThresholdPolicy.Snapshot after,
            int fpBudgetPerThousand
    ) {
        public ModelPolicyPreview {
            if (currentModelVersion == null
                    || currentModelVersion.isBlank()) {
                throw new IllegalArgumentException(
                        "Current model version is required"
                );
            }

            Objects.requireNonNull(target);
            Objects.requireNonNull(before);
            Objects.requireNonNull(after);
        }
    }

    private void verifyApprovedPreview(
            ModelPolicyPreview preview
    ) throws IOException, OrtException {
        if (!approvedReleasesRequired) {
            return;
        }

        ApprovedReleaseEvidence evidence =
                approvedPreviews.get(preview);

        if (evidence == null) {
            throw new SecurityException(
                    "Model activation requires an approved release"
            );
        }

        OnnxModelRegistry.ModelVersion model =
                registry.validateForActivation(
                        preview.target().version()
                );

        ApprovedReleaseEvidence.Verified approved =
                evidence.verify(model);

        float selectedThreshold = preview.after()
                .neuralReviewThresholdLogit()
                .orElseThrow();

        if (Float.compare(
                selectedThreshold,
                approved.reviewThresholdLogit()
        ) != 0) {
            throw new SecurityException(
                    "Activation threshold differs from approved calibration"
            );
        }

        if (preview.fpBudgetPerThousand()
                != approved.fpBudgetPerThousand()) {
            throw new SecurityException(
                    "Activation FP budget differs from approved release"
            );
        }

        if (preview.after()
                .neuralRejectThresholdLogit()
                .isPresent()) {
            throw new SecurityException(
                    "Automatic rejection requires separate approval"
            );
        }
    }

    public synchronized void activateWithPolicy(
            ModelPolicyPreview preview,
            String actorId
    ) throws IOException, OrtException {
        Objects.requireNonNull(preview);
        verifyApprovedPreview(preview);

        if (store instanceof PairedModelStateStoreAdapter paired) {
            requirePairedSettings();

            applyChange(
                    preview.target().version(),
                    preview.currentModelVersion(),
                    actorId,
                    ChangeType.ACTIVATE,
                    (before, next) -> {
                        verifyPreviewDuringCommit(preview);

                        persistPairedChange(
                                paired,
                                before,
                                next,
                                preview.before(),
                                preview.after()
                        );
                    }
            );

            approvedPreviews.remove(preview);

            return;
        }

        requireInMemoryPairedActivation();

        applyChange(
                preview.target().version(),
                preview.currentModelVersion(),
                actorId,
                ChangeType.ACTIVATE,
                (before, next) -> {
                    verifyPreviewDuringCommit(preview);

                    settingsManager.updateNeuralDecisionThresholds(
                            preview.after()
                                    .neuralReviewThresholdLogit()
                                    .orElseThrow(),
                            preview.after()
                                    .neuralRejectThresholdLogit()
                                    .orElse(null),
                            preview.before().version(),
                            actorId
                    );

                    commit(next, before.revision());
                }
        );

        previousPairedPolicy = preview.before();
        activePairedPolicy = preview.after();
        approvedPreviews.remove(preview);
    }

    private void verifyPreviewDuringCommit(
            ModelPolicyPreview preview
    ) {
        try {
            verifyActivationPreview(
                    preview,
                    settingsManager.runtimePolicy()
            );

            verifyApprovedPreview(preview);

        } catch (IOException | OrtException exception) {
            throw new IllegalStateException(
                    "Cannot verify model-policy preview",
                    exception
            );
        }
    }

    private void requirePairedSettings() {
        if (settingsManager == null
                || settingsManager.hasPersistentAuditStore()) {
            throw new IllegalStateException(
                    "Paired activation requires managed runtime settings"
            );
        }
    }

    private void requireInMemoryPairedActivation() {
        requirePairedSettings();

        if (store != null) {
            throw new UnsupportedOperationException(
                    "Paired activation requires a unified persistent store"
            );
        }
    }

    public synchronized void rollbackWithPolicy(
            String expectedCurrentVersion,
            String actorId
    ) throws IOException, OrtException {
        if (store instanceof PairedModelStateStoreAdapter paired) {
            requirePairedSettings();

            ModelPolicySnapshotStore.State persisted =
                    paired.loadPaired();

            RuntimeThresholdPolicy.Snapshot before =
                    persisted.activePolicy();

            RuntimeThresholdPolicy.Snapshot restore =
                    persisted.previousPolicy();

            if (restore == null) {
                throw new IllegalStateException(
                        "No previous paired policy available"
                );
            }

            RuntimeThresholdPolicy.Snapshot after =
                    new RuntimeThresholdPolicy.Snapshot(
                            Math.addExact(before.version(), 1L),
                            restore.reviewThreshold(),
                            restore.rejectThreshold(),
                            restore.neuralReviewThresholdLogit(),
                            restore.neuralRejectThresholdLogit(),
                            restore.neuralDataQualityPolicy()
                    );

            String target = engine.previousVersion().orElseThrow(
                    () -> new IllegalStateException(
                            "No previous model version"
                    )
            );

            applyChange(
                    target,
                    expectedCurrentVersion,
                    actorId,
                    ChangeType.ROLLBACK,
                    (original, next) -> persistPairedChange(
                            paired,
                            original,
                            next,
                            before,
                            after
                    )
            );

            return;
        }

        requireInMemoryPairedActivation();

        if (previousPairedPolicy == null
                || activePairedPolicy == null) {
            throw new IllegalStateException(
                    "No paired activation available"
            );
        }

        RuntimeThresholdPolicy.Snapshot expected =
                activePairedPolicy;

        RuntimeThresholdPolicy.Snapshot restore =
                previousPairedPolicy;

        RuntimeThresholdPolicy.Snapshot after =
                new RuntimeThresholdPolicy.Snapshot(
                        Math.addExact(expected.version(), 1L),
                        restore.reviewThreshold(),
                        restore.rejectThreshold(),
                        restore.neuralReviewThresholdLogit(),
                        restore.neuralRejectThresholdLogit(),
                        restore.neuralDataQualityPolicy()
                );

        String target = engine.previousVersion().orElseThrow(
                () -> new IllegalStateException(
                        "No previous model version"
                )
        );

        applyChange(
                target,
                expectedCurrentVersion,
                actorId,
                ChangeType.ROLLBACK,
                (before, next) -> {
                    if (!settingsManager.runtimePolicy()
                            .snapshot()
                            .equals(expected)) {
                        throw new ConcurrentModificationException(
                                "Policy changed since paired activation"
                        );
                    }

                    settingsManager.updateNeuralDecisionThresholds(
                            restore.neuralReviewThresholdLogit()
                                    .orElseThrow(),
                            restore.neuralRejectThresholdLogit()
                                    .orElse(null),
                            expected.version(),
                            actorId
                    );

                    commit(next, before.revision());
                }
        );

        previousPairedPolicy = expected;
        activePairedPolicy = after;
    }

    private void persistPairedChange(
            PairedModelStateStoreAdapter paired,
            ModelStateStore.State before,
            ModelStateStore.State next,
            RuntimeThresholdPolicy.Snapshot expectedPolicy,
            RuntimeThresholdPolicy.Snapshot nextPolicy
    ) throws IOException {
        ModelPolicySnapshotStore.State saved =
                paired.loadPaired();

        if (!saved.modelState().equals(before)
                || !saved.activePolicy().equals(expectedPolicy)
                || !settingsManager.runtimePolicy()
                .snapshot()
                .equals(expectedPolicy)
                || settingsManager.snapshotWithVersion().version()
                != expectedPolicy.version()) {
            throw new ConcurrentModificationException(
                    "Persistent model or policy changed"
            );
        }

        if (nextPolicy.version()
                != Math.addExact(expectedPolicy.version(), 1L)) {
            throw new IllegalArgumentException(
                    "Invalid paired policy revision"
            );
        }

        ModelPolicySnapshotStore.State nextSaved =
                new ModelPolicySnapshotStore.State(
                        next,
                        nextPolicy,
                        expectedPolicy
                );

        try {
            paired.savePaired(
                    nextSaved,
                    before.revision()
            );
        } catch (IOException | RuntimeException | Error failure) {
            try {
                if (!paired.loadPaired().equals(saved)) {
                    engine.failClosed();
                }
            } catch (IOException | RuntimeException verificationFailure) {
                engine.failClosed();
                failure.addSuppressed(verificationFailure);
            }

            throw failure;
        }

        try {
            settingsManager.applyPairedPolicy(
                    expectedPolicy,
                    nextPolicy
            );

            state = next;
        } catch (RuntimeException | Error failure) {
            engine.failClosed();
            throw failure;
        }
    }

    public void activate(
            String nextVersion,
            String expectedCurrentVersion
    ) throws IOException, OrtException {
        activate(nextVersion, expectedCurrentVersion, LEGACY_ACTOR);
    }

    public void rollback(String expectedCurrentVersion)
            throws IOException, OrtException {
        rollback(expectedCurrentVersion, LEGACY_ACTOR);
    }

    public RiskEngine riskEngine() {
        return riskEngineView;
    }

    @Override
    public synchronized void close() {
        engine.close();
    }

    private void applyChange(
            String nextVersion,
            String expectedCurrentVersion,
            String actorId,
            ChangeType type
    ) throws IOException, OrtException {
        applyChange(
                nextVersion,
                expectedCurrentVersion,
                actorId,
                type,
                (before, next) -> commit(
                        next,
                        before.revision()
                )
        );
    }

    private void applyChange(
            String nextVersion,
            String expectedCurrentVersion,
            String actorId,
            ChangeType type,
            ChangeCommit changeCommit
    ) throws IOException, OrtException {
        Objects.requireNonNull(changeCommit);
        if (actorId == null || actorId.isBlank()) {
            throw new IllegalArgumentException(
                    "Actor ID is required"
            );
        }

        String current = engine.activeVersion();

        if (!current.equals(expectedCurrentVersion)) {
            throw new ConcurrentModificationException(
                    "Active model version changed"
            );
        }

        if (current.equals(nextVersion)) {
            throw new IllegalArgumentException(
                    "Model version is already active"
            );
        }

        OnnxModelRegistry.ModelVersion target = registry.find(nextVersion)
                .orElseThrow(
                        () -> new IllegalArgumentException(
                                "Unknown model version: " + nextVersion
                        )
                );

        ModelStateStore.State before = state;

        if (!before.activeVersion().equals(current)) {
            throw new IllegalStateException(
                    "Model state out of sync"
            );
        }

        long nextAuditVersion = Math.addExact(
                before.auditVersion(), 1L
        );

        ModelAuditEntry entry = new ModelAuditEntry(
                nextAuditVersion,
                Instant.now(),
                actorId,
                type,
                current,
                nextVersion,
                target.modelSha256(),
                target.normalizationSha256()
        );

        List<ModelAuditEntry> history =
                new ArrayList<>(before.auditHistory());

        history.add(0, entry);

        if (store == null && history.size() > MAX_AUDIT_ENTRIES) {
            history.remove(history.size() - 1);
        }

        ModelStateStore.State next = new ModelStateStore.State(
                Math.addExact(before.revision(), 1L),
                nextAuditVersion,
                nextVersion,
                current,
                before.models(),
                history
        );

        engine.activate(
                nextVersion,
                expectedCurrentVersion,
                () -> changeCommit.commit(before, next),
                settingsManager == null
                        ? new Object()
                        : settingsManager
        );
    }

    @FunctionalInterface
    private interface ChangeCommit {
        void commit(
                ModelStateStore.State before,
                ModelStateStore.State next
        ) throws IOException;
    }

    private void commit(
            ModelStateStore.State next,
            long expectedRevision
    ) throws IOException {
        if (store != null) {
            store.saveIfRevisionMatches(next, expectedRevision);
        }

        state = next;
    }

    public enum ChangeType {
        ACTIVATE,
        ROLLBACK
    }

    public record ModelAuditEntry(
            long version,
            Instant changedAt,
            String actorId,
            ChangeType type,
            String fromVersion,
            String toVersion,
            String modelSha256,
            String normalizationSha256
    ) {
    }
}