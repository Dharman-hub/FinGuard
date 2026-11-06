package ru.finguard.starter.config;

import ru.finguard.core.decision.RuleDecisionThresholds;
import ru.finguard.core.decision.RuntimeThresholdPolicy;
import ru.finguard.core.neural.NeuralReviewThreshold;
import ru.finguard.core.decision.NeuralDataQualityPolicy;

import java.time.Instant;
import java.util.ArrayDeque;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

public final class FinGuardSettingsManager {

    private static final int MAX_AUDIT_ENTRIES = 1000;
    private static final String LEGACY_ACTOR = "LEGACY_API";

    private final RuleDecisionThresholds ruleThresholds;
    private final NeuralReviewThreshold neuralThreshold;
    private final RuntimeThresholdPolicy runtimePolicy;
    private final SettingsAuditStore auditStore;
    private final ArrayDeque<AuditEntry> auditHistory = new ArrayDeque<>();

    private long version;
    private boolean pairedStateControlled;
    private RuntimeThresholdPolicy.PairedMutationPermit pairedPermit;

    public FinGuardSettingsManager(
            RuleDecisionThresholds ruleThresholds,
            NeuralReviewThreshold neuralThreshold
    ) {
        this(ruleThresholds, neuralThreshold, null);
    }

    public FinGuardSettingsManager(
            RuleDecisionThresholds ruleThresholds,
            NeuralReviewThreshold neuralThreshold,
            SettingsAuditStore auditStore
    ) {
        this.ruleThresholds = Objects.requireNonNull(
                ruleThresholds,
                "Rule decision thresholds are required"
        );
        this.neuralThreshold = neuralThreshold;
        this.runtimePolicy = null;
        this.auditStore = auditStore;
        this.version = 0;

        if (auditStore != null) {
            refreshFromStore();
        }
    }

    public FinGuardSettingsManager(
            RuntimeThresholdPolicy runtimePolicy,
            SettingsAuditStore auditStore
    ) {
        this.runtimePolicy = Objects.requireNonNull(
                runtimePolicy,
                "Runtime policy is required"
        );
        this.ruleThresholds = null;
        this.neuralThreshold = null;
        this.auditStore = auditStore;
        this.version = runtimePolicy.snapshot().version();

        if (auditStore != null) {
            refreshFromStore();
        }
    }

    boolean hasPersistentAuditStore() {
        return auditStore != null;
    }

    public synchronized void restrictToPairedState() {
        if (runtimePolicy == null || auditStore != null) {
            throw new IllegalStateException(
                    "Paired state requires RuntimeThresholdPolicy without independent audit store"
            );
        }

        if (pairedStateControlled) {
            return;
        }

        pairedPermit = runtimePolicy.restrictToPairedMutations();
        pairedStateControlled = true;
    }

    synchronized void applyPairedPolicy(
            RuntimeThresholdPolicy.Snapshot expected,
            RuntimeThresholdPolicy.Snapshot next
    ) {
        Objects.requireNonNull(expected);
        Objects.requireNonNull(next);

        if (!pairedStateControlled) {
            throw new IllegalStateException(
                    "Paired policy mode is not enabled"
            );
        }

        if (runtimePolicy == null) {
            throw new IllegalStateException(
                    "Runtime threshold policy is required"
            );
        }

        if (version != expected.version()
                || !runtimePolicy.snapshot().equals(expected)) {
            throw new java.util.ConcurrentModificationException(
                    "Runtime policy changed"
            );
        }

        if (next.version() != Math.addExact(expected.version(), 1L)) {
            throw new IllegalArgumentException(
                    "Expected the next policy version"
            );
        }

        runtimePolicy.activatePaired(
                pairedPermit,
                expected.version(),
                next.version(),
                next.reviewThreshold(),
                next.rejectThreshold(),
                next.neuralReviewThresholdLogit().orElse(null),
                next.neuralRejectThresholdLogit().orElse(null),
                next.neuralDataQualityPolicy()
        );

        version = next.version();
    }

    public RuntimeThresholdPolicy runtimePolicy() {
        if (runtimePolicy == null) {
            throw new IllegalStateException(
                    "This manager was created with legacy threshold objects"
            );
        }

        return runtimePolicy;
    }

    public synchronized Snapshot snapshot() {
        if (runtimePolicy != null) {
            RuntimeThresholdPolicy.Snapshot active =
                    runtimePolicy.snapshot();

            return new Snapshot(
                    active.reviewThreshold(),
                    active.rejectThreshold(),
                    active.neuralReviewThresholdLogit(),
                    active.neuralRejectThresholdLogit(),
                    active.neuralDataQualityPolicy()
            );
        }

        RuleDecisionThresholds.Snapshot rules = ruleThresholds.snapshot();

        return new Snapshot(
                rules.reviewThreshold(),
                rules.rejectThreshold(),
                neuralThreshold == null
                        ? Optional.empty()
                        : Optional.of(neuralThreshold.get()),
                Optional.empty(),
                NeuralDataQualityPolicy.unrestricted()
        );
    }

    public synchronized VersionedSnapshot snapshotWithVersion() {
        verifyLocalPolicyVersion();
        return new VersionedSnapshot(version, snapshot());
    }

    public synchronized List<AuditEntry> auditHistory() {
        return List.copyOf(auditHistory);
    }

    public synchronized void updateRuleThresholds(
            int reviewThreshold,
            int rejectThreshold
    ) {
        updateRuleThresholds(
                reviewThreshold,
                rejectThreshold,
                version,
                LEGACY_ACTOR
        );
    }

    public synchronized void updateRuleThresholds(
            int reviewThreshold,
            int rejectThreshold,
            long expectedVersion,
            String actorId
    ) {
        validateVersion(expectedVersion);
        String actor = requireActor(actorId);

        new RuleDecisionThresholds.Snapshot(
                reviewThreshold,
                rejectThreshold
        );

        Snapshot before = snapshot();
        Snapshot after = new Snapshot(
                reviewThreshold,
                rejectThreshold,
                before.neuralReviewThresholdLogit(),
                before.neuralRejectThresholdLogit(),
                before.neuralDataQualityPolicy()
        );

        persistAndApply(
                ChangeType.RULE_THRESHOLDS,
                actor,
                before,
                after,
                () -> ruleThresholds.update(
                        reviewThreshold,
                        rejectThreshold
                )
        );
    }

    public synchronized void useNeuralMode(NeuralReviewMode mode) {
        useNeuralMode(mode, version, LEGACY_ACTOR);
    }

    public synchronized void useNeuralMode(
            NeuralReviewMode mode,
            long expectedVersion,
            String actorId
    ) {
        Objects.requireNonNull(mode, "Neural mode is required");

        if (mode == NeuralReviewMode.CUSTOM) {
            throw new IllegalArgumentException(
                    "Use setCustomNeuralThreshold for CUSTOM mode"
            );
        }

        validateVersion(expectedVersion);
        String actor = requireActor(actorId);
        requireNeuralEnabled();

        float value = mode.thresholdLogit();
        Snapshot before = snapshot();
        Snapshot after = new Snapshot(
                before.reviewThreshold(),
                before.rejectThreshold(),
                Optional.of(value),
                before.neuralRejectThresholdLogit(),
                before.neuralDataQualityPolicy()
        );

        persistAndApply(
                ChangeType.NEURAL_MODE,
                actor,
                before,
                after,
                () -> neuralThreshold.update(value)
        );
    }

    public synchronized void setCustomNeuralThreshold(float value) {
        setCustomNeuralThreshold(value, version, LEGACY_ACTOR);
    }

    public synchronized void setCustomNeuralThreshold(
            float value,
            long expectedVersion,
            String actorId
    ) {
        validateVersion(expectedVersion);
        String actor = requireActor(actorId);
        requireNeuralEnabled();

        if (!Float.isFinite(value)) {
            throw new IllegalArgumentException(
                    "Neural review threshold must be finite"
            );
        }

        Snapshot before = snapshot();
        Snapshot after = new Snapshot(
                before.reviewThreshold(),
                before.rejectThreshold(),
                Optional.of(value),
                before.neuralRejectThresholdLogit(),
                before.neuralDataQualityPolicy()
        );

        persistAndApply(
                ChangeType.NEURAL_CUSTOM,
                actor,
                before,
                after,
                () -> neuralThreshold.update(value)
        );
    }

    public synchronized void updateNeuralDecisionThresholds(
            float reviewThresholdLogit,
            Float rejectThresholdLogit,
            long expectedVersion,
            String actorId
    ) {
        validateVersion(expectedVersion);
        String actor = requireActor(actorId);
        requireNeuralEnabled();

        if (runtimePolicy == null) {
            throw new IllegalStateException(
                    "Neural rejection requires RuntimeThresholdPolicy"
            );
        }

        Snapshot before = snapshot();

        Snapshot after = new Snapshot(
                before.reviewThreshold(),
                before.rejectThreshold(),
                Optional.of(reviewThresholdLogit),
                Optional.ofNullable(rejectThresholdLogit),
                before.neuralDataQualityPolicy()
        );

        validateSnapshot(after);

        persistAndApply(
                ChangeType.NEURAL_DECISION_THRESHOLDS,
                actor,
                before,
                after,
                () -> {
                }
        );
    }

    public synchronized void updateNeuralDataQualityPolicy(
            NeuralDataQualityPolicy quality,
            long expectedVersion,
            String actorId
    ) {
        Objects.requireNonNull(
                quality,
                "Data quality policy is required"
        );

        validateVersion(expectedVersion);
        String actor = requireActor(actorId);
        requireNeuralEnabled();

        if (runtimePolicy == null) {
            throw new IllegalStateException(
                    "Data quality policy requires RuntimeThresholdPolicy"
            );
        }

        Snapshot before = snapshot();

        Snapshot after = new Snapshot(
                before.reviewThreshold(),
                before.rejectThreshold(),
                before.neuralReviewThresholdLogit(),
                before.neuralRejectThresholdLogit(),
                quality
        );

        validateSnapshot(after);

        persistAndApply(
                ChangeType.NEURAL_DATA_QUALITY,
                actor,
                before,
                after,
                () -> {
                }
        );
    }

    public synchronized void rollbackLastChange(
            long expectedVersion,
            String actorId
    ) {
        validateVersion(expectedVersion);

        if (version == 0) {
            throw new IllegalStateException(
                    "No settings change available for rollback"
            );
        }

        rollbackToVersion(
                version - 1,
                expectedVersion,
                actorId
        );
    }

    public synchronized void rollbackToVersion(
            long targetVersion,
            long expectedVersion,
            String actorId
    ) {
        validateVersion(expectedVersion);
        String actor = requireActor(actorId);

        if (targetVersion < 0 || targetVersion >= version) {
            throw new IllegalArgumentException(
                    "Target version must be earlier than current version"
            );
        }

        Snapshot current = snapshot();
        long lookupVersion = targetVersion == 0
                ? 1
                : targetVersion;

        Optional<AuditEntry> targetEntry;

        if (auditStore != null) {
            long storedVersion = auditStore.lastVersion();

            Optional<AuditEntry> latest = Objects.requireNonNull(
                    auditStore.latestEntry(),
                    "Settings audit store returned null"
            );

            if (storedVersion != version
                    || latest.isEmpty()
                    || latest.get().version() != version
                    || !current.equals(latest.get().after())) {
                throw new IllegalStateException(
                        "Persistent settings are out of sync"
                );
            }

            targetEntry = Objects.requireNonNull(
                    auditStore.findEntry(lookupVersion),
                    "Settings audit store returned null"
            );
        } else {
            AuditEntry latest = auditHistory.peekFirst();

            if (latest == null
                    || latest.version() != version
                    || !current.equals(latest.after())) {
                throw new IllegalStateException(
                        "Local settings history is inconsistent"
                );
            }

            targetEntry = auditHistory.stream()
                    .filter(entry -> entry.version() == lookupVersion)
                    .findFirst();
        }

        AuditEntry target = targetEntry.orElseThrow(
                () -> new IllegalStateException(
                        "Target settings version is unavailable"
                )
        );

        if (target.version() != lookupVersion) {
            throw new IllegalStateException(
                    "Incorrect audit entry version"
            );
        }

        Snapshot restored = Objects.requireNonNull(
                targetVersion == 0
                        ? target.before()
                        : target.after(),
                "Target settings snapshot is missing"
        );

        validateSnapshot(restored);

        persistAndApply(
                ChangeType.ROLLBACK,
                actor,
                current,
                restored,
                () -> applyLegacy(restored)
        );
    }

    public synchronized boolean refreshFromStore() {
        if (auditStore == null) {
            return false;
        }

        verifyLocalPolicyVersion();

        long storedVersion = auditStore.lastVersion();

        Optional<AuditEntry> latest = Objects.requireNonNull(
                auditStore.latestEntry(),
                "Settings audit store returned null"
        );

        if (storedVersion == 0 && latest.isEmpty()) {
            if (version != 0) {
                throw new IllegalStateException(
                        "Persistent settings history was cleared"
                );
            }

            return false;
        }

        if (storedVersion <= 0
                || storedVersion == Long.MAX_VALUE
                || latest.isEmpty()
                || latest.get().version() != storedVersion) {
            throw new IllegalStateException(
                    "Persistent settings version and latest entry do not match"
            );
        }

        if (storedVersion < version) {
            throw new IllegalStateException(
                    "Persistent settings version is older than local version"
            );
        }

        AuditEntry entry = latest.orElseThrow();

        Snapshot saved = Objects.requireNonNull(
                entry.after(),
                "Persisted settings snapshot is missing"
        );

        validateSnapshot(saved);

        if (storedVersion == version) {
            if (!snapshot().equals(saved)) {
                throw new IllegalStateException(
                        "Local settings differ from persisted settings"
                );
            }

            return false;
        }

        if (runtimePolicy != null) {
            activatePolicy(storedVersion, saved);
        } else {
            applyLegacy(saved);
        }

        version = storedVersion;
        auditHistory.addFirst(entry);
        trimHistory();

        return true;
    }

    private void persistAndApply(
            ChangeType type,
            String actorId,
            Snapshot before,
            Snapshot after,
            Runnable legacyApply
    ) {

        if (pairedStateControlled) {
            throw new IllegalStateException(
                    "Direct settings updates are disabled in paired storage mode"
            );
        }

        verifyLocalPolicyVersion();

        long nextVersion = Math.addExact(version, 1L);

        AuditEntry entry = new AuditEntry(
                nextVersion,
                Instant.now(),
                actorId,
                type,
                before,
                after
        );

        if (auditStore != null) {
            auditStore.appendIfVersionMatches(entry, version);
        }

        if (runtimePolicy != null) {
            activatePolicy(nextVersion, after);
        } else {
            legacyApply.run();
        }

        version = nextVersion;
        auditHistory.addFirst(entry);
        trimHistory();
    }

    private void activatePolicy(
            long newVersion,
            Snapshot after
    ) {
        runtimePolicy.activate(
                version,
                newVersion,
                after.reviewThreshold(),
                after.rejectThreshold(),
                after.neuralReviewThresholdLogit().orElse(null),
                after.neuralRejectThresholdLogit().orElse(null),
                after.neuralDataQualityPolicy()
        );
    }

    private void applyLegacy(Snapshot restored) {
        ruleThresholds.update(
                restored.reviewThreshold(),
                restored.rejectThreshold()
        );

        if (neuralThreshold != null) {
            neuralThreshold.update(
                    restored.neuralReviewThresholdLogit().orElseThrow()
            );
        }
    }

    private void validateSnapshot(Snapshot saved) {
        new RuntimeThresholdPolicy.Snapshot(
                0L,
                saved.reviewThreshold(),
                saved.rejectThreshold(),
                saved.neuralReviewThresholdLogit(),
                saved.neuralRejectThresholdLogit(),
                saved.neuralDataQualityPolicy()
        );

        boolean hasNeural = runtimePolicy != null
                ? runtimePolicy.snapshot()
                .neuralReviewThresholdLogit().isPresent()
                : neuralThreshold != null;

        if (saved.neuralReviewThresholdLogit().isPresent() != hasNeural) {
            throw new IllegalStateException(
                    "Persisted neural configuration is incompatible"
            );
        }

        if (saved.neuralReviewThresholdLogit().isPresent()
                && !Float.isFinite(
                saved.neuralReviewThresholdLogit().orElseThrow()
        )) {
            throw new IllegalStateException(
                    "Persisted neural threshold is invalid"
            );
        }
    }

    private void requireNeuralEnabled() {
        boolean enabled = runtimePolicy != null
                ? runtimePolicy.snapshot()
                .neuralReviewThresholdLogit().isPresent()
                : neuralThreshold != null;

        if (!enabled) {
            throw new IllegalStateException(
                    "Neural evaluation is disabled"
            );
        }
    }

    private void verifyLocalPolicyVersion() {
        if (runtimePolicy != null
                && runtimePolicy.snapshot().version() != version) {
            throw new IllegalStateException(
                    "Runtime policy was changed outside FinGuardSettingsManager"
            );
        }
    }

    private void validateVersion(long expectedVersion) {
        verifyLocalPolicyVersion();

        if (version != expectedVersion) {
            throw new IllegalStateException(
                    "Settings version conflict: expected "
                            + expectedVersion
                            + ", current "
                            + version
            );
        }
    }

    private static String requireActor(String actorId) {
        if (actorId == null || actorId.isBlank()) {
            throw new IllegalArgumentException(
                    "Actor ID is required"
            );
        }

        return actorId;
    }

    private void trimHistory() {
        if (auditHistory.size() > MAX_AUDIT_ENTRIES) {
            auditHistory.removeLast();
        }
    }

    public enum ChangeType {
        RULE_THRESHOLDS,
        NEURAL_MODE,
        NEURAL_CUSTOM,
        ROLLBACK,
        NEURAL_DATA_QUALITY,
        NEURAL_DECISION_THRESHOLDS,
    }

    public record Snapshot(
            int reviewThreshold,
            int rejectThreshold,
            Optional<Float> neuralReviewThresholdLogit,
            Optional<Float> neuralRejectThresholdLogit,
            NeuralDataQualityPolicy neuralDataQualityPolicy
    ) {
        public Snapshot(
                int reviewThreshold,
                int rejectThreshold,
                Optional<Float> neuralReviewThresholdLogit
        ) {
            this(
                    reviewThreshold,
                    rejectThreshold,
                    neuralReviewThresholdLogit,
                    Optional.empty(),
                    NeuralDataQualityPolicy.unrestricted()
            );
        }

        public Snapshot(
                int reviewThreshold,
                int rejectThreshold,
                Optional<Float> neuralReviewThresholdLogit,
                Optional<Float> neuralRejectThresholdLogit
        ) {
            this(
                    reviewThreshold,
                    rejectThreshold,
                    neuralReviewThresholdLogit,
                    neuralRejectThresholdLogit,
                    NeuralDataQualityPolicy.unrestricted()
            );
        }

        public Snapshot {
            Objects.requireNonNull(neuralReviewThresholdLogit);
            Objects.requireNonNull(neuralRejectThresholdLogit);
            Objects.requireNonNull(neuralDataQualityPolicy);

            new RuntimeThresholdPolicy.Snapshot(
                    0L,
                    reviewThreshold,
                    rejectThreshold,
                    neuralReviewThresholdLogit,
                    neuralRejectThresholdLogit,
                    neuralDataQualityPolicy
            );
        }
    }

    public record VersionedSnapshot(
            long version,
            Snapshot settings
    ) {
    }

    public record AuditEntry(
            long version,
            Instant changedAt,
            String actorId,
            ChangeType type,
            Snapshot before,
            Snapshot after
    ) {
    }
}