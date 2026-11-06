package ru.finguard.starter.config;

import ru.finguard.core.decision.RuntimeThresholdPolicy;

import java.io.IOException;
import java.util.Objects;
import java.util.Optional;

public final class PairedModelStateStoreAdapter implements ModelStateStore {

    private final ModelPolicySnapshotStore pairedStore;
    private final RuntimeThresholdPolicy runtimePolicy;

    public PairedModelStateStoreAdapter(
            ModelPolicySnapshotStore pairedStore,
            RuntimeThresholdPolicy runtimePolicy
    ) {
        this.pairedStore = Objects.requireNonNull(pairedStore);
        this.runtimePolicy = Objects.requireNonNull(runtimePolicy);
    }

    @Override
    public Optional<State> load() throws IOException {
        Optional<ModelPolicySnapshotStore.State> saved =
                pairedStore.load();

        saved.ifPresent(this::verifyPolicy);

        return saved.map(
                ModelPolicySnapshotStore.State::modelState
        );
    }

    @Override
    public void saveIfRevisionMatches(
            State next,
            long expectedRevision
    ) throws IOException {
        Objects.requireNonNull(next);

        Optional<ModelPolicySnapshotStore.State> saved =
                pairedStore.load();

        if (saved.isEmpty()) {
            if (expectedRevision != -1L
                    || next.revision() != 0L
                    || next.auditVersion() != 0L
                    || next.previousVersion() != null
                    || !next.auditHistory().isEmpty()) {
                throw new IllegalStateException(
                        "Only initial registration is allowed without paired state"
                );
            }

            pairedStore.saveIfRevisionMatches(
                    new ModelPolicySnapshotStore.State(
                            next,
                            runtimePolicy.snapshot(),
                            null
                    ),
                    -1L
            );

            return;
        }

        ModelPolicySnapshotStore.State before =
                saved.orElseThrow();

        verifyPolicy(before);

        ModelStateStore.State original =
                before.modelState();

        if (!next.activeVersion().equals(original.activeVersion())
                || !Objects.equals(
                next.previousVersion(),
                original.previousVersion()
        )
                || next.auditVersion() != original.auditVersion()
                || !next.auditHistory().equals(original.auditHistory())
                || next.models().size() != original.models().size() + 1
                || !next.models()
                .subList(0, original.models().size())
                .equals(original.models())) {
            throw new IllegalStateException(
                    "Model activation requires a paired transaction"
            );
        }

        pairedStore.saveIfRevisionMatches(
                new ModelPolicySnapshotStore.State(
                        next,
                        before.activePolicy(),
                        before.previousPolicy()
                ),
                expectedRevision
        );
    }

    ModelPolicySnapshotStore.State loadPaired() throws IOException {
        return pairedStore.load().orElseThrow(
                () -> new IllegalStateException(
                        "Paired state is not initialized"
                )
        );
    }

    void savePaired(
            ModelPolicySnapshotStore.State next,
            long expectedRevision
    ) throws IOException {
        pairedStore.saveIfRevisionMatches(
                next,
                expectedRevision
        );
    }

    private void verifyPolicy(
            ModelPolicySnapshotStore.State saved
    ) {
        if (!saved.activePolicy().equals(runtimePolicy.snapshot())) {
            throw new IllegalStateException(
                    "Runtime thresholds differ from paired persistent state"
            );
        }
    }
}