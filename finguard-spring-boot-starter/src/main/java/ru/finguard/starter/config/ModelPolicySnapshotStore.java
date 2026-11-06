package ru.finguard.starter.config;

import ru.finguard.core.decision.RuntimeThresholdPolicy;

import java.io.IOException;
import java.util.Objects;
import java.util.Optional;

public interface ModelPolicySnapshotStore {
    Optional<State> load() throws IOException;

    void saveIfRevisionMatches(State next, long expectedRevision) throws IOException;

    record State(
            ModelStateStore.State modelState,
            RuntimeThresholdPolicy.Snapshot activePolicy,
            RuntimeThresholdPolicy.Snapshot previousPolicy
    ) {
        public State {
            Objects.requireNonNull(modelState);
            Objects.requireNonNull(activePolicy);

            if (previousPolicy != null
                    && modelState.previousVersion() == null) {
                throw new IllegalArgumentException(
                        "Previous model is required for previous policy"
                );
            }
        }

        public long revision() {
            return modelState.revision();
        }
    }
}