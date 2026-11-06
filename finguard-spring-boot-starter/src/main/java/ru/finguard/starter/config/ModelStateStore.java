package ru.finguard.starter.config;

import ru.finguard.core.neural.OnnxModelRegistry;

import java.io.IOException;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

public interface ModelStateStore {

    Optional<State> load() throws IOException;

    void saveIfRevisionMatches(State next, long expectedRevision) throws IOException;

    record State(
            long revision,
            long auditVersion,
            String activeVersion,
            String previousVersion,
            List<OnnxModelRegistry.ModelVersion> models,
            List<FinGuardModelManager.ModelAuditEntry> auditHistory
    ) {
        public State {
            if (revision < 0 || auditVersion < 0) {
                throw new IllegalArgumentException("Negative model state revision");
            }
            Objects.requireNonNull(activeVersion);
            models = List.copyOf(models);
            auditHistory = List.copyOf(auditHistory);
            if (models.stream().noneMatch(model -> model.version().equals(activeVersion))) {
                throw new IllegalArgumentException("Active model is not registered");
            }
            if (previousVersion != null && models.stream().noneMatch(
                    model -> model.version().equals(previousVersion))) {
                throw new IllegalArgumentException("Previous model is not registered");
            }
        }
    }
}