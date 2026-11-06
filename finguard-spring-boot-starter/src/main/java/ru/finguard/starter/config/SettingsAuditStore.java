package ru.finguard.starter.config;

import java.util.Optional;

public interface SettingsAuditStore {

    long lastVersion();

    Optional<FinGuardSettingsManager.AuditEntry> latestEntry();

    Optional<FinGuardSettingsManager.AuditEntry> findEntry(long version);

    void appendIfVersionMatches(
            FinGuardSettingsManager.AuditEntry entry,
            long expectedVersion
    );
}