package ru.finguard.core.scoring;

import java.util.List;
import java.util.Objects;

public record RuleScore(
        int confirmedScore,
        int maximumScore,
        List<String> triggeredRules,
        List<String> missingDataRules
) {
    public RuleScore {
        if (confirmedScore < 0 || confirmedScore > 100) {
            throw new IllegalArgumentException("Confirmed score must be between 0 and 100");
        }

        if (maximumScore < confirmedScore || maximumScore > 100) {
            throw new IllegalArgumentException("Invalid maximum risk score");
        }

        triggeredRules = List.copyOf(Objects.requireNonNull(triggeredRules));
        missingDataRules = List.copyOf(Objects.requireNonNull(missingDataRules));
    }

    public boolean hasMissingData() {
        return !missingDataRules.isEmpty();
    }
}
