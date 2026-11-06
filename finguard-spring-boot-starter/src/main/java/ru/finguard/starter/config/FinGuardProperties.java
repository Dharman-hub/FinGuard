package ru.finguard.starter.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "finguard")
public class FinGuardProperties {

    private int reviewThreshold = 40;
    private int rejectThreshold = 80;

    private final Rules rules = new Rules();
    private final Neural neural = new Neural();

    public int getReviewThreshold() {
        return reviewThreshold;
    }

    public void setReviewThreshold(int reviewThreshold) {
        this.reviewThreshold = reviewThreshold;
    }

    public int getRejectThreshold() {
        return rejectThreshold;
    }

    public void setRejectThreshold(int rejectThreshold) {
        this.rejectThreshold = rejectThreshold;
    }

    public Rules getRules() {
        return rules;
    }

    public Neural getNeural() {
        return neural;
    }

    public static class Neural {

        private boolean enabled;
        private String modelPath;
        private String normalizationPath;
        private boolean versioningEnabled;
        private String initialVersion = "v3.0";

        private NeuralReviewMode mode = NeuralReviewMode.CUSTOM;
        private float reviewThresholdLogit = 1.5145587f;
        private Float rejectThresholdLogit;
        private boolean customThresholdConfigured;

        public boolean isEnabled() {
            return enabled;
        }

        public void setEnabled(boolean enabled) {
            this.enabled = enabled;
        }

        public String getModelPath() {
            return modelPath;
        }

        public void setModelPath(String modelPath) {
            this.modelPath = modelPath;
        }

        public String getNormalizationPath() {
            return normalizationPath;
        }

        public void setNormalizationPath(String normalizationPath) {
            this.normalizationPath = normalizationPath;
        }

        public NeuralReviewMode getMode() {
            return mode;
        }

        public void setMode(NeuralReviewMode mode) {
            if (mode == null) {
                throw new IllegalArgumentException(
                        "Neural review mode is required"
                );
            }

            this.mode = mode;
        }

        public float getReviewThresholdLogit() {
            return reviewThresholdLogit;
        }

        public void setReviewThresholdLogit(float reviewThresholdLogit) {
            if (!Float.isFinite(reviewThresholdLogit)) {
                throw new IllegalArgumentException(
                        "Invalid neural review threshold"
                );
            }

            this.reviewThresholdLogit = reviewThresholdLogit;
            this.customThresholdConfigured = true;
        }

        public float resolveReviewThresholdLogit() {
            if (mode == NeuralReviewMode.CUSTOM) {
                return reviewThresholdLogit;
            }

            if (customThresholdConfigured) {
                throw new IllegalStateException(
                        "Configure either neural.mode or "
                                + "neural.review-threshold-logit, not both"
                );
            }

            return mode.thresholdLogit();
        }

        public boolean isVersioningEnabled() {
            return versioningEnabled;
        }

        public void setVersioningEnabled(boolean versioningEnabled) {
            this.versioningEnabled = versioningEnabled;
        }

        public String getInitialVersion() {
            return initialVersion;
        }

        public void setInitialVersion(String initialVersion) {
            if (initialVersion == null
                    || !initialVersion.matches("[A-Za-z0-9][A-Za-z0-9._-]{0,63}")) {
                throw new IllegalArgumentException(
                        "Invalid initial ONNX model version"
                );
            }

            this.initialVersion = initialVersion;
        }

        public Float getRejectThresholdLogit() {
            return rejectThresholdLogit;
        }

        public void setRejectThresholdLogit(Float rejectThresholdLogit) {
            if (rejectThresholdLogit != null
                    && !Float.isFinite(rejectThresholdLogit)) {
                throw new IllegalArgumentException(
                        "Invalid neural reject threshold"
                );
            }

            this.rejectThresholdLogit = rejectThresholdLogit;
        }
    }

    public static class Rules {

        private final RuleSettings unusualAmount = new RuleSettings(25);
        private final RuleSettings highVelocity = new RuleSettings(35);
        private final RuleSettings newRecipient = new RuleSettings(20);
        private final RuleSettings unusualActivity = new RuleSettings(20);

        public RuleSettings getUnusualAmount() {
            return unusualAmount;
        }

        public RuleSettings getHighVelocity() {
            return highVelocity;
        }

        public RuleSettings getNewRecipient() {
            return newRecipient;
        }

        public RuleSettings getUnusualActivity() {
            return unusualActivity;
        }
    }

    public static class RuleSettings {

        private boolean enabled = true;
        private int weight;

        public RuleSettings(int weight) {
            this.weight = weight;
        }

        public boolean isEnabled() {
            return enabled;
        }

        public void setEnabled(boolean enabled) {
            this.enabled = enabled;
        }

        public int getWeight() {
            return weight;
        }

        public void setWeight(int weight) {
            this.weight = weight;
        }
    }
}