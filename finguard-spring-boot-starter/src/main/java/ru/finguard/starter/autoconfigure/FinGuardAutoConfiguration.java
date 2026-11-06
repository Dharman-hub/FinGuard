package ru.finguard.starter.autoconfigure;

import ai.onnxruntime.OrtException;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.core.env.Environment;
import org.springframework.context.annotation.Bean;

import ru.finguard.core.api.RiskEngine;
import ru.finguard.core.api.RiskEngineBuilder;
import ru.finguard.core.decision.RuntimeThresholdPolicy;
import ru.finguard.core.feature.V3InputNormalizer;
import ru.finguard.core.neural.NeuralPredictor;
import ru.finguard.core.neural.OnnxNeuralPredictor;
import ru.finguard.core.neural.V3NeuralRiskEvaluator;
import ru.finguard.core.provider.ConfirmedFraudProvider;
import ru.finguard.core.provider.SupplierIbanDataProvider;
import ru.finguard.core.provider.TransactionDataProvider;
import ru.finguard.starter.config.FinGuardProperties;
import ru.finguard.starter.config.FinGuardProperties.RuleSettings;
import ru.finguard.starter.config.FinGuardSettingsManager;
import ru.finguard.starter.config.SettingsAuditStore;
import ru.finguard.starter.config.ThresholdPolicyChangeService;
import ru.finguard.core.monitoring.MonitoringRiskEngine;
import ru.finguard.core.monitoring.RiskMetrics;
import ru.finguard.core.monitoring.RiskMetricsPublisher;
import ru.finguard.core.monitoring.RiskMetricsSink;
import ru.finguard.core.neural.OnnxModelRegistry;
import ru.finguard.core.neural.VersionedOnnxRiskEngine;
import ru.finguard.starter.config.FinGuardModelManager;
import ru.finguard.starter.config.ModelStateStore;
import ru.finguard.starter.config.ModelPolicySnapshotStore;
import ru.finguard.starter.config.PairedModelStateStoreAdapter;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

@AutoConfiguration
@EnableConfigurationProperties(FinGuardProperties.class)
@ConditionalOnProperty(
        prefix = "finguard",
        name = "enabled",
        havingValue = "true",
        matchIfMissing = true
)
public class FinGuardAutoConfiguration {

    @Bean(destroyMethod = "close")
    @ConditionalOnProperty(
            prefix = "finguard.neural",
            name = "enabled",
            havingValue = "true"
    )
    @ConditionalOnProperty(
            prefix = "finguard.neural",
            name = "versioning-enabled",
            havingValue = "false",
            matchIfMissing = true
    )
    @ConditionalOnMissingBean(NeuralPredictor.class)
    public OnnxNeuralPredictor onnxNeuralPredictor(
            FinGuardProperties properties
    ) throws OrtException {
        Path path = requiredPath(
                properties.getNeural().getModelPath(),
                "finguard.neural.model-path"
        );

        return new OnnxNeuralPredictor(path);
    }

    @Bean
    @ConditionalOnProperty(
            prefix = "finguard.neural",
            name = "enabled",
            havingValue = "true"
    )
    @ConditionalOnProperty(
            prefix = "finguard.neural",
            name = "versioning-enabled",
            havingValue = "false",
            matchIfMissing = true
    )
    @ConditionalOnMissingBean(V3InputNormalizer.class)
    public V3InputNormalizer v3InputNormalizer(
            FinGuardProperties properties
    ) throws IOException {
        Path path = requiredPath(
                properties.getNeural().getNormalizationPath(),
                "finguard.neural.normalization-path"
        );

        return V3InputNormalizer.load(path);
    }

    @Bean
    @ConditionalOnProperty(
            prefix = "finguard.neural",
            name = "enabled",
            havingValue = "true"
    )
    @ConditionalOnProperty(
            prefix = "finguard.neural",
            name = "versioning-enabled",
            havingValue = "false",
            matchIfMissing = true
    )
    @ConditionalOnMissingBean(V3NeuralRiskEvaluator.class)
    public V3NeuralRiskEvaluator v3NeuralRiskEvaluator(
            V3InputNormalizer normalizer,
            NeuralPredictor predictor
    ) {
        return new V3NeuralRiskEvaluator(
                normalizer,
                predictor
        );
    }

    @Bean
    @ConditionalOnBean(TransactionDataProvider.class)
    @ConditionalOnMissingBean(RuntimeThresholdPolicy.class)
    public RuntimeThresholdPolicy runtimeThresholdPolicy(
            FinGuardProperties properties,
            ObjectProvider<ModelPolicySnapshotStore> pairedStoreProvider
    ) throws IOException {
        ModelPolicySnapshotStore pairedStore =
                pairedStoreProvider.getIfAvailable();

        if (pairedStore != null) {
            if (!properties.getNeural().isEnabled()
                    || !properties.getNeural().isVersioningEnabled()) {
                throw new IllegalStateException(
                        "Paired storage requires enabled neural versioning"
                );
            }

            var saved = pairedStore.load();

            if (saved.isPresent()) {
                RuntimeThresholdPolicy.Snapshot policy =
                        saved.orElseThrow().activePolicy();

                if (policy.neuralReviewThresholdLogit().isEmpty()) {
                    throw new IllegalStateException(
                            "Restored neural policy is disabled"
                    );
                }

                return new RuntimeThresholdPolicy(
                        policy.version(),
                        policy.reviewThreshold(),
                        policy.rejectThreshold(),
                        policy.neuralReviewThresholdLogit().orElseThrow(),
                        policy.neuralRejectThresholdLogit().orElse(null),
                        policy.neuralDataQualityPolicy()
                );
            }
        }

        Float review = properties.getNeural().isEnabled()
                ? properties.getNeural().resolveReviewThresholdLogit()
                : null;

        Float reject = properties.getNeural().getRejectThresholdLogit();

        if (reject != null && !properties.getNeural().isEnabled()) {
            throw new IllegalStateException(
                    "Neural rejection requires neural evaluation"
            );
        }

        return new RuntimeThresholdPolicy(
                0L,
                properties.getReviewThreshold(),
                properties.getRejectThreshold(),
                review,
                reject
        );
    }

    @Bean
    @ConditionalOnBean(TransactionDataProvider.class)
    @ConditionalOnMissingBean(FinGuardSettingsManager.class)
    public FinGuardSettingsManager finGuardSettingsManager(
            RuntimeThresholdPolicy runtimePolicy,
            ObjectProvider<SettingsAuditStore> auditStoreProvider,
            ObjectProvider<ModelPolicySnapshotStore> pairedStoreProvider
    ) {
        SettingsAuditStore auditStore =
                auditStoreProvider.getIfAvailable();

        ModelPolicySnapshotStore pairedStore =
                pairedStoreProvider.getIfAvailable();

        if (pairedStore != null && auditStore != null) {
            throw new IllegalStateException(
                    "Independent settings audit store is incompatible with paired storage"
            );
        }

        FinGuardSettingsManager manager =
                new FinGuardSettingsManager(runtimePolicy, auditStore);

        if (pairedStore != null) {
            manager.restrictToPairedState();
        }

        return manager;
    }

    @Bean
    @ConditionalOnBean(TransactionDataProvider.class)
    @ConditionalOnMissingBean(RiskMetrics.class)
    @ConditionalOnProperty(
            prefix = "finguard.monitoring",
            name = "enabled",
            havingValue = "true",
            matchIfMissing = true
    )
    public RiskMetrics riskMetrics() {
        return new RiskMetrics();
    }

    @Bean
    @ConditionalOnBean({
            TransactionDataProvider.class,
            RiskMetricsSink.class
    })
    @ConditionalOnMissingBean(RiskMetricsPublisher.class)
    @ConditionalOnProperty(
            prefix = "finguard.monitoring",
            name = "enabled",
            havingValue = "true",
            matchIfMissing = true
    )
    public RiskMetricsPublisher riskMetricsPublisher(
            RiskMetrics metrics,
            RiskMetricsSink sink
    ) {
        return new RiskMetricsPublisher(metrics, sink);
    }

    @Bean
    @ConditionalOnBean(FinGuardSettingsManager.class)
    @ConditionalOnMissingBean(ThresholdPolicyChangeService.class)
    public ThresholdPolicyChangeService thresholdPolicyChangeService(
            FinGuardSettingsManager settingsManager,
            ObjectProvider<SettingsAuditStore> auditStoreProvider
    ) {
        return new ThresholdPolicyChangeService(
                settingsManager,
                auditStoreProvider.getIfAvailable()
        );
    }

    @Bean(destroyMethod = "close")
    @ConditionalOnBean(TransactionDataProvider.class)
    @ConditionalOnMissingBean(FinGuardModelManager.class)
    @ConditionalOnProperty(
            prefix = "finguard.neural",
            name = "versioning-enabled",
            havingValue = "true"
    )
    public FinGuardModelManager finGuardModelManager(
            TransactionDataProvider transactionDataProvider,
            FinGuardProperties properties,
            FinGuardSettingsManager settingsManager,
            ObjectProvider<SupplierIbanDataProvider> ibanProvider,
            ObjectProvider<ConfirmedFraudProvider> confirmationProvider,
            ObjectProvider<ModelStateStore> stateStoreProvider,
            ObjectProvider<ModelPolicySnapshotStore> pairedStoreProvider,
            Environment environment
    ) throws IOException, OrtException {
        if (!properties.getNeural().isEnabled()) {
            throw new IllegalStateException(
                    "ONNX versioning requires neural evaluation"
            );
        }

        ModelPolicySnapshotStore pairedStore =
                pairedStoreProvider.getIfAvailable();

        ModelStateStore independentStore =
                stateStoreProvider.getIfAvailable();

        if (pairedStore != null && independentStore != null) {
            throw new IllegalStateException(
                    "Cannot use independent model storage with paired storage"
            );
        }

        if (pairedStore != null) {
            settingsManager.restrictToPairedState();
        }

        ModelStateStore store = pairedStore == null
                ? independentStore
                : new PairedModelStateStoreAdapter(
                pairedStore,
                settingsManager.runtimePolicy()
        );

        ModelStateStore.State restored = store == null
                ? null
                : store.load().orElse(null);

        OnnxModelRegistry registry = new OnnxModelRegistry();

        String startVersion;
        String previousVersion = null;

        if (restored == null) {
            startVersion = properties.getNeural().getInitialVersion();

            registry.register(
                    startVersion,
                    requiredPath(
                            properties.getNeural().getModelPath(),
                            "finguard.neural.model-path"
                    ),
                    requiredPath(
                            properties.getNeural().getNormalizationPath(),
                            "finguard.neural.normalization-path"
                    )
            );
        } else {
            for (OnnxModelRegistry.ModelVersion model : restored.models()) {
                OnnxModelRegistry.ModelVersion checked = registry.register(
                        model.version(),
                        model.modelPath(),
                        model.normalizationPath()
                );

                if (!checked.equals(model)) {
                    throw new IllegalStateException(
                            "Stored ONNX model files changed: " + model.version()
                    );
                }
            }

            startVersion = restored.activeVersion();
            previousVersion = restored.previousVersion();
        }

        VersionedOnnxRiskEngine engine = new VersionedOnnxRiskEngine(
                registry,
                startVersion,
                previousVersion,
                (evaluator, modelSha256) -> buildEngine(
                        transactionDataProvider,
                        properties,
                        settingsManager,
                        confirmationProvider.getIfAvailable(),
                        ibanProvider.getIfAvailable(),
                        evaluator,
                        modelSha256
                )
        );

        try {
            FinGuardModelManager manager =
                    new FinGuardModelManager(
                            registry,
                            engine,
                            store,
                            settingsManager
                    );

            if (environment.getProperty(
                    "finguard.neural.require-approved-releases",
                    Boolean.class,
                    false
            )) {
                manager.requireApprovedReleases();
            }

            return manager;
        } catch (RuntimeException exception) {
            engine.close();
            throw exception;
        }
    }

    @Bean
    @ConditionalOnBean(TransactionDataProvider.class)
    @ConditionalOnMissingBean(RiskEngine.class)
    public RiskEngine riskEngine(
            TransactionDataProvider transactionDataProvider,
            FinGuardProperties properties,
            FinGuardSettingsManager settingsManager,
            ObjectProvider<V3NeuralRiskEvaluator> neuralProvider,
            ObjectProvider<OnnxNeuralPredictor> onnxPredictorProvider,
            ObjectProvider<SupplierIbanDataProvider> ibanProvider,
            ObjectProvider<ConfirmedFraudProvider> confirmationProvider,
            ObjectProvider<RiskMetrics> metricsProvider,
            ObjectProvider<FinGuardModelManager> modelManagerProvider
    ) throws IOException {
        FinGuardModelManager modelManager =
                modelManagerProvider.getIfAvailable();

        RiskEngine engine;

        if (modelManager != null) {
            engine = modelManager.riskEngine();
        } else {
            V3NeuralRiskEvaluator evaluator =
                    properties.getNeural().isEnabled()
                            ? neuralProvider.getObject()
                            : null;

            String modelSha256 = null;

            if (evaluator != null
                    && onnxPredictorProvider.getIfAvailable() != null) {
                Path modelPath = requiredPath(
                        properties.getNeural().getModelPath(),
                        "finguard.neural.model-path"
                );

                modelSha256 = calculateSha256(modelPath);
            }

            engine = buildEngine(
                    transactionDataProvider,
                    properties,
                    settingsManager,
                    confirmationProvider.getIfAvailable(),
                    ibanProvider.getIfAvailable(),
                    evaluator,
                    modelSha256
            );
        }

        RiskMetrics metrics = metricsProvider.getIfAvailable();

        return metrics == null
                ? engine
                : new MonitoringRiskEngine(engine, metrics);
    }

    private RiskEngine buildEngine(
            TransactionDataProvider transactionDataProvider,
            FinGuardProperties properties,
            FinGuardSettingsManager settingsManager,
            ConfirmedFraudProvider confirmedProvider,
            SupplierIbanDataProvider ibanProvider,
            V3NeuralRiskEvaluator evaluator,
            String modelSha256
    ) {
        RiskEngineBuilder builder = new RiskEngineBuilder(
                transactionDataProvider
        ).runtimeThresholdPolicy(
                settingsManager.runtimePolicy()
        );

        FinGuardProperties.Rules rules = properties.getRules();

        configureRule(builder, "UNUSUAL_AMOUNT", rules.getUnusualAmount());
        configureRule(builder, "HIGH_VELOCITY", rules.getHighVelocity());
        configureRule(builder, "NEW_RECIPIENT", rules.getNewRecipient());
        configureRule(builder, "UNUSUAL_ACTIVITY", rules.getUnusualActivity());

        if (confirmedProvider != null) {
            builder.confirmedFraudProvider(confirmedProvider);
        }

        if (evaluator != null) {
            builder.neuralReview(
                    evaluator,
                    settingsManager.runtimePolicy().snapshot()
                            .neuralReviewThresholdLogit().orElseThrow()
            );

            if (ibanProvider != null) {
                builder.supplierIbanDataProvider(ibanProvider);
            }

            if (modelSha256 != null) {
                builder.modelSha256(modelSha256);
            }
        }

        return builder.build();
    }

    private static String calculateSha256(Path path) throws IOException {
        MessageDigest digest;

        try {
            digest = MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException(
                    "SHA-256 is unavailable",
                    exception
            );
        }

        try (InputStream input = Files.newInputStream(path)) {
            byte[] buffer = new byte[8192];
            int count;

            while ((count = input.read(buffer)) != -1) {
                digest.update(buffer, 0, count);
            }
        }

        return HexFormat.of().formatHex(digest.digest());
    }

    private static Path requiredPath(
            String value,
            String propertyName
    ) {
        if (value == null || value.isBlank()) {
            throw new IllegalStateException(
                    "Missing FinGuard property: " + propertyName
            );
        }

        return Path.of(value);
    }

    private void configureRule(
            RiskEngineBuilder builder,
            String code,
            RuleSettings settings
    ) {
        if (settings.isEnabled()) {
            builder.weight(code, settings.getWeight());
        } else {
            builder.removeRule(code);
        }
    }
}