package ru.finguard.starter.autoconfigure;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import ru.finguard.core.decision.NeuralDataQualityPolicy;
import ru.finguard.core.decision.RuntimeThresholdPolicy;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.neural.OnnxModelRegistry;
import ru.finguard.core.provider.TransactionDataProvider;
import ru.finguard.starter.config.FileModelPolicySnapshotStore;
import ru.finguard.starter.config.FinGuardModelManager;
import ru.finguard.starter.config.FinGuardSettingsManager;
import ru.finguard.starter.config.ModelPolicySnapshotStore;
import ru.finguard.starter.config.ModelStateStore;

import java.nio.file.Files;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.Optional;

import static org.junit.jupiter.api.Assertions.*;

class FinGuardPairedStateRestartTest {

    @TempDir
    Path directory;

    @Test
    void shouldRestorePairedModelAndPolicyAfterRestart() throws Exception {
        Path model = resource("finguard-v3/finguard-model.onnx");
        Path normalization = resource("finguard-v3/normalization.csv");

        var registry = new OnnxModelRegistry();

        var original = registry.register(
                "v3.0",
                model,
                normalization
        );

        var candidate = registry.register(
                "bank-v2",
                model,
                normalization
        );

        RuntimeThresholdPolicy.Snapshot initialPolicy = policy(
                0L,
                1.5145587f
        );

        RuntimeThresholdPolicy.Snapshot activePolicy = policy(
                1L,
                2.25f
        );

        var store = new FileModelPolicySnapshotStore(
                directory.resolve("model-policy-state.bin")
        );

        store.saveIfRevisionMatches(
                new ModelPolicySnapshotStore.State(
                        new ModelStateStore.State(
                                0L,
                                0L,
                                "v3.0",
                                null,
                                List.of(original),
                                List.of()
                        ),
                        initialPolicy,
                        null
                ),
                -1L
        );

        store.saveIfRevisionMatches(
                new ModelPolicySnapshotStore.State(
                        new ModelStateStore.State(
                                1L,
                                0L,
                                "v3.0",
                                null,
                                List.of(original, candidate),
                                List.of()
                        ),
                        initialPolicy,
                        null
                ),
                0L
        );

        var audit = new FinGuardModelManager.ModelAuditEntry(
                1L,
                java.time.Instant.parse("2026-11-03T00:00:00Z"),
                "bank-admin",
                FinGuardModelManager.ChangeType.ACTIVATE,
                "v3.0",
                "bank-v2",
                candidate.modelSha256(),
                candidate.normalizationSha256()
        );

        store.saveIfRevisionMatches(
                new ModelPolicySnapshotStore.State(
                        new ModelStateStore.State(
                                2L,
                                1L,
                                "bank-v2",
                                "v3.0",
                                List.of(original, candidate),
                                List.of(audit)
                        ),
                        activePolicy,
                        initialPolicy
                ),
                1L
        );

        verifyRestart(store, model, normalization);
        verifyRestart(store, model, normalization);
    }

    @Test
    void shouldPersistRegistrationWithoutSwitchingModel()
            throws Exception {
        Path model = resource("finguard-v3/finguard-model.onnx");
        Path normalization = resource("finguard-v3/normalization.csv");

        var store = new FileModelPolicySnapshotStore(
                directory.resolve("new-state.bin")
        );

        runner(store, model, normalization).run(context -> {
            assertNull(context.getStartupFailure());

            var manager = context.getBean(FinGuardModelManager.class);

            assertEquals("v3.0", manager.activeVersion());

            assertDoesNotThrow(() -> manager.register(
                    "bank-v1",
                    model,
                    normalization
            ));

            assertEquals("v3.0", manager.activeVersion());
        });

        var saved = store.load().orElseThrow();

        assertEquals(1L, saved.revision());
        assertEquals("v3.0", saved.modelState().activeVersion());
        assertEquals(2, saved.modelState().models().size());
        assertEquals(0L, saved.activePolicy().version());

        runner(store, model, normalization).run(context -> {
            assertNull(context.getStartupFailure());

            var manager = context.getBean(FinGuardModelManager.class);

            assertEquals("v3.0", manager.activeVersion());
            assertEquals(2, manager.versions().size());
            assertTrue(manager.find("bank-v1").isPresent());
        });
    }

    @Test
    void shouldActivateRollbackAndRestorePairedStateAcrossRestarts()
            throws Exception {
        Path model = resource("finguard-v3/finguard-model.onnx");
        Path profile = resource("finguard-v3/normalization.csv");

        var store = new FileModelPolicySnapshotStore(
                directory.resolve("activated-pair.bin")
        );

        runner(store, model, profile).run(context -> {
            assertNull(context.getStartupFailure());

            var manager = context.getBean(FinGuardModelManager.class);
            var settings = context.getBean(FinGuardSettingsManager.class);

            assertDoesNotThrow(
                    () -> manager.register(
                            "bank-v1",
                            model,
                            profile
                    )
            );

            var preview = assertDoesNotThrow(
                    () -> manager.previewActivationWithPolicy(
                            "bank-v1",
                            settings.runtimePolicy(),
                            2.25f,
                            null,
                            5
                    )
            );

            assertDoesNotThrow(
                    () -> manager.activateWithPolicy(
                            preview,
                            "bank-admin"
                    )
            );

            assertEquals("bank-v1", manager.activeVersion());
            assertEquals(1L, settings.snapshotWithVersion().version());

            assertEquals(
                    2.25f,
                    settings.runtimePolicy()
                            .snapshot()
                            .neuralReviewThresholdLogit()
                            .orElseThrow()
            );

            assertEquals(1L, manager.auditVersion());

            assertThrows(
                    IllegalStateException.class,
                    () -> settings.runtimePolicy().activate(
                            1L,
                            2L,
                            40,
                            80,
                            3.0f,
                            null
                    )
            );

            assertThrows(
                    IllegalStateException.class,
                    () -> manager.rollback(
                            "bank-v1",
                            "bank-admin"
                    )
            );
        });

        runner(store, model, profile).run(context -> {
            assertNull(context.getStartupFailure());

            var manager = context.getBean(FinGuardModelManager.class);
            var settings = context.getBean(FinGuardSettingsManager.class);

            assertEquals("bank-v1", manager.activeVersion());

            assertEquals(
                    2.25f,
                    settings.runtimePolicy()
                            .snapshot()
                            .neuralReviewThresholdLogit()
                            .orElseThrow()
            );

            assertDoesNotThrow(
                    () -> manager.rollbackWithPolicy(
                            "bank-v1",
                            "bank-admin"
                    )
            );

            assertEquals("v3.0", manager.activeVersion());
            assertEquals(2L, manager.auditVersion());
            assertEquals(2L, settings.snapshotWithVersion().version());
        });

        runner(store, model, profile).run(context -> {
            assertNull(context.getStartupFailure());

            var manager = context.getBean(FinGuardModelManager.class);
            var settings = context.getBean(FinGuardSettingsManager.class);

            assertEquals("v3.0", manager.activeVersion());

            assertEquals(
                    "bank-v1",
                    manager.previousVersion().orElseThrow()
            );

            assertEquals(
                    1.5145587f,
                    settings.runtimePolicy()
                            .snapshot()
                            .neuralReviewThresholdLogit()
                            .orElseThrow()
            );

            assertEquals(2L, settings.snapshotWithVersion().version());
            assertEquals(2L, manager.auditVersion());
        });
    }

    @Test
    void shouldKeepOldPairWhenPersistenceFailsBeforeSaving()
            throws Exception {
        Path model = resource("finguard-v3/finguard-model.onnx");
        Path profile = resource("finguard-v3/normalization.csv");

        var disk = new FileModelPolicySnapshotStore(
                directory.resolve("rejected-activation.bin")
        );

        var fail = new java.util.concurrent.atomic.AtomicBoolean();

        ModelPolicySnapshotStore store = new ModelPolicySnapshotStore() {
            @Override
            public Optional<State> load() throws java.io.IOException {
                return disk.load();
            }

            @Override
            public void saveIfRevisionMatches(
                    State next,
                    long revision
            ) throws java.io.IOException {
                if (fail.getAndSet(false)) {
                    throw new java.io.IOException(
                            "Simulated disk failure"
                    );
                }

                disk.saveIfRevisionMatches(next, revision);
            }
        };

        runner(store, model, profile).run(context -> {
            assertNull(context.getStartupFailure());

            var manager = context.getBean(FinGuardModelManager.class);
            var settings = context.getBean(FinGuardSettingsManager.class);

            assertDoesNotThrow(
                    () -> manager.register(
                            "bank-v1",
                            model,
                            profile
                    )
            );

            var preview = assertDoesNotThrow(
                    () -> manager.previewActivationWithPolicy(
                            "bank-v1",
                            settings.runtimePolicy(),
                            2.25f,
                            null,
                            5
                    )
            );

            fail.set(true);

            assertThrows(
                    java.io.IOException.class,
                    () -> manager.activateWithPolicy(
                            preview,
                            "bank-admin"
                    )
            );

            assertEquals("v3.0", manager.activeVersion());
            assertEquals(0L, settings.snapshotWithVersion().version());
            assertEquals(0L, manager.auditVersion());
        });

        assertEquals(
                "v3.0",
                disk.load().orElseThrow()
                        .modelState()
                        .activeVersion()
        );

        assertEquals(
                0L,
                disk.load().orElseThrow()
                        .activePolicy()
                        .version()
        );
    }

    @Test
    void shouldFailClosedIfDiskCommitSucceededButReportedFailure()
            throws Exception {
        Path model = resource("finguard-v3/finguard-model.onnx");
        Path profile = resource("finguard-v3/normalization.csv");

        var disk = new FileModelPolicySnapshotStore(
                directory.resolve("uncertain-activation.bin")
        );

        var failAfter = new java.util.concurrent.atomic.AtomicBoolean();

        ModelPolicySnapshotStore store = new ModelPolicySnapshotStore() {
            @Override
            public Optional<State> load() throws java.io.IOException {
                return disk.load();
            }

            @Override
            public void saveIfRevisionMatches(
                    State next,
                    long revision
            ) throws java.io.IOException {
                disk.saveIfRevisionMatches(next, revision);

                if (failAfter.getAndSet(false)) {
                    throw new java.io.IOException(
                            "Failure after disk commit"
                    );
                }
            }
        };

        runner(store, model, profile).run(context -> {
            assertNull(context.getStartupFailure());

            var manager = context.getBean(FinGuardModelManager.class);
            var settings = context.getBean(FinGuardSettingsManager.class);

            assertDoesNotThrow(
                    () -> manager.register(
                            "bank-v1",
                            model,
                            profile
                    )
            );

            var preview = assertDoesNotThrow(
                    () -> manager.previewActivationWithPolicy(
                            "bank-v1",
                            settings.runtimePolicy(),
                            2.25f,
                            null,
                            5
                    )
            );

            failAfter.set(true);

            assertThrows(
                    java.io.IOException.class,
                    () -> manager.activateWithPolicy(
                            preview,
                            "bank-admin"
                    )
            );

            assertThrows(
                    IllegalStateException.class,
                    manager::activeVersion
            );

            assertThrows(
                    IllegalStateException.class,
                    () -> manager.riskEngine().evaluate(
                            new ru.finguard.core.model.TransferData(
                                    101L,
                                    201L,
                                    301L,
                                    java.math.BigDecimal.valueOf(1000),
                                    "RUB",
                                    java.time.Instant.parse(
                                            "2026-11-03T00:00:00Z"
                                    )
                            )
                    )
            );
        });

        runner(disk, model, profile).run(context -> {
            assertNull(context.getStartupFailure());

            var manager = context.getBean(FinGuardModelManager.class);
            var settings = context.getBean(FinGuardSettingsManager.class);

            assertEquals("bank-v1", manager.activeVersion());
            assertEquals(1L, settings.snapshotWithVersion().version());

            assertEquals(
                    2.25f,
                    settings.runtimePolicy()
                            .snapshot()
                            .neuralReviewThresholdLogit()
                            .orElseThrow()
            );
        });
    }

    @Test
    void shouldRequireApprovedReleaseBeforeActivation() throws Exception {
        Path model = resource("finguard-v3/finguard-model.onnx");
        Path normalization = resource("finguard-v3/normalization.csv");

        Path release = directory.resolve("approved-release");
        Files.createDirectories(release);

        Path releaseModel = release.resolve("finguard-model.onnx");
        Path releaseNormalization = release.resolve("normalization.csv");
        Path qualityFile = release.resolve("quality.properties");

        Files.copy(model, releaseModel);
        Files.copy(normalization, releaseNormalization);

        Files.writeString(
                qualityFile,
                """
                        format=bank-v3-candidate-1
                        eligibleForManualReview=true
                        automaticActivation=false
                        fpBudgetPerThousand=5
                        reviewThresholdLogit=2.25
                        modelSha256=%s
                        normalizationSha256=%s
                        """.formatted(
                        sha256(releaseModel),
                        sha256(releaseNormalization)
                )
        );

        String approvedSha256 = sha256(qualityFile);
        String originalQuality = Files.readString(qualityFile);

        var store = new FileModelPolicySnapshotStore(
                directory.resolve("protected-activation.bin")
        );

        runner(store, model, normalization)
                .withPropertyValues(
                        "finguard.neural.require-approved-releases=true"
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    var manager = context.getBean(
                            FinGuardModelManager.class
                    );

                    var settings = context.getBean(
                            FinGuardSettingsManager.class
                    );

                    assertDoesNotThrow(() -> {
                        manager.register(
                                "bank-v1",
                                releaseModel,
                                releaseNormalization
                        );

                        String wrongSha256 = "0".repeat(64);

                        if (wrongSha256.equals(approvedSha256)) {
                            wrongSha256 = "1".repeat(64);
                        }

                        String invalidApproval = wrongSha256;

                        assertThrows(
                                SecurityException.class,
                                () -> manager.previewApprovedReleaseActivation(
                                        "bank-v1",
                                        settings.runtimePolicy(),
                                        qualityFile,
                                        invalidApproval
                                )
                        );

                        assertEquals("v3.0", manager.activeVersion());

                        var unapproved =
                                manager.previewActivationWithPolicy(
                                        "bank-v1",
                                        settings.runtimePolicy(),
                                        2.25f,
                                        null,
                                        5
                                );

                        assertThrows(
                                SecurityException.class,
                                () -> manager.activateWithPolicy(
                                        unapproved,
                                        "bank-admin"
                                )
                        );

                        assertEquals(
                                "v3.0",
                                manager.activeVersion()
                        );

                        var approved = manager.previewApprovedReleaseActivation(
                                "bank-v1",
                                settings.runtimePolicy(),
                                qualityFile,
                                approvedSha256
                        );

                        Files.writeString(
                                qualityFile,
                                originalQuality.replace(
                                        "reviewThresholdLogit=2.25",
                                        "reviewThresholdLogit=1.0"
                                )
                        );

                        assertThrows(
                                SecurityException.class,
                                () -> manager.activateWithPolicy(
                                        approved,
                                        "bank-admin"
                                )
                        );

                        assertEquals("v3.0", manager.activeVersion());
                        assertEquals(0L, settings.snapshotWithVersion().version());

                        Files.writeString(qualityFile, originalQuality);

                        manager.activateWithPolicy(
                                approved,
                                "bank-admin"
                        );

                        assertEquals(
                                "bank-v1",
                                manager.activeVersion()
                        );

                        assertEquals(
                                2.25f,
                                settings.runtimePolicy()
                                        .snapshot()
                                        .neuralReviewThresholdLogit()
                                        .orElseThrow()
                        );

                        assertEquals(
                                1L,
                                settings.snapshotWithVersion().version()
                        );
                    });
                });
    }

    private void verifyRestart(
            ModelPolicySnapshotStore store,
            Path model,
            Path normalization
    ) {
        runner(store, model, normalization).run(context -> {
            assertNull(context.getStartupFailure());

            var manager = context.getBean(FinGuardModelManager.class);
            var settings = context.getBean(FinGuardSettingsManager.class);

            assertEquals("bank-v2", manager.activeVersion());
            assertEquals("v3.0", manager.previousVersion().orElseThrow());
            assertEquals(1L, manager.auditVersion());

            assertEquals(1L, settings.snapshotWithVersion().version());

            assertEquals(
                    2.25f,
                    settings.runtimePolicy().snapshot()
                            .neuralReviewThresholdLogit()
                            .orElseThrow()
            );

            assertThrows(
                    IllegalStateException.class,
                    () -> settings.updateNeuralDecisionThresholds(
                            3.0f,
                            null,
                            1L,
                            "bank-admin"
                    )
            );

            assertThrows(
                    IllegalStateException.class,
                    () -> manager.activate(
                            "v3.0",
                            "bank-v2",
                            "bank-admin"
                    )
            );

            assertEquals("bank-v2", manager.activeVersion());
            assertEquals(1L, settings.snapshotWithVersion().version());
        });
    }

    private ApplicationContextRunner runner(
            ModelPolicySnapshotStore store,
            Path model,
            Path normalization
    ) {
        return new ApplicationContextRunner()
                .withConfiguration(
                        AutoConfigurations.of(
                                FinGuardAutoConfiguration.class
                        )
                )
                .withBean(
                        ModelPolicySnapshotStore.class,
                        () -> store
                )
                .withBean(
                        TransactionDataProvider.class,
                        () -> transfer -> TransactionHistorySnapshot.empty(
                                transfer.timestamp()
                        )
                )
                .withPropertyValues(
                        "finguard.neural.enabled=true",
                        "finguard.neural.versioning-enabled=true",
                        "finguard.neural.initial-version=v3.0",
                        "finguard.neural.model-path=" + model,
                        "finguard.neural.normalization-path=" + normalization
                );
    }

    private RuntimeThresholdPolicy.Snapshot policy(
            long version,
            float review
    ) {
        return new RuntimeThresholdPolicy.Snapshot(
                version,
                40,
                80,
                Optional.of(review),
                Optional.empty(),
                NeuralDataQualityPolicy.unrestricted()
        );
    }

    private Path resource(String name) throws Exception {
        return Path.of(Objects.requireNonNull(
                getClass().getClassLoader().getResource(name)
        ).toURI());
    }

    private static String sha256(Path file) throws Exception {
        byte[] bytes = Files.readAllBytes(file);

        return HexFormat.of().formatHex(
                MessageDigest.getInstance("SHA-256")
                        .digest(bytes)
        );
    }
}