package ru.finguard.starter.autoconfigure;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import ru.finguard.core.api.RiskEngine;
import ru.finguard.core.model.RiskEvaluationTrace;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.provider.TransactionDataProvider;
import ru.finguard.starter.config.FileModelPolicySnapshotStore;
import ru.finguard.starter.config.FinGuardModelManager;
import ru.finguard.starter.config.FinGuardSettingsManager;
import ru.finguard.starter.config.ModelPolicySnapshotStore;

import java.io.IOException;
import java.math.BigDecimal;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.ConcurrentModificationException;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicBoolean;

import static org.junit.jupiter.api.Assertions.*;

class FinGuardPairedConcurrencyTest {

    @TempDir
    Path directory;

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void shouldBlockTransfersBetweenDiskCommitAndModelInstallation()
            throws Exception {
        Path model = resource("finguard-v3/finguard-model.onnx");
        Path normalization = resource("finguard-v3/normalization.csv");

        var disk = new FileModelPolicySnapshotStore(
                directory.resolve("blocking.bin")
        );

        var pause = new AtomicBoolean();
        var persisted = new CountDownLatch(1);
        var continueActivation = new CountDownLatch(1);

        ModelPolicySnapshotStore store = new ModelPolicySnapshotStore() {
            @Override
            public Optional<State> load() throws IOException {
                return disk.load();
            }

            @Override
            public void saveIfRevisionMatches(
                    State next,
                    long revision
            ) throws IOException {
                disk.saveIfRevisionMatches(next, revision);

                if (pause.compareAndSet(true, false)) {
                    persisted.countDown();

                    try {
                        if (!continueActivation.await(
                                10,
                                TimeUnit.SECONDS
                        )) {
                            throw new IOException(
                                    "Activation was not released"
                            );
                        }
                    } catch (InterruptedException exception) {
                        Thread.currentThread().interrupt();

                        throw new IOException(
                                "Activation interrupted",
                                exception
                        );
                    }
                }
            }
        };

        try {
            runner(store, model, normalization).run(context -> {
                assertNull(context.getStartupFailure());

                var manager = context.getBean(
                        FinGuardModelManager.class
                );

                var settings = context.getBean(
                        FinGuardSettingsManager.class
                );

                var riskEngine = context.getBean(RiskEngine.class);

                assertDoesNotThrow(() -> {
                    manager.register(
                            "bank-v1",
                            model,
                            normalization
                    );

                    var preview = manager.previewActivationWithPolicy(
                            "bank-v1",
                            settings.runtimePolicy(),
                            2.25f,
                            null,
                            5
                    );

                    try (var executor = Executors.newFixedThreadPool(2)) {
                        pause.set(true);

                        var activation = executor.submit(() -> {
                            manager.activateWithPolicy(
                                    preview,
                                    "bank-admin"
                            );

                            return null;
                        });

                        assertTrue(
                                persisted.await(10, TimeUnit.SECONDS)
                        );

                        assertEquals(
                                "bank-v1",
                                disk.load().orElseThrow()
                                        .modelState()
                                        .activeVersion()
                        );

                        var evaluationStarted = new CountDownLatch(1);

                        var evaluation = executor.submit(() -> {
                            evaluationStarted.countDown();

                            return riskEngine.evaluate(transfer())
                                    .trace()
                                    .orElseThrow();
                        });

                        assertTrue(
                                evaluationStarted.await(
                                        5,
                                        TimeUnit.SECONDS
                                )
                        );

                        assertThrows(
                                TimeoutException.class,
                                () -> evaluation.get(
                                        250,
                                        TimeUnit.MILLISECONDS
                                )
                        );

                        continueActivation.countDown();

                        activation.get(10, TimeUnit.SECONDS);

                        var trace = evaluation.get(
                                10,
                                TimeUnit.SECONDS
                        );

                        assertEquals(1L, trace.policyVersion());

                        assertEquals(
                                2.25f,
                                trace.neuralReviewThresholdLogit()
                                        .orElseThrow()
                        );

                        assertEquals(
                                "bank-v1",
                                manager.activeVersion()
                        );
                    } finally {
                        continueActivation.countDown();
                    }
                });
            });
        } finally {
            continueActivation.countDown();
        }
    }

    @Test
    @Timeout(value = 30, unit = TimeUnit.SECONDS)
    void shouldAllowOnlyOneActivationFromConcurrentPreviews()
            throws Exception {
        Path model = resource("finguard-v3/finguard-model.onnx");
        Path normalization = resource("finguard-v3/normalization.csv");

        var store = new FileModelPolicySnapshotStore(
                directory.resolve("competition.bin")
        );

        runner(store, model, normalization).run(context -> {
            assertNull(context.getStartupFailure());

            var manager = context.getBean(
                    FinGuardModelManager.class
            );

            var settings = context.getBean(
                    FinGuardSettingsManager.class
            );

            assertDoesNotThrow(() -> {
                manager.register("bank-v1", model, normalization);
                manager.register("bank-v2", model, normalization);

                var first = manager.previewActivationWithPolicy(
                        "bank-v1",
                        settings.runtimePolicy(),
                        2.25f,
                        null,
                        5
                );

                var second = manager.previewActivationWithPolicy(
                        "bank-v2",
                        settings.runtimePolicy(),
                        3.75f,
                        null,
                        5
                );

                var start = new CountDownLatch(1);

                try (var executor = Executors.newFixedThreadPool(2)) {
                    var firstResult = executor.submit(() -> {
                        start.await();

                        try {
                            manager.activateWithPolicy(
                                    first,
                                    "admin-1"
                            );

                            return true;
                        } catch (ConcurrentModificationException expected) {
                            return false;
                        }
                    });

                    var secondResult = executor.submit(() -> {
                        start.await();

                        try {
                            manager.activateWithPolicy(
                                    second,
                                    "admin-2"
                            );

                            return true;
                        } catch (ConcurrentModificationException expected) {
                            return false;
                        }
                    });

                    start.countDown();

                    boolean firstWon = firstResult.get(
                            10,
                            TimeUnit.SECONDS
                    );

                    boolean secondWon = secondResult.get(
                            10,
                            TimeUnit.SECONDS
                    );

                    assertNotEquals(firstWon, secondWon);

                    assertEquals(
                            firstWon ? "bank-v1" : "bank-v2",
                            manager.activeVersion()
                    );

                    assertEquals(
                            firstWon ? 2.25f : 3.75f,
                            settings.runtimePolicy()
                                    .snapshot()
                                    .neuralReviewThresholdLogit()
                                    .orElseThrow()
                    );

                    assertEquals(1L, manager.auditVersion());

                    assertEquals(
                            1L,
                            settings.snapshotWithVersion().version()
                    );

                    assertEquals(
                            3L,
                            store.load().orElseThrow().revision()
                    );

                    assertEquals(
                            manager.activeVersion(),
                            store.load().orElseThrow()
                                    .modelState()
                                    .activeVersion()
                    );
                } finally {
                    start.countDown();
                }
            });
        });
    }

    @Test
    @Timeout(value = 45, unit = TimeUnit.SECONDS)
    void shouldNeverReturnMixedThresholdsDuringRepeatedSwitches()
            throws Exception {
        Path model = resource("finguard-v3/finguard-model.onnx");
        Path normalization = resource("finguard-v3/normalization.csv");

        var store = new FileModelPolicySnapshotStore(
                directory.resolve("stress.bin")
        );

        runner(store, model, normalization).run(context -> {
            assertNull(context.getStartupFailure());

            var manager = context.getBean(
                    FinGuardModelManager.class
            );

            var settings = context.getBean(
                    FinGuardSettingsManager.class
            );

            var engine = context.getBean(RiskEngine.class);

            assertDoesNotThrow(() -> {
                manager.register("bank-v1", model, normalization);

                var start = new CountDownLatch(1);

                try (var executor = Executors.newFixedThreadPool(5)) {
                    List<java.util.concurrent.Future<List<RiskEvaluationTrace>>>
                            readers = new ArrayList<>();

                    for (int thread = 0; thread < 4; thread++) {
                        readers.add(executor.submit(() -> {
                            start.await();

                            List<RiskEvaluationTrace> traces =
                                    new ArrayList<>();

                            for (int i = 0; i < 60; i++) {
                                traces.add(
                                        engine.evaluate(transfer())
                                                .trace()
                                                .orElseThrow()
                                );
                            }

                            return traces;
                        }));
                    }

                    var switching = executor.submit(() -> {
                        start.await();

                        for (int i = 0; i < 3; i++) {
                            var preview =
                                    manager.previewActivationWithPolicy(
                                            "bank-v1",
                                            settings.runtimePolicy(),
                                            2.25f,
                                            null,
                                            5
                                    );

                            manager.activateWithPolicy(
                                    preview,
                                    "bank-admin"
                            );

                            manager.rollbackWithPolicy(
                                    "bank-v1",
                                    "bank-admin"
                            );
                        }

                        return null;
                    });

                    start.countDown();

                    switching.get(30, TimeUnit.SECONDS);

                    int checked = 0;

                    for (var reader : readers) {
                        for (var trace : reader.get(
                                30,
                                TimeUnit.SECONDS
                        )) {
                            float expected =
                                    trace.policyVersion() % 2 == 0
                                            ? 1.5145587f
                                            : 2.25f;

                            assertEquals(
                                    expected,
                                    trace.neuralReviewThresholdLogit()
                                            .orElseThrow()
                            );

                            checked++;
                        }
                    }

                    assertEquals(240, checked);
                    assertEquals("v3.0", manager.activeVersion());
                    assertEquals(6L, manager.auditVersion());

                    assertEquals(
                            6L,
                            settings.snapshotWithVersion().version()
                    );

                    assertEquals(
                            6L,
                            store.load().orElseThrow()
                                    .activePolicy()
                                    .version()
                    );
                } finally {
                    start.countDown();
                }
            });
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

    private TransferData transfer() {
        return new TransferData(
                101L,
                201L,
                301L,
                BigDecimal.valueOf(1000),
                "RUB",
                Instant.parse("2026-11-03T00:00:00Z")
        );
    }

    private Path resource(String name) throws Exception {
        return Path.of(Objects.requireNonNull(
                getClass().getClassLoader().getResource(name)
        ).toURI());
    }
}