package ru.finguard.starter.autoconfigure;

import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import ru.finguard.core.api.RiskEngine;
import ru.finguard.core.decision.RuntimeThresholdPolicy;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.provider.TransactionDataProvider;
import ru.finguard.starter.config.FinGuardSettingsManager;

import java.math.BigDecimal;
import java.time.Instant;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FinGuardRuntimePolicyIntegrationTest {

    @Test
    void shouldUseOnePolicyVersionForEntireTransfer() {
        AtomicReference<FinGuardSettingsManager> managerReference =
                new AtomicReference<>();

        AtomicInteger calls = new AtomicInteger();

        new ApplicationContextRunner()
                .withConfiguration(
                        AutoConfigurations.of(
                                FinGuardAutoConfiguration.class
                        )
                )
                .withBean(
                        TransactionDataProvider.class,
                        () -> transfer -> {
                            if (calls.getAndIncrement() == 0) {
                                managerReference.get()
                                        .updateRuleThresholds(30, 90);
                            }

                            return TransactionHistorySnapshot.empty(
                                    transfer.timestamp()
                            );
                        }
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    FinGuardSettingsManager manager =
                            context.getBean(FinGuardSettingsManager.class);

                    RuntimeThresholdPolicy policy =
                            context.getBean(RuntimeThresholdPolicy.class);

                    RiskEngine engine = context.getBean(RiskEngine.class);

                    managerReference.set(manager);

                    assertSame(policy, manager.runtimePolicy());

                    var first = engine.evaluate(transfer());
                    var firstTrace = first.trace().orElseThrow();

                    assertEquals(0, firstTrace.policyVersion());
                    assertEquals(40, firstTrace.reviewThreshold());
                    assertEquals(80, firstTrace.rejectThreshold());

                    assertEquals(1, manager.snapshotWithVersion().version());

                    var second = engine.evaluate(transfer());
                    var secondTrace = second.trace().orElseThrow();

                    assertEquals(1, secondTrace.policyVersion());
                    assertEquals(30, secondTrace.reviewThreshold());
                    assertEquals(90, secondTrace.rejectThreshold());

                    assertFalse(
                            firstTrace.thresholdsFingerprint().equals(
                                    secondTrace.thresholdsFingerprint()
                            )
                    );

                    manager.rollbackToVersion(
                            0,
                            1,
                            "test-admin"
                    );

                    var third = engine.evaluate(transfer());
                    var thirdTrace = third.trace().orElseThrow();

                    assertEquals(2, thirdTrace.policyVersion());
                    assertEquals(40, thirdTrace.reviewThreshold());
                    assertEquals(80, thirdTrace.rejectThreshold());

                    assertEquals(
                            firstTrace.thresholdsFingerprint(),
                            thirdTrace.thresholdsFingerprint()
                    );

                    assertSame(engine, context.getBean(RiskEngine.class));
                });
    }

    @Test
    void shouldWorkWithoutNeuralModel() {
        new ApplicationContextRunner()
                .withConfiguration(
                        AutoConfigurations.of(
                                FinGuardAutoConfiguration.class
                        )
                )
                .withBean(
                        TransactionDataProvider.class,
                        () -> transfer -> TransactionHistorySnapshot.empty(
                                transfer.timestamp()
                        )
                )
                .run(context -> {
                    assertNull(context.getStartupFailure());

                    FinGuardSettingsManager manager =
                            context.getBean(FinGuardSettingsManager.class);

                    assertTrue(
                            manager.snapshot()
                                    .neuralReviewThresholdLogit()
                                    .isEmpty()
                    );

                    manager.updateRuleThresholds(20, 90);

                    var result = context.getBean(RiskEngine.class)
                            .evaluate(transfer());

                    assertEquals(
                            1,
                            result.trace().orElseThrow().policyVersion()
                    );

                    assertTrue(
                            result.trace().orElseThrow()
                                    .neuralReviewThresholdLogit()
                                    .isEmpty()
                    );
                });
    }

    private TransferData transfer() {
        return new TransferData(
                1001L,
                2001L,
                3001L,
                new BigDecimal("1000.00"),
                "EUR",
                Instant.parse("2026-10-30T12:00:00Z")
        );
    }
}