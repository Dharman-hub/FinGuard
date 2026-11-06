package ru.finguard.demo.provider;

import org.springframework.stereotype.Component;
import ru.finguard.core.model.ActivityStatistics;
import ru.finguard.core.model.AmountStatistics;
import ru.finguard.core.model.RecipientStatistics;
import ru.finguard.core.model.RiskHistoryStatistics;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.model.VelocityStatistics;
import ru.finguard.core.provider.TransactionDataProvider;

import java.math.BigDecimal;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalLong;

@Component
public class DemoTransactionDataProvider implements TransactionDataProvider {

    @Override
    public TransactionHistorySnapshot loadSnapshot(TransferData transfer) {
        if (!"RUB".equals(transfer.currency())) {
            return TransactionHistorySnapshot.empty(transfer.timestamp());
        }

        if (transfer.userId() == 125L) {
            return createHistory(
                    transfer,
                    1,
                    2,
                    transfer.recipientId() == 458L ? 2 : 0,
                    0,
                    0.2
            );
        }

        if (transfer.userId() == 126L) {
            return createHistory(
                    transfer,
                    7,
                    25,
                    transfer.recipientId() == 777L ? 0 : 2,
                    3,
                    0.005
            );
        }

        return TransactionHistorySnapshot.empty(transfer.timestamp());
    }

    private TransactionHistorySnapshot createHistory(
            TransferData transfer,
            long transfers5m,
            long transfersHour,
            long completedToRecipient,
            long newRecipients24h,
            double hourShare
    ) {
        AmountStatistics amount = new AmountStatistics(
                Optional.of(new BigDecimal("5000.00")),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty()
        );

        VelocityStatistics velocity = new VelocityStatistics(
                OptionalLong.of(transfers5m),
                OptionalLong.of(transfersHour),
                OptionalLong.empty(),
                OptionalLong.empty(),
                OptionalLong.of(120),
                Optional.empty(),
                OptionalLong.empty(),
                OptionalLong.of(newRecipients24h),
                OptionalLong.empty(),
                OptionalLong.empty()
        );

        RecipientStatistics recipient = new RecipientStatistics(
                OptionalLong.of(completedToRecipient),
                Optional.empty(),
                OptionalLong.empty(),
                OptionalDouble.empty(),
                OptionalLong.empty()
        );

        ActivityStatistics activity = new ActivityStatistics(
                OptionalLong.of(120),
                OptionalLong.of(30),
                OptionalDouble.empty(),
                OptionalDouble.empty(),
                OptionalDouble.of(hourShare),
                OptionalDouble.empty(),
                OptionalLong.empty()
        );

        return new TransactionHistorySnapshot(
                amount,
                velocity,
                recipient,
                activity,
                RiskHistoryStatistics.empty(),
                transfer.timestamp()
        );
    }
}