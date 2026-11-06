package ru.finguard.training.dataset.synsepa;

import ru.finguard.core.model.ActivityStatistics;
import ru.finguard.core.model.AmountStatistics;
import ru.finguard.core.model.RecipientStatistics;
import ru.finguard.core.model.RiskHistoryStatistics;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.model.VelocityStatistics;
import ru.finguard.training.dataset.synsepa.SynSepChronologicalReader.Transaction;

import java.math.BigDecimal;
import java.math.MathContext;
import java.time.DayOfWeek;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.NavigableMap;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalDouble;
import java.util.OptionalLong;
import java.util.Set;
import java.util.TreeMap;
import java.util.function.Consumer;

public final class SynSepHistoryBuilder {

    private SynSepHistoryBuilder() {
    }

    public static void forEachTransaction(
            List<Transaction> transactions,
            Consumer<PreparedTransaction> consumer
    ) {
        Objects.requireNonNull(transactions);
        Objects.requireNonNull(consumer);

        if (transactions.isEmpty()) {
            return;
        }

        validateTransactions(transactions);

        Window fiveMinutes = new Window(Duration.ofMinutes(5));
        Window oneHour = new Window(Duration.ofHours(1));
        Window oneDay = new Window(Duration.ofDays(1));
        Window sevenDays = new Window(Duration.ofDays(7));
        Window thirtyDays = new Window(Duration.ofDays(30));

        List<Window> windows = List.of(
                fiveMinutes,
                oneHour,
                oneDay,
                sevenDays,
                thirtyDays
        );

        Map<String, Long> recipientIds = new HashMap<>();
        Map<String, Instant> lastRecipientTransfer = new HashMap<>();
        Set<String> seenRecipients = new HashSet<>();

        long userId = parseAccountId(
                transactions.get(0).accountId()
        );

        Instant firstTransferTime = null;
        Instant lastTransferTime = null;

        int index = 0;

        while (index < transactions.size()) {
            Instant now = transactions.get(index).timestamp();

            for (Window window : windows) {
                window.expire(now);
            }

            int end = index;

            while (end < transactions.size()
                    && transactions.get(end).timestamp().equals(now)) {
                end++;
            }

            for (int i = index; i < end; i++) {
                Transaction transaction = transactions.get(i);

                Long recipientId = recipientIds.get(
                        transaction.beneficiaryIban()
                );

                if (recipientId == null) {
                    recipientId = (long) recipientIds.size() + 1;

                    recipientIds.put(
                            transaction.beneficiaryIban(),
                            recipientId
                    );
                }

                TransferData transfer = new TransferData(
                        transaction.lineNumber(),
                        userId,
                        recipientId,
                        transaction.amount(),
                        "EUR",
                        transaction.timestamp()
                );

                TransactionHistorySnapshot snapshot = createSnapshot(
                        transaction,
                        fiveMinutes,
                        oneHour,
                        oneDay,
                        sevenDays,
                        thirtyDays,
                        firstTransferTime,
                        lastTransferTime,
                        lastRecipientTransfer
                );

                consumer.accept(new PreparedTransaction(
                        transaction,
                        transfer,
                        snapshot
                ));
            }

            for (int i = index; i < end; i++) {
                Transaction transaction = transactions.get(i);

                var dateTime = transaction.timestamp()
                        .atZone(ZoneOffset.UTC);

                int hour = dateTime.getHour();

                boolean weekend =
                        dateTime.getDayOfWeek() == DayOfWeek.SATURDAY
                                || dateTime.getDayOfWeek() == DayOfWeek.SUNDAY;

                boolean newRecipient = seenRecipients.add(
                        transaction.beneficiaryIban()
                );

                Event event = new Event(
                        transaction.timestamp(),
                        transaction.beneficiaryIban(),
                        transaction.amount(),
                        hour,
                        hour < 6,
                        weekend,
                        newRecipient
                );

                for (Window window : windows) {
                    window.add(event);
                }

                if (firstTransferTime == null) {
                    firstTransferTime = now;
                }

                lastTransferTime = now;

                lastRecipientTransfer.put(
                        transaction.beneficiaryIban(),
                        now
                );
            }

            index = end;
        }
    }

    private static void validateTransactions(
            List<Transaction> transactions
    ) {
        String accountId = transactions.get(0).accountId();
        Instant previous = null;

        for (Transaction transaction : transactions) {
            Objects.requireNonNull(transaction);

            if (!transaction.accountId().equals(accountId)) {
                throw new IllegalArgumentException(
                        "Transactions must belong to one account"
                );
            }

            if (transaction.amount().signum() <= 0) {
                throw new IllegalArgumentException(
                        "Transfer amount must be positive"
                );
            }

            if (previous != null
                    && transaction.timestamp().isBefore(previous)) {
                throw new IllegalArgumentException(
                        "Transactions must be sorted by timestamp"
                );
            }

            previous = transaction.timestamp();
        }
    }

    private static long parseAccountId(String accountId) {
        if (!accountId.matches("ACC_[0-9]+")) {
            throw new IllegalArgumentException(
                    "Invalid SynSEPA account ID: " + accountId
            );
        }

        return Math.addExact(
                Long.parseLong(accountId.substring(4)),
                1L
        );
    }

    private static TransactionHistorySnapshot createSnapshot(
            Transaction transaction,
            Window fiveMinutes,
            Window oneHour,
            Window oneDay,
            Window sevenDays,
            Window thirtyDays,
            Instant firstTransferTime,
            Instant lastTransferTime,
            Map<String, Instant> lastRecipientTransfer
    ) {
        long count30d = thirtyDays.count();

        AmountStatistics amounts = new AmountStatistics(
                thirtyDays.average(),
                thirtyDays.median(),
                thirtyDays.maximum(),
                thirtyDays.percentile95(),
                thirtyDays.standardDeviation(),
                Optional.of(oneDay.sum()),
                Optional.of(sevenDays.sum())
        );

        OptionalLong secondsSinceLast = lastTransferTime == null
                ? OptionalLong.empty()
                : OptionalLong.of(
                        Duration.between(
                                lastTransferTime,
                                transaction.timestamp()
                        ).getSeconds()
                );

        VelocityStatistics velocity = new VelocityStatistics(
                OptionalLong.of(fiveMinutes.count()),
                OptionalLong.of(oneHour.count()),
                OptionalLong.of(oneDay.count()),
                OptionalLong.of(sevenDays.count()),
                OptionalLong.of(count30d),
                Optional.of(oneHour.sum()),
                OptionalLong.of(oneDay.uniqueRecipients()),
                OptionalLong.of(oneDay.newRecipients()),
                secondsSinceLast,
                OptionalLong.of(
                        oneHour.amountCount(transaction.amount())
                )
        );

        Instant previousRecipient = lastRecipientTransfer.get(
                transaction.beneficiaryIban()
        );

        OptionalLong secondsSinceRecipient = previousRecipient == null
                ? OptionalLong.empty()
                : OptionalLong.of(
                        Duration.between(
                                previousRecipient,
                                transaction.timestamp()
                        ).getSeconds()
                );

        OptionalDouble recipientShare = count30d == 0
                ? OptionalDouble.empty()
                : OptionalDouble.of(
                        (double) thirtyDays.recipientCount(
                                transaction.beneficiaryIban()
                        ) / count30d
                );

        RecipientStatistics recipients = new RecipientStatistics(
                OptionalLong.empty(),
                Optional.empty(),
                secondsSinceRecipient,
                recipientShare,
                OptionalLong.of(thirtyDays.uniqueRecipients())
        );

        OptionalLong historyDays = firstTransferTime == null
                ? OptionalLong.empty()
                : OptionalLong.of(
                        Duration.between(
                                firstTransferTime,
                                transaction.timestamp()
                        ).toDays()
                );

        int currentHour = transaction.timestamp()
                .atZone(ZoneOffset.UTC)
                .getHour();

        ActivityStatistics activity = new ActivityStatistics(
                OptionalLong.empty(),
                historyDays,
                OptionalDouble.of((double) count30d / 30.0),
                share(thirtyDays.nightCount(), count30d),
                share(thirtyDays.hourCount(currentHour), count30d),
                share(thirtyDays.weekendCount(), count30d),
                OptionalLong.empty()
        );

        return new TransactionHistorySnapshot(
                amounts,
                velocity,
                recipients,
                activity,
                RiskHistoryStatistics.empty(),
                transaction.timestamp()
        );
    }

    private static OptionalDouble share(
            long numerator,
            long denominator
    ) {
        return denominator == 0
                ? OptionalDouble.empty()
                : OptionalDouble.of((double) numerator / denominator);
    }

    public record PreparedTransaction(
            Transaction transaction,
            TransferData transfer,
            TransactionHistorySnapshot snapshot
    ) {
        public PreparedTransaction {
            Objects.requireNonNull(transaction);
            Objects.requireNonNull(transfer);
            Objects.requireNonNull(snapshot);
        }
    }

    private record Event(
            Instant timestamp,
            String recipient,
            BigDecimal amount,
            int hour,
            boolean night,
            boolean weekend,
            boolean newRecipient
    ) {
    }

    private static final class Window {

        private final Duration duration;
        private final Deque<Event> events = new ArrayDeque<>();
        private final Map<String, Integer> recipients = new HashMap<>();
        private final NavigableMap<BigDecimal, Integer> amounts =
                new TreeMap<>();
        private final int[] hours = new int[24];

        private BigDecimal sum = BigDecimal.ZERO;
        private double sumSquares;
        private long newRecipients;
        private long nightCount;
        private long weekendCount;

        private Window(Duration duration) {
            this.duration = duration;
        }

        private void add(Event event) {
            events.addLast(event);

            sum = sum.add(event.amount());

            double amount = event.amount().doubleValue();
            sumSquares += amount * amount;

            recipients.merge(
                    event.recipient(),
                    1,
                    Integer::sum
            );

            amounts.merge(
                    event.amount(),
                    1,
                    Integer::sum
            );

            hours[event.hour()]++;

            if (event.newRecipient()) {
                newRecipients++;
            }

            if (event.night()) {
                nightCount++;
            }

            if (event.weekend()) {
                weekendCount++;
            }
        }

        private void expire(Instant now) {
            Instant cutoff = now.minus(duration);

            while (!events.isEmpty()
                    && !events.peekFirst().timestamp().isAfter(cutoff)) {
                Event event = events.removeFirst();

                sum = sum.subtract(event.amount());

                double amount = event.amount().doubleValue();
                sumSquares -= amount * amount;

                recipients.computeIfPresent(
                        event.recipient(),
                        (key, count) -> count == 1 ? null : count - 1
                );

                amounts.computeIfPresent(
                        event.amount(),
                        (key, count) -> count == 1 ? null : count - 1
                );

                hours[event.hour()]--;

                if (event.newRecipient()) {
                    newRecipients--;
                }

                if (event.night()) {
                    nightCount--;
                }

                if (event.weekend()) {
                    weekendCount--;
                }
            }

            if (events.isEmpty()) {
                sum = BigDecimal.ZERO;
                sumSquares = 0.0;
            }
        }

        private long count() {
            return events.size();
        }

        private BigDecimal sum() {
            return sum;
        }

        private int uniqueRecipients() {
            return recipients.size();
        }

        private int recipientCount(String recipient) {
            return recipients.getOrDefault(recipient, 0);
        }

        private int amountCount(BigDecimal amount) {
            return amounts.getOrDefault(amount, 0);
        }

        private long newRecipients() {
            return newRecipients;
        }

        private long nightCount() {
            return nightCount;
        }

        private long weekendCount() {
            return weekendCount;
        }

        private int hourCount(int hour) {
            return hours[hour];
        }

        private Optional<BigDecimal> average() {
            if (events.isEmpty()) {
                return Optional.empty();
            }

            return Optional.of(
                    sum.divide(
                            BigDecimal.valueOf(count()),
                            MathContext.DECIMAL64
                    )
            );
        }

        private Optional<BigDecimal> maximum() {
            return amounts.isEmpty()
                    ? Optional.empty()
                    : Optional.of(amounts.lastKey());
        }

        private Optional<BigDecimal> median() {
            int count = events.size();

            if (count == 0) {
                return Optional.empty();
            }

            if (count % 2 == 1) {
                return Optional.of(select(count / 2 + 1));
            }

            return Optional.of(
                    select(count / 2)
                            .add(select(count / 2 + 1))
                            .divide(
                                    BigDecimal.valueOf(2),
                                    MathContext.DECIMAL64
                            )
            );
        }

        private Optional<BigDecimal> percentile95() {
            int count = events.size();

            if (count == 0) {
                return Optional.empty();
            }

            int rank = (int) ((95L * count + 99) / 100);

            return Optional.of(select(rank));
        }

        private Optional<BigDecimal> standardDeviation() {
            if (events.isEmpty()) {
                return Optional.empty();
            }

            double mean = sum.doubleValue() / count();

            double variance = Math.max(
                    0.0,
                    sumSquares / count() - mean * mean
            );

            double deviation = Math.sqrt(variance);

            if (!Double.isFinite(deviation)) {
                throw new IllegalStateException(
                        "Invalid standard deviation"
                );
            }

            return Optional.of(BigDecimal.valueOf(deviation));
        }

        private BigDecimal select(int rank) {
            if (rank < 1 || rank > events.size()) {
                throw new IllegalArgumentException("Invalid rank");
            }

            int accumulated = 0;

            for (var entry : amounts.entrySet()) {
                accumulated += entry.getValue();

                if (accumulated >= rank) {
                    return entry.getKey();
                }
            }

            throw new IllegalStateException(
                    "Amount distribution is inconsistent"
            );
        }
    }
}