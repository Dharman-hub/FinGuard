# FinGuard

**FinGuard** — встраиваемая Java-библиотека для обнаружения подозрительных денежных переводов с использованием правил и нейронной сети.

Библиотека предназначена для интеграции непосредственно в банковское приложение. Оценка риска выполняется локально через Java API и ONNX Runtime — без обязательного отдельного сервиса, сетевых запросов к ML API или Python-процесса.

**Стек:** Java 21 · Spring Boot 3.5 · Maven · ONNX Runtime · DJL · PyTorch

## Возможности

- **Три решения:** `APPROVED`, `REVIEW`, `REJECTED`.
- **Гибридная оценка риска:** правила и нейросеть.
- **MLP 70 → 32 → 16 → 1:** локальный инференс через ONNX Runtime.
- **Конфигурируемая политика:** веса правил, пороги, автоматическое отклонение и ручная проверка.
- **Неполные данные:** маски доступности признаков, без обязательной полной истории клиента.
- **Ручное обучение:** обучение с нуля и fine-tuning существующей модели.
- **Управление моделями:** версионирование, безопасная активация, откат и проверка SHA-256.

FinGuard оценивает риск денежных переводов, включая переводы, которые клиент самостоятельно совершает под воздействием мошенника. Онлайн-покупки и обычные карточные платежи не входят в текущую область применения.

## Quick Start

### 1. Подключение зависимости

Для использования FinGuard без Spring Boot достаточно модуля `finguard-core`.

**Maven:**

```xml
<dependency>
    <groupId>ru.finguard</groupId>
    <artifactId>finguard-core</artifactId>
    <version>0.1.0-SNAPSHOT</version>
</dependency>
```

Для интеграции со Spring Boot:

```xml
<dependency>
    <groupId>ru.finguard</groupId>
    <artifactId>finguard-spring-boot-starter</artifactId>
    <version>0.1.0-SNAPSHOT</version>
</dependency>
```

**Важно:** версия `0.1.0-SNAPSHOT` не опубликована в Maven Central. Пока библиотеку необходимо собрать из исходников и установить в локальный Maven-репозиторий:

```bash
git clone https://github.com/Dharman-hub/FinGuard.git
cd FinGuard
mvn -pl finguard-spring-boot-starter -am install
```

Команда установит необходимые модули, включая `finguard-core`. Для использования только ядра достаточно:

```bash
mvn -pl finguard-core -am install
```

### 2. Минимальный Java-пример

FinGuard получает данные текущего перевода и историю клиента через `TransactionDataProvider`.

```java
import ru.finguard.core.api.RiskEngine;
import ru.finguard.core.api.RiskEngineBuilder;
import ru.finguard.core.model.RiskAssessment;
import ru.finguard.core.model.TransactionHistorySnapshot;
import ru.finguard.core.model.TransferData;
import ru.finguard.core.provider.TransactionDataProvider;

import java.math.BigDecimal;
import java.time.Instant;

public class FinGuardExample {
    public static void main(String[] args) {
        TransactionDataProvider provider = transfer ->
                TransactionHistorySnapshot.empty(transfer.timestamp());

        RiskEngine engine = new RiskEngineBuilder(provider)
                .reviewThreshold(40)
                .rejectThreshold(80)
                .build();

        TransferData transfer = new TransferData(
                1001L,
                126L,
                777L,
                new BigDecimal("12500"),
                "RUB",
                Instant.now()
        );

        RiskAssessment result = engine.evaluate(transfer);

        System.out.println(result.decision());
        System.out.println(result.riskScore());
        System.out.println(result.factors());
    }
}
```

Этот пример запускает **движок правил без нейросети**. Для простоты он передаёт пустую историю: неизвестные исторические показатели не подменяются нулями.

При реальной интеграции банк предоставляет доступные исторические агрегаты. Финальное решение зависит от входных данных и настроенной политики.

### 3. Интеграция с банковскими данными

Центральная точка интеграции:

```java
public interface TransactionDataProvider {
    TransactionHistorySnapshot loadSnapshot(TransferData transfer);
}
```

`TransferData` содержит:

| Поле | Тип | Назначение |
|---|---|---|
| `transferId` | `long` | Идентификатор перевода |
| `userId` | `long` | Идентификатор отправителя |
| `recipientId` | `long` | Идентификатор получателя |
| `amount` | `BigDecimal` | Сумма |
| `currency` | `String` | Валюта |
| `timestamp` | `Instant` | Время перевода |

`TransactionHistorySnapshot` объединяет пять групп статистики:

| Группа | Примеры данных |
|---|---|
| `AmountStatistics` | Средняя, медианная и максимальная сумма переводов |
| `VelocityStatistics` | Количество переводов за разные интервалы времени |
| `RecipientStatistics` | История переводов конкретному получателю |
| `ActivityStatistics` | Особенности активности клиента |
| `RiskHistoryStatistics` | Доступная история рисков |

Пример реализации провайдера с одним известным историческим показателем:

```java
import ru.finguard.core.model.*;
import ru.finguard.core.provider.TransactionDataProvider;

import java.math.BigDecimal;
import java.util.Optional;

public class BankHistoryProvider implements TransactionDataProvider {

    @Override
    public TransactionHistorySnapshot loadSnapshot(TransferData transfer) {
        AmountStatistics amount = new AmountStatistics(
                Optional.of(new BigDecimal("5000")),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty(),
                Optional.empty()
        );

        return new TransactionHistorySnapshot(
                amount,
                VelocityStatistics.empty(),
                RecipientStatistics.empty(),
                ActivityStatistics.empty(),
                RiskHistoryStatistics.empty(),
                transfer.timestamp()
        );
    }
}
```

Значение `5000` здесь условное. В банковском приложении оно должно поступать из рассчитанной истории клиента.

FinGuard не выполняет отдельные SQL-запросы для каждого признака. Банк самостоятельно получает агрегаты удобным ему способом и передаёт готовый snapshot.

Отсутствующие показатели представлены через `Optional`, `OptionalLong` и `OptionalDouble`.

**Отсутствующий показатель и измеренный ноль — разные состояния.**

## Конфигурация правил

FinGuard содержит четыре встроенных правила:

| Правило | Вес по умолчанию |
|---|---:|
| `UNUSUAL_AMOUNT` | 25 |
| `HIGH_VELOCITY` | 35 |
| `NEW_RECIPIENT` | 20 |
| `UNUSUAL_ACTIVITY` | 20 |

Через Java API можно изменять веса, удалять правила, добавлять собственные реализации `RiskRule` и устанавливать пороги.

```java
RiskEngine engine = new RiskEngineBuilder(provider)
        .weight("HIGH_VELOCITY", 45)
        .weight("NEW_RECIPIENT", 25)
        .removeRule("UNUSUAL_ACTIVITY")
        .reviewThreshold(40)
        .rejectThreshold(80)
        .build();
```

Для Spring Boot предусмотрена конфигурация через `application.yml`:

```yaml
finguard:
  enabled: true

  review-threshold: 40
  reject-threshold: 80

  rules:
    unusual-amount:
      enabled: true
      weight: 25

    high-velocity:
      enabled: true
      weight: 35

    new-recipient:
      enabled: true
      weight: 20

    unusual-activity:
      enabled: true
      weight: 20
```

Пороги относятся к итоговому баллу правил от 0 до 100. Нейросеть использует отдельные пороги в пространстве логитов.

## Принятие решений

| Решение | Значение |
|---|---|
| `APPROVED` | Перевод соответствует условиям автоматического одобрения |
| `REVIEW` | Требуется дополнительная проверка |
| `REJECTED` | Перевод отклонён по действующей политике |

Нейросеть возвращает логит риска. Банк может установить отдельные пороги для `REVIEW` и `REJECTED`.

Для автоматического `REJECTED` по нейросети нужен явно заданный порог. Если он отсутствует, даже высокий логит сам по себе не приводит к автоматической блокировке.

Настройки определяются банком. FinGuard не навязывает обязательные исторические признаки или универсальные требования к полноте данных.

## Нейросетевая модель

Исходная модель FinGuard v3 — многослойный персептрон:

```text
Input (70)
    ↓
Linear (32) + ReLU
    ↓
Linear (16) + ReLU
    ↓
Linear (1)
    ↓
Risk logit
```

Входной вектор состоит из:

- 35 числовых значений признаков;
- 35 масок доступности этих признаков.

Один из признаков связан с историей IBAN получателя. Остальные описывают текущий перевод, клиента и доступную историю операций.

Нормализация выполняется с использованием профиля, соответствующего конкретной модели.

### Исходный датасет SynSEPA

Для создания исходной модели использовался синтетический набор денежных переводов SynSEPA и подготовленные в проекте инструменты обработки данных.

В рассматриваемую версию включены три типа мошенничества:

- `Invoice fraud`
- `Impersonation fraud`
- `CEO fraud`

`Romance fraud` не входит в оцениваемую выборку.

Инструменты подготовки SynSEPA находятся в пакете `ru.finguard.training.dataset.synsepa`. Они отделены от универсального банковского конвейера обучения.

### Разделение данных

Для исходной SynSEPA v3 использовалось разбиение по аккаунтам с фиксированным распределением приблизительно 70/15/15:

| Выборка | Переводов |
|---|---:|
| TRAIN | 1 283 654 |
| VALIDATION | 276 807 |
| TEST | 274 867 |
| **Всего** | **1 835 328** |

Разделение по аккаунтам предотвращает попадание одного аккаунта в несколько выборок. Внутри истории аккаунта соблюдается порядок транзакций.

Это не следует путать с глобальным хронологическим разделением банковских данных: для него используется отдельный `BankV3DatasetPreparer`.

### Результаты независимого TEST

Пороги подбирались на VALIDATION. TEST использовался для независимой оценки, без повторной настройки порога.

Основной режим исходной модели: **5 FP на 1000 легитимных переводов**.

| Метрика | TEST |
|---|---:|
| Всего переводов | 274 867 |
| Легитимных | 274 444 |
| Мошеннических | 423 |
| TP | 423 |
| FP | 1 323 |
| TN | 273 121 |
| FN | 0 |
| Precision | 24,23% |
| Recall | 100% |
| F1-score | 39,00% |
| Average Precision (AP) | 69,56% |
| FP / 1000 | 4,8207 |
| Порог REVIEW (logit) | 1,5145587 |

Значения Precision, Recall и F1 относятся к конкретному рабочему порогу. Average Precision характеризует ранжирование модели по всей PR-кривой.

**ROC-AUC в приведённых результатах не измерялся и не подменяется Average Precision.**

Высокий Recall в данном синтетическом тесте не означает гарантированного обнаружения всех мошеннических переводов в реальной банковской системе.

Из-за сильного дисбаланса классов даже небольшой процент ложных срабатываний может приводить к заметному количеству проверок легитимных переводов.

### Другие рабочие пороги

| Бюджет VALIDATION | Порог logit | Recall на TEST |
|---|---:|---:|
| 1 FP / 1000 | 5,048477 | 61,47% |
| 2 FP / 1000 | 3,7777753 | 78,01% |
| 5 FP / 1000 | 1,5145587 | 100% |

Это результаты одной исходной модели на SynSEPA, а не рекомендуемые универсальные пороги для всех банков.

Каждая переобученная модель должна калиброваться отдельно.

## Производительность

Локальные измерения проводились на Java 21 с использованием ONNX Runtime. В одном запуске бенчмарка использовалось 8 000 проверок.

| Сценарий | Среднее | p95 | p99 |
|---|---:|---:|---:|
| Только ONNX-инференс | 0,0168 мс | 0,0218 мс | 0,0426 мс |
| `RiskEngine` с нейросетью | 0,0189 мс | 0,0230 мс | 0,0297 мс |
| `RiskEngine` без нейросети | 0,0268 мс | 0,0348 мс | 0,0523 мс |

Это результаты отдельных локальных измерений, а не универсальные гарантии задержки. Сценарии измерялись отдельно; различия могут зависеть от прогрева JVM, входных данных и условий выполнения.

Бенчмарк находится в `FinGuardOnnxBenchmark`.

Потребление памяти в качестве отдельного подтверждённого показателя не публикуется.

## Spring Boot Starter

Модуль `finguard-spring-boot-starter` предоставляет автоконфигурацию FinGuard для Spring Boot 3.5.x.

Чтобы включить нейросеть:

```yaml
finguard:
  neural:
    enabled: true
    versioning-enabled: true
    initial-version: v3.0
    mode: CUSTOM
    review-threshold-logit: 1.5145587
    model-path: /path/to/finguard-model.onnx
    normalization-path: /path/to/normalization.csv
```

Пути необходимо заменить абсолютными путями к доступным артефактам модели.

При использовании Spring Boot приложение должно зарегистрировать реализацию `TransactionDataProvider` как Spring Bean.

Если `finguard.neural.enabled=false` или свойство отсутствует, FinGuard может использовать движок правил без нейросети.

Если нейросеть включена, но обязательные модель или профиль нормализации отсутствуют либо некорректны, это ошибка инициализации соответствующих компонентов. FinGuard не должен рассматриваться как система, которая гарантированно и незаметно переключается на rules-only при любой ошибке загрузки ONNX.

## Запуск демонстрации

В проект входит отдельное демонстрационное Spring Boot приложение.

Это удобный способ посмотреть работу FinGuard через HTTP, но **REST не обязателен для использования библиотеки**.

### Сборка демо

Из корня репозитория:

```bash
mvn -pl finguard-demo -am clean package
```

Команда собирает необходимые для демо модули без обязательной сборки модуля обучения.

### Запуск

Перейдите в каталог демонстрационного приложения:

```bash
cd finguard-demo
```

По умолчанию запускается режим правил:

```bash
java -jar target/finguard-demo-0.1.0-SNAPSHOT.jar
```

Для запуска с исходной нейросетью:

```bash
java -jar target/finguard-demo-0.1.0-SNAPSHOT.jar --spring.profiles.active=neural
```

Профиль `neural` находится в `src/main/resources/application-neural.yml`. Относительные пути к модели в этом профиле рассчитаны на рабочую директорию `finguard-demo`.

Демо доступно по адресу `127.0.0.1:8081`.

### Пример HTTP-запроса

```http
POST http://localhost:8081/demo/risk-checks
Content-Type: application/json
```

```json
{
  "transferId": 1001,
  "userId": 126,
  "recipientId": 777,
  "amount": 12500,
  "currency": "RUB",
  "timestamp": "2026-11-06T12:00:00Z"
}
```

### Пример ответа с нейросетью

Ниже приведён результат одного демонстрационного запуска:

```json
{
  "decision": "REVIEW",
  "riskScore": 75,
  "factors": [
    "HIGH_VELOCITY",
    "NEW_RECIPIENT",
    "UNUSUAL_ACTIVITY",
    "NEURAL_REVIEW"
  ],
  "neural": {
    "logit": 8.785048,
    "reviewThresholdLogit": 1.5145587
  },
  "reason": "NEURAL_REVIEW",
  "trace": {
    "modelSha256": "33742ed17bbd4e1b6dc31d7feeb866c2257be479e587d3f4fba80a69a5110e27",
    "thresholdsFingerprint": "475195d3577b1c4838a881c08952ccd3518af6dadbb7f79442c596d1b07d42c9",
    "reviewThreshold": 40,
    "rejectThreshold": 80,
    "neuralReviewThresholdLogit": 1.5145587,
    "policyVersion": 0,
    "neuralRejectThresholdLogit": null
  }
}
```

В этом сценарии правила и нейросеть указывают на повышенный риск.

Итоговое решение — `REVIEW`, потому что модель превысила порог проверки, а отдельный порог нейросетевого `REJECTED` не был установлен.

Результат зависит от переданных данных и состояния политики.

## Ручное обучение банковской модели

Обучение выполняется в отдельном модуле `finguard-ml-training`.

Для обучения используется Java API, а не REST, CLI-сервис или автоматически выполняемая задача.

### Формат входных данных

Банк предоставляет реализации `TrainingDataProvider`:

```java
@FunctionalInterface
public interface TrainingDataProvider {
    Stream<ConfirmedTrainingTransfer> streamConfirmedTransfers();
}
```

Каждый `ConfirmedTrainingTransfer` включает:

| Поле | Назначение |
|---|---|
| `transferId` | Уникальный идентификатор |
| `transfer` | Данные перевода |
| `history` | История, доступная на момент перевода |
| `ibanSignal` | Необязательный IBAN-признак |
| `label` | Подтверждённая метка |
| `confirmedAt` | Время подтверждения метки |

Метка принимает одно из значений:

- `TrainingLabel.FRAUD`
- `TrainingLabel.LEGITIMATE`

`BankV3TrainingSampleMapper` извлекает признаки на Java и формирует входной вектор из 70 чисел.

Предоставлять CSV необязательно. Источник может быть базой данных, файлом, локальным хранилищем или другим адаптером, реализующим `TrainingDataProvider`.

### Подготовка выборки

`BankV3DatasetPreparer` разделяет обучающие примеры по времени события.

```java
BankV3DatasetPreparer.Result dataset =
        new BankV3DatasetPreparer().prepare(
                new BankV3TrainingSampleMapper().stream(bankProvider),
                Path.of("work/dataset-v1"),
                Instant.parse("2025-07-01T00:00:00Z"),
                Instant.parse("2025-10-01T00:00:00Z"),
                Instant.parse("2026-01-01T00:00:00Z")
        );
```

`bankProvider` — экземпляр `TrainingDataProvider`, предоставленный интегрирующим приложением.

Аргументы времени задают:

- `trainingEnd` — конец TRAIN;
- `validationEnd` — конец VALIDATION;
- `labelsAsOf` — момент, до которого допустимы подтверждённые метки для TEST.

Примеры с метками, подтверждёнными позже соответствующей временной границы, исключаются, чтобы уменьшить риск утечки информации из будущего.

Это **временное разделение**, а не случайная стратификация.

Результатом являются:

```text
dataset-v1/
├── train.fgbin
├── validation.fgbin
├── test.fgbin
└── manifest.properties
```

Файл `manifest.properties` фиксирует временные границы, количество мошеннических и легитимных переводов и число пропущенных поздних подтверждений.

### Запуск обучения

Ниже пример ручного запуска после подготовки бинарной выборки.

```java
import ru.finguard.training.bank.*;

import java.nio.file.Path;

public class BankTrainingExample {

    public static void run() throws Exception {
        var settings = BankV3FromScratchTrainer.Settings.defaults();

        var criteria = new BankV3CandidateReleaseGate.Criteria(
                5,
                10000,
                100,
                10000,
                100,
                5.0,
                0.70,
                0.50,
                10.0,
                0.30
        );

        var limits = new BankV3ManualTrainingManager.ResourceLimits(
                5, 512, 8192, 512
        );

        var request = new BankV3ManualTrainingManager.Request(
                Path.of("work/dataset-v1"),
                Path.of("work/trained-v1"),
                Path.of("work/release-v1"),
                settings,
                5,
                512,
                criteria
        );

        try (var manager = new BankV3ManualTrainingManager(limits)) {
            long jobId = manager.startFromScratch(request);
            var status = manager.awaitCompletion(jobId);

            if (status.state() != BankV3ManualTrainingManager.State.SUCCEEDED) {
                throw new IllegalStateException(status.error());
            }

            var release = manager.result(jobId).orElseThrow().release();
            System.out.println(release.modelPath());
        }
    }
}
```

Числа в `Criteria` здесь **иллюстративные**, а не рекомендуемые требования к реальному банковскому внедрению. Папка `work/dataset-v1` должна уже содержать подготовленную выборку с достаточным числом примеров обоих классов.

Для fine-tuning используется `startFineTune(request, sourceCheckpoint)`.

Автоматического переобучения по расписанию или после каждого перевода нет.

### Критерии выпуска модели

`BankV3CandidateReleaseGate.Criteria` позволяет задать:

| Параметр | Назначение |
|---|---|
| `fpBudgetPerThousand` | Бюджет FP/1000 для калибровки |
| `minValidationLegitimate` | Минимум легитимных примеров VALIDATION |
| `minValidationFraud` | Минимум мошеннических примеров VALIDATION |
| `minTestLegitimate` | Минимум легитимных примеров TEST |
| `minTestFraud` | Минимум мошеннических примеров TEST |
| `maxTestFpPerThousand` | Максимальный FP/1000 на TEST |
| `minTestRecall` | Минимальный Recall на TEST |
| `minTestAveragePrecision` | Минимальный AP на TEST |
| `maxNoVelocityFpPerThousand` | Максимальный FP/1000 без истории скорости переводов |
| `minNoRecipientRecall` | Минимальный Recall без истории получателя и IBAN |

Кандидат также проверяется на сценариях отсутствия исторических данных. Это **офлайн-стресс-тестирование модели**, а не отдельный режим runtime-инференса.

Если проверка качества не пройдена, публикация кандидата не выполняется.

Если пройдена, релиз создаётся, но **не активируется автоматически**.

## Артефакты релиза

После успешного обучения и проверки качества:

```text
release-v1/
├── finguard-model.onnx
├── normalization.csv
├── training.properties
└── quality.properties
```

### training.properties

Пример структуры:

```properties
format=bank-v3-1
inputSize=70
trainingMode=FROM_SCRATCH
sourceModel=NONE
trainTotal=80000
trainFraud=800
validationTotal=20000
validationFraud=200
positiveWeight=99.0
bestEpoch=4
bestValidationLoss=0.42
```

Значения приведены для иллюстрации формата.

### quality.properties

Содержит результаты проверки кандидата и параметры допуска:

```properties
format=bank-v3-candidate-1
eligibleForManualReview=true
automaticActivation=false
fpBudgetPerThousand=5
reviewThresholdLogit=2.25
validationLegitimate=19800
validationFraud=200
validationFpPerThousand=4.8
validationRecall=0.80
testLegitimate=19800
testFraud=200
testFpPerThousand=4.9
testRecall=0.79
testAveragePrecision=0.65
noVelocityFpPerThousand=8.0
noRecipientRecall=0.60
minValidationLegitimate=10000
minValidationFraud=100
minTestLegitimate=10000
minTestFraud=100
maxTestFpPerThousand=5.0
minTestRecall=0.70
minTestAveragePrecision=0.50
maxNoVelocityFpPerThousand=10.0
minNoRecipientRecall=0.30
modelSha256=<SHA-256 модели>
normalizationSha256=<SHA-256 нормализации>
```

Это пример полей, а не реальный отчёт об обучении. Отпечатки и числовые результаты генерируются на основании конкретного релиза.

Файл не следует редактировать вручную после публикации.

## Активация, версии и откат

`FinGuardModelManager` управляет регистрацией, проверкой, активацией и откатом ONNX-моделей.

Предварительный просмотр позволяет проверить новую модель и настройки до применения.

При использовании `ModelPolicySnapshotStore` модель и её политика порогов сохраняются как согласованное состояние.

При откате восстанавливается предыдущая модель вместе с соответствующей политикой.

### Защищённая активация

Опционально можно включить:

```yaml
finguard:
  neural:
    require-approved-releases: true
```

В этом режиме FinGuard дополнительно проверяет соответствие зарегистрированных артефактов утверждённому релизу.

Используются SHA-256 ONNX, профиля нормализации и `quality.properties`, а также подтверждённый порог REVIEW.

Отпечаток одобренного отчёта должен поступать из доверенного процесса согласования банка.

Этот механизм защищает согласованность релиза, но не является цифровой подписью и не заменяет оценку качества модели.

## Обработка ошибок

Если обязательный `TransactionDataProvider` не может предоставить историю перевода, FinGuard выбрасывает `RiskCheckUnavailableException`.

В таком случае проверка не завершена. Библиотека не превращает ошибку автоматически в `APPROVED`.

Банковское приложение должно самостоятельно выбрать поведение: сохранить перевод в `PENDING`, повторить проверку или передать случай на операционный разбор.

Дополнительные источники данных и нейросетевая оценка имеют собственные сценарии обработки недоступности.

## Сборка и поддерживаемые платформы

Для работы с исходниками необходимы JDK 21 и Maven.

### Сборка ядра и Starter

```bash
mvn -pl finguard-spring-boot-starter -am clean test
mvn -pl finguard-spring-boot-starter -am package
```

### Полная сборка

```bash
mvn clean test
mvn package
```

**Внимание:** текущая конфигурация `finguard-ml-training` использует нативную зависимость PyTorch для Windows x86-64.

Поэтому полная сборка с тестами обучения на Linux или macOS не гарантируется без адаптации нативных зависимостей.

Модули обычного банковского инференса не требуют DJL/PyTorch: они используют ONNX Runtime.

## Тестирование

FinGuard включает модульные, интеграционные и многопоточные тесты.

Они проверяют:

- Правила, баллы риска и принятие решений.
- Корректность извлечения признаков.
- Отсутствующие значения и маски.
- ONNX-инференс и совместимость с DJL.
- Подготовку TRAIN/VALIDATION/TEST.
- Обучение с нуля и fine-tuning.
- Калибровку и оценку качества модели.
- Проверку кандидата перед публикацией.
- Активацию, откат и восстановление состояния.
- SHA-256 и защиту одобренных релизов.
- Поведение при сбоях хранилищ и конкурентном доступе.

Некоторые тесты исходной SynSEPA v3 требуют локальных датасетов. Побайтовое сравнение с оригинальной выборкой запускается отдельно через системное свойство `finguard.verify.synsepa=true`.

## Структура проекта

```text
FinGuard/
├── finguard-core/
│   └── API, правила, признаки, ONNX-инференс
├── finguard-spring-boot-starter/
│   └── Автоконфигурация, политики, версии, откат
├── finguard-ml-training/
│   └── Обучение, калибровка, тестирование, экспорт
├── finguard-demo/
│   └── Демонстрационное Spring Boot приложение
├── pom.xml
└── README.md
```

`finguard-ml-training` не является обязательной зависимостью банковского приложения, которое только проверяет переводы.

## Ограничения

FinGuard — **пет-проект**, ориентированный на проектирование и реализацию встраиваемой Java-системы обнаружения мошеннических переводов.

Проект не заявляется как готовая промышленная антифрод-платформа.

Основные ограничения:

- Исходная нейросеть оценивалась на синтетическом SynSEPA, а не на реальном банковском потоке.
- Для промышленного применения необходимы независимая валидация, мониторинг качества и проверка на актуальных банковских данных.
- Текущий модуль обучения с PyTorch настроен под Windows x86-64.
- Представленные значения задержек относятся к локальному бенчмарку.
- Демо не заменяет реальную банковскую интеграцию, авторизацию и инфраструктуру обработки платежей.
- Защищённая активация проверяет согласованность и одобрение релиза, но не заменяет полноценную инфраструктуру управления доверенными артефактами.