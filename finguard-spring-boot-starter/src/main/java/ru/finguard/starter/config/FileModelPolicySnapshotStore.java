package ru.finguard.starter.config;

import ru.finguard.core.decision.NeuralDataQualityPolicy;
import ru.finguard.core.decision.RuntimeThresholdPolicy;
import ru.finguard.core.neural.OnnxModelRegistry;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.channels.FileLock;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.ConcurrentModificationException;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;

public final class FileModelPolicySnapshotStore implements ModelPolicySnapshotStore {
    private static final int MAGIC = 0x46475053;
    private static final int FORMAT_VERSION = 1;
    private static final int CHECKSUM_BYTES = 32;
    private static final int MAX_FILE_BYTES = 16 * 1024 * 1024;
    private static final int MAX_COLLECTION_SIZE = 100_000;

    private static final ConcurrentHashMap<Path, Object> LOCAL_LOCKS =
            new ConcurrentHashMap<>();

    private final Path file;
    private final Path lockFile;

    public FileModelPolicySnapshotStore(Path file) throws IOException {
        this.file = Objects.requireNonNull(file)
                .toAbsolutePath()
                .normalize();

        Path parent = this.file.getParent();

        if (parent == null) {
            throw new IllegalArgumentException(
                    "State file must have a parent directory"
            );
        }

        Files.createDirectories(parent);

        this.lockFile = this.file.resolveSibling(
                this.file.getFileName() + ".lock"
        );
    }

    @Override
    public Optional<State> load() throws IOException {
        return withLock(this::read);
    }

    @Override
    public void saveIfRevisionMatches(
            State next,
            long expectedRevision
    ) throws IOException {
        Objects.requireNonNull(next);

        withLock(() -> {
            long actual = read()
                    .map(State::revision)
                    .orElse(-1L);

            if (actual != expectedRevision) {
                throw new ConcurrentModificationException(
                        "Expected revision "
                                + expectedRevision
                                + ", found "
                                + actual
                );
            }

            if (next.revision()
                    != Math.addExact(expectedRevision, 1L)) {
                throw new IllegalArgumentException(
                        "New revision must be incremented by one"
                );
            }

            writeAtomically(next);
            return null;
        });
    }

    private Optional<State> read() throws IOException {
        if (Files.isSymbolicLink(file)) {
            throw new IOException(
                    "State path must not be a symbolic link"
            );
        }

        if (!Files.exists(file)) {
            return Optional.empty();
        }

        long size = Files.size(file);

        if (size <= CHECKSUM_BYTES || size > MAX_FILE_BYTES) {
            throw new IOException("Invalid state file size");
        }

        byte[] bytes = Files.readAllBytes(file);
        int payloadSize = bytes.length - CHECKSUM_BYTES;

        byte[] expected = digest(bytes, 0, payloadSize);
        byte[] actual = Arrays.copyOfRange(
                bytes,
                payloadSize,
                bytes.length
        );

        if (!MessageDigest.isEqual(expected, actual)) {
            throw new IOException("State checksum mismatch");
        }

        try (DataInputStream input = new DataInputStream(
                new ByteArrayInputStream(bytes, 0, payloadSize))) {

            if (input.readInt() != MAGIC
                    || input.readInt() != FORMAT_VERSION) {
                throw new IOException("Unsupported state format");
            }

            ModelStateStore.State modelState =
                    readModelState(input);

            RuntimeThresholdPolicy.Snapshot activePolicy =
                    readPolicy(input);

            RuntimeThresholdPolicy.Snapshot previousPolicy =
                    input.readBoolean() ? readPolicy(input) : null;

            if (input.available() != 0) {
                throw new IOException(
                        "Trailing data in state file"
                );
            }

            return Optional.of(
                    new State(
                            modelState,
                            activePolicy,
                            previousPolicy
                    )
            );
        } catch (RuntimeException exception) {
            throw new IOException(
                    "Invalid persisted model-policy state",
                    exception
            );
        }
    }

    private void writeAtomically(State next) throws IOException {
        ByteArrayOutputStream payload =
                new ByteArrayOutputStream();

        try (DataOutputStream output =
                     new DataOutputStream(payload)) {

            output.writeInt(MAGIC);
            output.writeInt(FORMAT_VERSION);

            writeModelState(output, next.modelState());
            writePolicy(output, next.activePolicy());

            output.writeBoolean(
                    next.previousPolicy() != null
            );

            if (next.previousPolicy() != null) {
                writePolicy(output, next.previousPolicy());
            }
        }

        byte[] content = payload.toByteArray();

        if (content.length + CHECKSUM_BYTES > MAX_FILE_BYTES) {
            throw new IOException(
                    "State exceeds maximum size"
            );
        }

        byte[] checksum = digest(
                content,
                0,
                content.length
        );

        Path temporary = Files.createTempFile(
                file.getParent(),
                file.getFileName().toString(),
                ".tmp"
        );

        try {
            try (FileChannel channel = FileChannel.open(
                    temporary,
                    StandardOpenOption.WRITE)) {

                try (DataOutputStream output =
                             new DataOutputStream(
                                     java.nio.channels.Channels
                                             .newOutputStream(channel))) {

                    output.write(content);
                    output.write(checksum);
                    output.flush();
                    channel.force(true);
                }
            }

            try {
                Files.move(
                        temporary,
                        file,
                        StandardCopyOption.ATOMIC_MOVE,
                        StandardCopyOption.REPLACE_EXISTING
                );
            } catch (AtomicMoveNotSupportedException exception) {
                throw new IOException(
                        "Atomic file replacement is required",
                        exception
                );
            }
        } finally {
            Files.deleteIfExists(temporary);
        }
    }

    private void writeModelState(
            DataOutputStream output,
            ModelStateStore.State state
    ) throws IOException {
        output.writeLong(state.revision());
        output.writeLong(state.auditVersion());
        output.writeUTF(state.activeVersion());

        writeNullableString(
                output,
                state.previousVersion()
        );

        output.writeInt(state.models().size());

        for (OnnxModelRegistry.ModelVersion model : state.models()) {
            output.writeUTF(model.version());
            output.writeUTF(model.modelPath().toString());
            output.writeUTF(model.normalizationPath().toString());
            output.writeUTF(model.modelSha256());
            output.writeUTF(model.normalizationSha256());
            output.writeInt(model.inputSize());
            output.writeInt(model.featureCount());
        }

        output.writeInt(state.auditHistory().size());

        for (FinGuardModelManager.ModelAuditEntry entry
                : state.auditHistory()) {

            output.writeLong(entry.version());
            output.writeLong(entry.changedAt().getEpochSecond());
            output.writeInt(entry.changedAt().getNano());
            output.writeUTF(entry.actorId());
            output.writeUTF(entry.type().name());
            output.writeUTF(entry.fromVersion());
            output.writeUTF(entry.toVersion());
            output.writeUTF(entry.modelSha256());
            output.writeUTF(entry.normalizationSha256());
        }
    }

    private ModelStateStore.State readModelState(
            DataInputStream input
    ) throws IOException {
        long revision = input.readLong();
        long auditVersion = input.readLong();
        String activeVersion = input.readUTF();
        String previousVersion = readNullableString(input);

        int modelCount = readCount(input);

        List<OnnxModelRegistry.ModelVersion> models =
                new ArrayList<>(modelCount);

        for (int i = 0; i < modelCount; i++) {
            models.add(new OnnxModelRegistry.ModelVersion(
                    input.readUTF(),
                    Path.of(input.readUTF()),
                    Path.of(input.readUTF()),
                    input.readUTF(),
                    input.readUTF(),
                    input.readInt(),
                    input.readInt()
            ));
        }

        int auditCount = readCount(input);

        List<FinGuardModelManager.ModelAuditEntry> history =
                new ArrayList<>(auditCount);

        for (int i = 0; i < auditCount; i++) {
            history.add(
                    new FinGuardModelManager.ModelAuditEntry(
                            input.readLong(),
                            Instant.ofEpochSecond(
                                    input.readLong(),
                                    input.readInt()
                            ),
                            input.readUTF(),
                            FinGuardModelManager.ChangeType.valueOf(
                                    input.readUTF()
                            ),
                            input.readUTF(),
                            input.readUTF(),
                            input.readUTF(),
                            input.readUTF()
                    )
            );
        }

        return new ModelStateStore.State(
                revision,
                auditVersion,
                activeVersion,
                previousVersion,
                models,
                history
        );
    }

    private void writePolicy(
            DataOutputStream output,
            RuntimeThresholdPolicy.Snapshot policy
    ) throws IOException {
        output.writeLong(policy.version());
        output.writeInt(policy.reviewThreshold());
        output.writeInt(policy.rejectThreshold());

        output.writeBoolean(
                policy.neuralReviewThresholdLogit().isPresent()
        );

        if (policy.neuralReviewThresholdLogit().isPresent()) {
            output.writeFloat(
                    policy.neuralReviewThresholdLogit()
                            .orElseThrow()
            );
        }

        output.writeBoolean(
                policy.neuralRejectThresholdLogit().isPresent()
        );

        if (policy.neuralRejectThresholdLogit().isPresent()) {
            output.writeFloat(
                    policy.neuralRejectThresholdLogit()
                            .orElseThrow()
            );
        }

        NeuralDataQualityPolicy quality =
                policy.neuralDataQualityPolicy();

        writeRequirements(
                output,
                quality.approvalRequirements()
        );

        writeRequirements(
                output,
                quality.rejectionRequirements()
        );

        output.writeUTF(
                quality.insufficientDataAction().name()
        );
    }

    private RuntimeThresholdPolicy.Snapshot readPolicy(
            DataInputStream input
    ) throws IOException {
        long version = input.readLong();
        int review = input.readInt();
        int reject = input.readInt();

        Optional<Float> neuralReview = input.readBoolean()
                ? Optional.of(input.readFloat())
                : Optional.empty();

        Optional<Float> neuralReject = input.readBoolean()
                ? Optional.of(input.readFloat())
                : Optional.empty();

        NeuralDataQualityPolicy quality =
                new NeuralDataQualityPolicy(
                        readRequirements(input),
                        readRequirements(input),
                        NeuralDataQualityPolicy.MissingDataAction
                                .valueOf(input.readUTF())
                );

        return new RuntimeThresholdPolicy.Snapshot(
                version,
                review,
                reject,
                neuralReview,
                neuralReject,
                quality
        );
    }

    private void writeRequirements(
            DataOutputStream output,
            NeuralDataQualityPolicy.Requirements requirements
    ) throws IOException {
        output.writeInt(
                requirements.minimumObservedFeatures()
        );
        output.writeInt(
                requirements.minimumVelocityFeatures()
        );
        output.writeInt(
                requirements.minimumRecipientFeatures()
        );
        output.writeBoolean(
                requirements.requireIban()
        );
    }

    private NeuralDataQualityPolicy.Requirements readRequirements(
            DataInputStream input
    ) throws IOException {
        return new NeuralDataQualityPolicy.Requirements(
                input.readInt(),
                input.readInt(),
                input.readInt(),
                input.readBoolean()
        );
    }

    private void writeNullableString(
            DataOutputStream output,
            String value
    ) throws IOException {
        output.writeBoolean(value != null);

        if (value != null) {
            output.writeUTF(value);
        }
    }

    private String readNullableString(
            DataInputStream input
    ) throws IOException {
        return input.readBoolean() ? input.readUTF() : null;
    }

    private int readCount(DataInputStream input) throws IOException {
        int count = input.readInt();

        if (count < 0 || count > MAX_COLLECTION_SIZE) {
            throw new IOException(
                    "Invalid collection size: " + count
            );
        }

        return count;
    }

    private byte[] digest(
            byte[] value,
            int offset,
            int length
    ) {
        try {
            MessageDigest sha256 =
                    MessageDigest.getInstance("SHA-256");

            sha256.update(value, offset, length);
            return sha256.digest();
        } catch (NoSuchAlgorithmException exception) {
            throw new IllegalStateException(
                    "SHA-256 unavailable",
                    exception
            );
        }
    }

    private <T> T withLock(IoAction<T> action) throws IOException {
        Object monitor = LOCAL_LOCKS.computeIfAbsent(
                lockFile,
                ignored -> new Object()
        );

        synchronized (monitor) {
            try (FileChannel channel = FileChannel.open(
                    lockFile,
                    StandardOpenOption.CREATE,
                    StandardOpenOption.WRITE
            );
                 FileLock ignored = channel.lock()) {

                return action.perform();
            }
        }
    }

    @FunctionalInterface
    private interface IoAction<T> {
        T perform() throws IOException;
    }
}