package ru.finguard.training.bank;

import ru.finguard.core.feature.V3InputNormalizer;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.Objects;

public final class BankV3DatasetBinary {
    private static final int MAGIC = 0x46474233;
    private static final int VERSION = 1;
    private static final int INPUT_SIZE = V3InputNormalizer.INPUT_SIZE;

    private BankV3DatasetBinary() {
    }

    public static Writer openWriter(Path file) throws IOException {
        return new Writer(file);
    }

    public static Reader openReader(Path file) throws IOException {
        return new Reader(file);
    }

    public static final class Writer implements AutoCloseable {
        private final Path file;
        private final DataOutputStream output;
        private long count;
        private boolean closed;

        private Writer(Path file) throws IOException {
            this.file = Objects.requireNonNull(file);
            output = new DataOutputStream(new BufferedOutputStream(
                    Files.newOutputStream(file), 1024 * 1024));

            output.writeInt(MAGIC);
            output.writeInt(VERSION);
            output.writeInt(INPUT_SIZE);
            output.writeLong(0L);
        }

        public void write(BankV3TrainingSample sample) throws IOException {
            Objects.requireNonNull(sample);

            if (closed) {
                throw new IllegalStateException("Dataset writer is closed");
            }

            output.writeUTF(sample.transferId());
            output.writeLong(sample.userId());
            output.writeLong(sample.eventTime().toEpochMilli());
            output.writeLong(sample.confirmedAt().toEpochMilli());
            output.writeByte(sample.label());

            for (float value : sample.modelInput()) {
                output.writeFloat(value);
            }

            count++;
        }

        public long count() {
            return count;
        }

        @Override
        public void close() throws IOException {
            if (closed) {
                return;
            }

            closed = true;
            output.close();

            try (RandomAccessFile patch =
                         new RandomAccessFile(file.toFile(), "rw")) {
                patch.seek(12L);
                patch.writeLong(count);
            }
        }
    }

    public static final class Reader implements AutoCloseable {
        private final DataInputStream input;
        private long remaining;

        private Reader(Path file) throws IOException {
            DataInputStream stream = new DataInputStream(
                    new BufferedInputStream(
                            Files.newInputStream(Objects.requireNonNull(file)),
                            1024 * 1024
                    )
            );

            try {
                int magic = stream.readInt();
                int version = stream.readInt();
                int size = stream.readInt();
                long count = stream.readLong();

                if (magic != MAGIC
                        || version != VERSION
                        || size != INPUT_SIZE
                        || count < 0) {
                    throw new IOException("Unsupported bank V3 dataset");
                }

                remaining = count;
                input = stream;
            } catch (IOException | RuntimeException failure) {
                stream.close();
                throw failure;
            }
        }

        public BankV3TrainingSample readNext() throws IOException {
            if (remaining == 0) {
                if (input.read() != -1) {
                    throw new IOException("Extra data after bank dataset");
                }

                return null;
            }

            try {
                String id = input.readUTF();
                long userId = input.readLong();
                Instant eventTime = Instant.ofEpochMilli(input.readLong());
                Instant confirmedAt = Instant.ofEpochMilli(input.readLong());
                int label = input.readUnsignedByte();

                float[] values = new float[INPUT_SIZE];

                for (int i = 0; i < values.length; i++) {
                    values[i] = input.readFloat();
                }

                BankV3TrainingSample sample = new BankV3TrainingSample(
                        id,
                        userId,
                        eventTime,
                        confirmedAt,
                        label,
                        values
                );

                remaining--;
                return sample;
            } catch (RuntimeException failure) {
                throw new IOException("Invalid bank V3 dataset record", failure);
            }
        }

        @Override
        public void close() throws IOException {
            input.close();
        }
    }
}