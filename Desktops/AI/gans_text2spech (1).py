import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM, Embedding, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Dữ liệu mẫu
texts = [
    "Hello, how are you?",
    "What's the weather like today?",
    "I'm learning about GANs.",
    "This is a text-to-speech example."
]

# Tạo tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(texts)

# Chuyển đổi văn bản thành chuỗi số
sequences = tokenizer.texts_to_sequences(texts)

# Độ dài chuỗi tối đa
max_sequence_length = max(len(seq) for seq in sequences)

# Chuẩn bị dữ liệu huấn luyện
sequences_padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Xây dựng mô hình Generator
latent_dim = 100
num_words = len(tokenizer.word_index) + 1

generator_input = Input(shape=(latent_dim,))
x = Dense(128)(generator_input)
x = Dense(256)(x)
generator_output = Dense(max_sequence_length * num_words, activation='softmax')(x)
generator_output = Reshape((max_sequence_length, num_words))(generator_output)

generator = Model(generator_input, generator_output)

# Xây dựng mô hình Discriminator
discriminator_input = Input(shape=(max_sequence_length, num_words))
x = LSTM(128)(discriminator_input)
discriminator_output = Dense(1, activation='sigmoid')(x)

discriminator = Model(discriminator_input, discriminator_output)

# Kết hợp Generator và Discriminator thành mô hình GAN
discriminator.trainable = False
gan_input = Input(shape=(latent_dim,))
gan_output = discriminator(generator(gan_input))
gan = Model(gan_input, gan_output)

# Compile các mô hình
#optimizer = Adam(lr=0.0002, beta_1=0.5)
optimizer = Adam(learning_rate=0.0002, beta_1=0.5)

discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
gan.compile(loss='binary_crossentropy', optimizer=optimizer)

# Huấn luyện GAN
num_epochs = 1000
batch_size = 16

for epoch in range(num_epochs):
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    fake_texts = generator.predict(noise)
    real_texts = sequences_padded[np.random.randint(0, len(sequences_padded), batch_size)]

    # Tạo nhãn cho dữ liệu
    fake_labels = np.zeros((batch_size, 1))
    real_labels = np.ones((batch_size, 1))

    # Huấn luyện Discriminator
    discriminator.trainable = True
    d_loss_fake = discriminator.train_on_batch(fake_texts, fake_labels)
    d_loss_real = discriminator.train_on_batch(real_texts, real_labels)

    # Huấn luyện Generator thông qua GAN
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    gan_labels = np.ones((batch_size, 1))
    discriminator.trainable = False
    g_loss = gan.train_on_batch(noise, gan_labels)

    # In kết quả sau mỗi epoch
    print(f"Epoch {epoch+1}/{num_epochs}, D Loss Real: {d_loss_real[0]}, D Loss Fake: {d_loss_fake[0]}, G Loss: {g_loss}")

# Tạo văn bản từ mô hình Generator
def generate_text(generator, tokenizer, latent_dim):
    noise = np.random.normal(0, 1, (1, latent_dim))
    generated_sequence = generator.predict(noise)
    generated_text = tokenizer.sequences_to_texts(np.argmax(generated_sequence, axis=2))
    return generated_text

# Sử dụng mô hình Generator để tạo văn bản thành giọng nói
generated_text = generate_text(generator, tokenizer, latent_dim)
print("Generated Text:", generated_text)
