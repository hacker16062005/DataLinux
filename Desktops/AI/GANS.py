import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.datasets import mnist

# Load dữ liệu MNIST
(X_train, _), (_, _) = mnist.load_data()

# Chuẩn hóa và chuẩn bị dữ liệu
X_train = X_train.astype('float32') / 255.0

# Xây dựng mô hình Generator
def build_generator(latent_dim):
    model = Sequential()
    model.add(Dense(7*7*128, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((7, 7, 128)))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(1, (7,7), activation='sigmoid', padding='same'))
    return model

# Xây dựng mô hình Discriminator
def build_discriminator(input_shape=(28,28,1)):
    model = Sequential()
    model.add(Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    return model

# Xây dựng GAN
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Xây dựng và compile các mô hình
latent_dim = 100
discriminator = build_discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
generator = build_generator(latent_dim)
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Huấn luyện GAN
def train_gan(generator, discriminator, gan, X_train, latent_dim, n_epochs=100, batch_size=128):
    for epoch in range(n_epochs):
        # Chọn ngẫu nhiên một lượng dữ liệu từ tập huấn luyện
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        real_images = X_train[idx]

        # Tạo các vector ngẫu nhiên
        noise = np.random.normal(0, 1, (batch_size, latent_dim))

        # Tạo ảnh giả từ vector ngẫu nhiên bằng Generator
        generated_images = generator.predict(noise)

        # Kết hợp ảnh thật và ảnh giả thành một tập dữ liệu đầu vào cho Discriminator
        X = np.concatenate([real_images, generated_images.reshape(batch_size, 28, 28, 1)])

        # Tạo nhãn cho dữ liệu đầu vào của Discriminator
        y_dis = np.zeros(2*batch_size)
        y_dis[:batch_size] = 0.9

        # Huấn luyện Discriminator
        discriminator.trainable = True
        d_loss = discriminator.train_on_batch(X, y_dis)

        # Tạo các vector ngẫu nhiên mới
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        y_gen = np.ones(batch_size)

        # Đóng băng Discriminator và huấn luyện GAN
        discriminator.trainable = False
        g_loss = gan.train_on_batch(noise, y_gen)

        # In ra kết quả sau mỗi epoch
        print(f"Epoch {epoch+1}/{n_epochs}, Discriminator Loss: {d_loss[0]}, Generator Loss: {g_loss}")

# Huấn luyện GAN
train_gan(generator, discriminator, gan, X_train, latent_dim, n_epochs=100, batch_size=128)

# Hiển thị và lưu các ảnh được tạo ra
def save_generated_images(generator, latent_dim, n_images=10):
    noise = np.random.normal(0, 1, (n_images, latent_dim))
    generated_images = generator.predict(noise)
    for i in range(n_images):
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
        plt.savefig(f'generated_image_{i}.png')
        plt.show()

# Lưu và hiển thị các ảnh được tạo ra
save_generated_images(generator, latent_dim, n_images=10)
