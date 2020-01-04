import numpy as np
import matplotlib.pyplot as plt
# Uses tensorflow version 1.x
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image


# Hyperparameters
NOISE_SIZE = 100
INPUT_SHAPE = (128, 128, 3)
BATCH_NORM_MOMENTUM = 0.9
LEAKY_RELU_ALPHA = 0.1
LEARNING_RATE = 0.0002
BATCH_SIZE = 16
N_EPOCHS = 200
LABEL_SMOOTHING = 0.1
LABEL_FLIP_PROB = 0.05
NUM_GEN_FILTERS = 256
NUM_DISC_FILTERS = 128

def make_generator():
    input_layer = layers.Input(shape = (NOISE_SIZE,))
    gen_shape = INPUT_SHAPE[0] // 2

    # Fully connected hidden layer
    hidden = layers.Dense(NUM_GEN_FILTERS * gen_shape * gen_shape, activation = "relu")(input_layer)
    hidden = layers.BatchNormalization(momentum = BATCH_NORM_MOMENTUM)(hidden)
    hidden = layers.LeakyReLU(alpha = LEAKY_RELU_ALPHA)(hidden)
    hidden = layers.Reshape((gen_shape, gen_shape, NUM_GEN_FILTERS))(hidden)
    # Output shape (gen_shape, gen_shape, NUM_GEN_FILTERS)

    hidden = layers.Conv2D(NUM_GEN_FILTERS, kernel_size = 5, strides = 1,padding = "same")(hidden)
    hidden = layers.BatchNormalization(momentum = BATCH_NORM_MOMENTUM)(hidden)
    hidden = layers.LeakyReLU(alpha = LEAKY_RELU_ALPHA)(hidden)
    # Output shape (gen_shape, gen_shape, NUM_GEN_FILTERS)

    # Upsampling and convolutional hidden layer
    hidden = layers.Conv2DTranspose(NUM_GEN_FILTERS, kernel_size = 4, strides = 2, padding = "same")(hidden)
    hidden = layers.BatchNormalization(momentum = BATCH_NORM_MOMENTUM)(hidden)
    hidden = layers.LeakyReLU(alpha = LEAKY_RELU_ALPHA)(hidden)
    # Output shape (gen_shape * 2, gen_shape * 2, NUM_GEN_FILTERS)

    hidden = layers.Conv2D(NUM_GEN_FILTERS, kernel_size = 5, strides = 1, padding = "same")(hidden)
    hidden = layers.BatchNormalization(momentum = BATCH_NORM_MOMENTUM)(hidden)
    hidden = layers.LeakyReLU(alpha = LEAKY_RELU_ALPHA)(hidden)
    # Output shape (gen_shape * 2, gen_shape * 2, NUM_GEN_FILTERS)

    hidden = layers.Conv2D(NUM_GEN_FILTERS, kernel_size = 5, strides = 1, padding = "same")(hidden)
    hidden = layers.BatchNormalization(momentum = BATCH_NORM_MOMENTUM)(hidden)
    hidden = layers.LeakyReLU(alpha = LEAKY_RELU_ALPHA)(hidden)
    # Output shape (gen_shape * 2, gen_shape * 2, NUM_GEN_FILTERS)

    hidden = layers.Conv2D(3, kernel_size = 5, strides = 1, padding = "same")(hidden)
    output_layer = layers.Activation("tanh")(hidden)
    # Output shape (gen_shape * 2, gen_shape * 2, 3)

    model = Model(input_layer, output_layer)
    return model

def make_discriminator():
    input_layer = layers.Input(shape = INPUT_SHAPE)

    hidden = layers.Conv2D(NUM_DISC_FILTERS, kernel_size = 3, strides = 1, padding = "same")(input_layer)
    hidden = layers.BatchNormalization(momentum = BATCH_NORM_MOMENTUM)(hidden)
    hidden = layers.LeakyReLU(alpha = LEAKY_RELU_ALPHA)(hidden)
    # Output shape (INPUT_SHAPE[0], INPUT_SHAPE[0], NUM_DISC_FILTERS)

    hidden = layers.Conv2D(NUM_DISC_FILTERS, kernel_size = 4, strides = 2, padding = "same")(hidden)
    hidden = layers.BatchNormalization(momentum = BATCH_NORM_MOMENTUM)(hidden)
    hidden = layers.LeakyReLU(alpha = LEAKY_RELU_ALPHA)(hidden)
    # Output shape (INPUT_SHAPE[0] / 2, INPUT_SHAPE[0] / 2, NUM_DISC_FILTERS)

    hidden = layers.Conv2D(NUM_DISC_FILTERS, kernel_size = 4, strides = 2, padding = "same")(hidden)
    hidden = layers.BatchNormalization(momentum = BATCH_NORM_MOMENTUM)(hidden)
    hidden = layers.LeakyReLU(alpha = LEAKY_RELU_ALPHA)(hidden)
    # Output shape (INPUT_SHAPE[0] / 4, INPUT_SHAPE[0] / 4, NUM_DISC_FILTERS)

    hidden = layers.Conv2D(NUM_DISC_FILTERS, kernel_size = 4, strides = 2, padding = "same")(hidden)
    hidden = layers.BatchNormalization(momentum = BATCH_NORM_MOMENTUM)(hidden)
    hidden = layers.LeakyReLU(alpha = LEAKY_RELU_ALPHA)(hidden)
    # Output shape (INPUT_SHAPE[0] / 8, INPUT_SHAPE[0] / 8, NUM_DISC_FILTERS)

    # Fully connected output layer
    hidden = layers.Flatten()(hidden)
    hidden = layers.Dropout(0.4)(hidden)
    output_layer = layers.Dense(1, activation = "sigmoid")(hidden)
    # Output shape (1,)

    model = Model(input_layer, output_layer)
    return model

def gen_noise(num_samples):
    return np.random.normal(0, 1, size = (num_samples, NOISE_SIZE))

def change_labels(orig_labels):
    smoothing = np.random.uniform(low = 0.0, high = LABEL_SMOOTHING, size = (BATCH_SIZE, 1))
    labels = np.absolute(orig_labels - smoothing)
    flipped_idx = np.random.choice(np.arange(len(labels)), size = int(LABEL_FLIP_PROB * len(labels)))
    labels[flipped_idx] = 1 - labels[flipped_idx]
    return labels

def show_images(generator, epoch):
    # Generate 9 images
    gen_images = generator.predict(gen_noise(9))

    # Plot images
    figure, axes = plt.subplots(3, 3)
    count = 0
    for i in range(3):
      for j in range(3):
        axes[i, j].imshow(image.array_to_img(gen_images[count], scale = True))
        axes[i, j].axis("off")
        count += 1

    plt.savefig("./generated_images/current/images_at_epoch_{}".format(epoch))
    plt.show()
    plt.close()

def make_models():
    # Make discriminator
    discriminator = make_discriminator()
    discriminator.compile(optimizer = Adam(LEARNING_RATE, 0.5), loss = "binary_crossentropy", metrics = ["accuracy"])
    discriminator.trainable = False

    # Make generator
    generator = make_generator()

    # Make GAN
    gan_input = layers.Input(shape = (NOISE_SIZE,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    gan.compile(optimizer = Adam(LEARNING_RATE, 0.5), loss = "binary_crossentropy")

    return gan, discriminator, generator

def train(train_set):
    gan, discriminator, generator = make_models()
    num_batches = train_set.shape[0] // BATCH_SIZE

    for epoch in range(N_EPOCHS):
        total_d_loss = 0.
        total_g_loss = 0.

        for batch_idx in range(num_batches):
            # Get the next batch of real and fake images
            real_images = train_set[(batch_idx * BATCH_SIZE):((batch_idx + 1) * BATCH_SIZE)]
            fake_images = generator.predict(gen_noise(BATCH_SIZE))

            # Smooth and flip labels
            real_labels = change_labels(np.zeros((BATCH_SIZE, 1)))
            fake_labels = change_labels(np.ones((BATCH_SIZE, 1)))

            # Train discriminator on real and generated data
            d_loss_disc = discriminator.train_on_batch(real_images, real_labels)
            d_loss_gen = discriminator.train_on_batch(fake_images, fake_labels)

            # Train generator
            noise_data = gen_noise(BATCH_SIZE)
            g_loss = gan.train_on_batch(noise_data, np.zeros((BATCH_SIZE, 1)))

            total_d_loss += 0.5 * np.add(d_loss_disc, d_loss_gen)[0]
            total_g_loss += g_loss

        if (epoch + 1) % 10 == 0:
            generator.save_weights("./training_checkpoints/epoch_{}_checkpoint".format(epoch + 1))
            print("Saved checkpoint")

        print("Epoch: {}, Generator Loss: {}, Discriminator Loss: {}".format(epoch + 1, total_g_loss / num_batches, total_d_loss / num_batches))
        show_images(generator, epoch)

if __name__ == "__main__":
    # Get training images
    train_set = np.load("./training_images_size128.npy").astype("float32")
    train(train_set)
