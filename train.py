import os
import time
import math
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from IPython import display
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential, layers, Model, optimizers, losses


IMAGE_SIZE = 32
NOISE_DIM = 100
BATCH_SIZE = 32
MOMENTUM = 0.8

def make_generator():
    layer_size = 4

    model = Sequential()
    model.add(layers.Dense(layer_size * layer_size * 256, input_dim = NOISE_DIM))
    model.add(layers.BatchNormalization(momentum = MOMENTUM))
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((layer_size, layer_size, 256)))
    assert model.output_shape == (None, layer_size, layer_size, 256)

    while layer_size < IMAGE_SIZE:
        layer_size *= 2
        model.add(layers.UpSampling2D())
        model.add(layers.Conv2D(256, kernel_size = (3, 3), padding = "same"))
        model.add(layers.BatchNormalization(momentum = MOMENTUM))
        model.add(layers.LeakyReLU())
        assert model.output_shape == (None, layer_size, layer_size, 256)

    model.add(layers.Conv2D(3, kernel_size = (3, 3), padding = "same"))
    model.add(layers.Activation("tanh"))
    assert model.output_shape == (None, IMAGE_SIZE, IMAGE_SIZE, 3)

    return model

def generator_loss(fake_out):
    cross_entropy = losses.BinaryCrossentropy(from_logits = True)
    return cross_entropy(tf.ones_like(fake_out), fake_out)

def make_discriminator():
    filter_size = 32
    max_filter = 512
    dropout = 0.25
    image_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

    model = Sequential()
    model.add(layers.Conv2D(filter_size, kernel_size = (3, 3), strides = (2, 2), padding = "same", input_shape = image_shape))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(dropout))

    while filter_size < max_filter:
        filter_size *= 2
        model.add(layers.Conv2D(filter_size, kernel_size = (3, 3), strides = (2, 2), padding = "same"))
        model.add(layers.BatchNormalization(momentum = MOMENTUM))
        model.add(layers.LeakyReLU(alpha = 0.2))
        model.add(layers.Dropout(dropout))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    model.add(layers.Activation("sigmoid"))

    return model

def discriminator_loss(real_out, fake_out):
    cross_entropy = losses.BinaryCrossentropy(from_logits = True)
    real_loss = cross_entropy(tf.ones_like(real_out), real_out)
    fake_loss = cross_entropy(tf.zeros_like(fake_out), fake_out)
    return real_loss + fake_loss

def generate_and_save_images(model, epoch, input):
    pred = model(input, training = False)
    side_size = math.sqrt(pred.shape[0])
    fig = plt.figure(figsize = (side_size, side_size))

    for i in range(pred.shape[0]):
        plt.subplot(side_size, side_size, i + 1)
        plt.imshow(((pred[i] + 1) * 127.5).numpy().astype(np.int32))
        plt.axis("off")

    plt.savefig("./generated_images/current/images_at_epoch_{}.png".format(epoch))
    #plt.show()
    plt.close()

def train(dataset):
    gen_lr = 1.5e-4
    disc_lr = 1.5e-4
    epochs = 100
    num_to_generate = 16

    generator = make_generator()
    discriminator = make_discriminator()
    generator_optimizer = optimizers.Adam(gen_lr)
    discriminator_optimizer = optimizers.Adam(disc_lr)

    checkpoint_dir = './training_checkpoints/current'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                     discriminator_optimizer = discriminator_optimizer,
                                     generator = generator,
                                     discriminator = discriminator)

    seed = tf.random.normal([num_to_generate, NOISE_DIM])

    @tf.function
    def train_step(images):
        noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = generator(noise, training = True)

            real_out = discriminator(images, training = True)
            fake_out = discriminator(generated_images, training = True)

            gen_loss = generator_loss(fake_out)
            disc_loss = discriminator_loss(real_out, fake_out)

        gen_grad = gen_tape.gradient(gen_loss, generator.trainable_variables)
        disc_grad = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gen_grad, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(disc_grad, discriminator.trainable_variables))

    for epoch in range(epochs):
        start_time = time.time()

        for image_batch in tqdm(dataset):
            train_step(image_batch)

        display.clear_output(wait = True)
        generate_and_save_images(generator, epoch + 1, seed)

        if (epoch + 1) % 10 == 0:
            checkpoint.save(file_prefix = checkpoint_prefix)

        print("Epoch {} completed in {} sec".format(epoch + 1, time.time() - start_time))

    display.clear_output(wait = True)
    generate_and_save_images(generator, epochs, seed)

def main():
    training_set = np.load("training_images.npy").astype("float32")
    training_set = tf.data.Dataset.from_tensor_slices(training_set).shuffle(training_set.shape[0]).batch(BATCH_SIZE)
    train(training_set)

if __name__ == "__main__":
    main()
