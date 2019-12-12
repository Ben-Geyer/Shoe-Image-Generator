import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


@np.vectorize
def get_y(x):
    return 10 + x ** 2

def sample_data(n = 10000, scale = 100):
    x = np.random.random((n,)) - 0.5
    x *= scale
    return np.vstack((x, get_y(x))).T

def generator(Z, sizes = [16, 16], reuse = False):
    with tf.variable_scope("generator", reuse = reuse):
        layer = Z

        for size in sizes:
            layer = tf.layers.dense(layer, size, activation = tf.nn.leaky_relu)

        out = tf.layers.dense(layer, 2)

    return out

def discriminator(X, sizes = [16, 16], reuse = False):
    with tf.variable_scope("discriminator", reuse = reuse):
        layer = X

        for size in sizes:
            layer = tf.layers.dense(layer, size, activation = tf.nn.leaky_relu)

        layer = tf.layers.dense(layer, 2)
        out = tf.layers.dense(layer, 1)

    return out, layer

def train(n_iter = 10000, d_steps = 10, g_steps = 10, lr = 0.001):
    # Real samples
    X = tf.placeholder(tf.float32, [None, 2])
    # Noise (fake) samples
    Z = tf.placeholder(tf.float32, [None, 2])

    generated = generator(Z)
    r_logits, r_rep = discriminator(X)
    g_logits, g_rep = discriminator(generated, reuse = True)

    def cross_entropy(logits, label_func):
        return tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = label_func(logits))

    disc_loss = tf.reduce_mean(cross_entropy(r_logits, tf.ones_like)) + cross_entropy(g_logits, tf.zeros_like)
    gen_loss = tf.reduce_mean(cross_entropy(g_logits, tf.ones_like))

    gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "generator")
    disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = "discriminator")

    gen_step = tf.train.RMSPropOptimizer(learning_rate = lr).minimize(gen_loss, var_list = gen_vars)
    disc_step = tf.train.RMSPropOptimizer(learning_rate = lr).minimize(disc_loss, var_list = disc_vars)

    session = tf.Session()
    tf.global_variables_initializer().run(session = session)

    batch_size = 256
    x_plot = sample_data()

    for i in range(n_iter + 1):
        X_batch = sample_data(n = batch_size)
        Z_batch = np.random.uniform(-1., 1., size = [batch_size, 2])

        for _ in range(d_steps):
            session.run([disc_step, disc_loss], feed_dict = {X: X_batch, Z: Z_batch})
        session.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})

        for _ in range(g_steps):
            session.run([gen_step, gen_loss], feed_dict={Z: Z_batch})
        session.run([r_rep, g_rep], feed_dict={X: X_batch, Z: Z_batch})

        if i % 10 == 0:
            print("Iteration: {}".format(i))
        if i % 1000 == 0:
            g_plot = session.run(generated, feed_dict = {Z: Z_batch})
            plt.plot(x_plot[:,0], x_plot[:,1], "bo")
            plt.plot(g_plot[:,0], g_plot[:,1], "ro")
            plt.show()

train()
