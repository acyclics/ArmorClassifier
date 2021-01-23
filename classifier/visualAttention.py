import tensorflow as tf


class GaussMask(tf.keras.layers.Layer):

    def __init__(self):
        super(GaussMask, self).__init__()
        self.mu = tf.Variable(initial_value=tf.zeros_initializer((1, 1, 1)), trainable=True)
        self.sigma = tf.Variable(initial_value=tf.keras.initializers.GlorotNormal((1)), trainable=True)
        self.shift = tf.Variable(initial_value=tf.keras.initializers.GlorotNormal((1)), trainable=True)
    
    def call(self, inputs, rows, cols):
        r = tf.to_float(tf.reshape(tf.range(rows), (1, 1, rows)))
        c = tf.to_float(tf.reshape(tf.range(cols), (1, cols, 1)))
        centres = self.mu + r * self.shift
        column_centres = c - centres
        mask = tf.exp(-.5 * tf.square(column_centres / self.sigma))
        normalised_mask = mask / (tf.reduce_sum(mask, 1, keep_dims=True) + 1e-9)
        return normalised_mask


class GaussAttention(tf.keras.Model):

    def __init__(self):
        super(GaussAttention, self).__init__()
        self.h = tf.keras.layers.Dense()
        self.gaussMask_x = GaussMask()
        self.gaussMask_y = GaussMask()
    
    def call(self, inputs):
        H, W = tf.shape(inputs)[1:3]
        Ay = self.gaussMask_x(inputs, H, H)
        Ax = self.gaussMask_y(inputs, W, W)
        glimpse = tf.matmul(tf.matmul(Ay, inputs, adjoint_a=True), Ax)
        return glimpse


class SoftAttention(tf.keras.Model):

    def __init__(self, rows, cols):
        super(SoftAttention, self).__init__()
        self.mask = tf.Variable(initial_value=tf.keras.initializers.GlorotNormal((rows, cols)), trainable=True)
