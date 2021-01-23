import numpy as np
import tensorflow as tf


class Discriminator(tf.keras.layers.Layer):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=96, kernel_size=3, strides=2)
        self.conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2)
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=3, strides=1)
        self.conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2)
        self.conv4 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2)
        self.conv5 = tf.keras.layers.Conv2D(filters=2, kernel_size=3, strides=2)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    def call(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.pool1(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        outputs = tf.nn.softmax(outputs)
        return outputs
    
    def logits(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.pool1(outputs)
        outputs = self.conv3(outputs)
        outputs = self.conv4(outputs)
        outputs = self.conv5(outputs)
        return outputs

    def train(self, images, refined, batch_size):
        zeros = np.zeros([batch_size, 11, 11])
        ones = np.ones([batch_size, 11, 11])
        if refined:
            logits = np.stack([zeros, ones], axis=-1)
        else:
            logits = np.stack([ones, zeros], axis=-1)
        with tf.GradientTape() as tape:
            discriminations = self.logits(images)
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=logits, logits=discriminations, axis=-1)
        netVars = self.trainable_variables
        grads = tape.gradient(loss, netVars)
        grads, _grad_norm = tf.clip_by_global_norm(grads, 5)
        grads_and_var = zip(grads, netVars)
        self.optimizer.apply_gradients(grads_and_var)
        return loss
