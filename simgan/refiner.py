import tensorflow as tf
from resnet import Residual_block


class Refiner(tf.keras.layers.Layer):

    def __init__(self):
        super(Refiner, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding="same")
        self.resnet_blocks = []
        for _ in range(4):
            self.resnet_blocks.append(Residual_block(nfilters1=64, nfilters2=64))
        self.conv2 = tf.keras.layers.Conv2D(filters=1, kernel_size=1, strides=1, padding="same")
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    def call(self, inputs, training=True):
        outputs = self.conv1(inputs)
        for rblock in self.resnet_blocks:
            outputs = rblock(outputs, training)
        outputs = self.conv2(outputs)
        outputs = tf.math.sigmoid(outputs)
        return outputs
