import tensorflow as tf
from conv2d import Conv2d


class YoloBlock(tf.keras.Model):

    def __init__(self, nfilters1, nfilters2):
        super(YoloBlock, self).__init__()
        self.conv2d_1 = Conv2d(filters=nfilters1, kernel_size=1, strides=1, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.conv2d_2 = Conv2d(filters=nfilters2, kernel_size=3, strides=1, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.conv2d_3 = Conv2d(filters=nfilters1, kernel_size=1, strides=1, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.conv2d_4 = Conv2d(filters=nfilters2, kernel_size=3, strides=1, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.conv2d_5 = Conv2d(filters=nfilters1, kernel_size=1, strides=1, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.conv2d_6 = Conv2d(filters=nfilters2, kernel_size=3, strides=1, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
    
    def call(self, inputs, training):
        output = self.conv2d_1(inputs, training)
        output = self.conv2d_2(output, training)
        output = self.conv2d_3(output, training)
        output = self.conv2d_4(output, training)
        output = self.conv2d_5(output, training)
        route = output
        output = self.conv2d_6(output, training)
        return route, output
