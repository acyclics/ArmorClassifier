import numpy as np
import tensorflow as tf

from conv2d import Conv2d


class Residual_block(tf.keras.Model):

    def __init__(self, nfilters1, nfilters2):
        super(Residual_block, self).__init__()
        self.conv2d_1 = Conv2d(filters=nfilters1, kernel_size=1, strides=1, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.conv2d_2 = Conv2d(filters=nfilters2, kernel_size=3, strides=1, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)

    def call(self, inputs, training):
        shortcut = inputs
        output = self.conv2d_1(inputs, training)
        output = self.conv2d_2(output, training)
        return output + shortcut
        

class DarkNet53(tf.keras.Model):

    def __init__(self):
        super(DarkNet53, self).__init__()
        self.conv2d_1 = Conv2d(filters=32, kernel_size=3, strides=1, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.conv2d_2 = Conv2d(filters=64, kernel_size=3, strides=2, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.residualBlock_3 = [Residual_block(32, 64)]
        self.conv2d_4 = Conv2d(filters=128, kernel_size=3, strides=2, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.residualBlock_5 = [Residual_block(64, 128) for _ in range(2)]
        self.conv2d_6 = Conv2d(filters=256, kernel_size=3, strides=2, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.residualBlock_7 = [Residual_block(128, 256) for _ in range(8)]
        self.conv2d_8 = Conv2d(filters=512, kernel_size=3, strides=2, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.residualBlock_9 = [Residual_block(256, 512) for _ in range(8)]
        self.conv2d_10 = Conv2d(filters=1024, kernel_size=3, strides=2, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.residualBlock_11 = [Residual_block(512, 1024) for _ in range(4)]

    def load_pretrained_weights(self, weights_file):
        with open(weights_file, "rb") as fp:
            np.fromfile(fp, dtype=np.int32, count=5)
            weights = np.fromfile(fp, dtype=np.float32)
        def load_conv_weights(ptr, conv_layer):
            conv_bn_vars = conv_layer.variables
            shape = conv_bn_vars[3].shape
            num_params = np.prod(shape)
            beta = weights[ptr:ptr + num_params].reshape(shape)
            ptr += num_params
            shape = conv_bn_vars[2].shape
            num_params = np.prod(shape)
            gamma = weights[ptr:ptr + num_params].reshape(shape)
            ptr += num_params
            shape = conv_bn_vars[4].shape
            num_params = np.prod(shape)
            mean = weights[ptr:ptr + num_params].reshape(shape)
            ptr += num_params
            shape = conv_bn_vars[5].shape
            num_params = np.prod(shape)
            variance = weights[ptr:ptr + num_params].reshape(shape)
            ptr += num_params
            shape = conv_bn_vars[1].shape
            num_params = np.prod(shape)
            bias = weights[ptr:ptr + num_params].reshape(shape)
            ptr += num_params
            shape = conv_bn_vars[0].shape
            num_params = np.prod(shape)
            kernel = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
            kernel = np.transpose(kernel, (2, 3, 1, 0))
            ptr += num_params
            conv_weights = [kernel, bias, gamma, beta, mean, variance]
            conv_layer.set_weights(conv_weights)
            return ptr
        ptr = 0
        # Load convolution weights
        ptr = load_conv_weights(ptr, self.conv2d_1)
        ptr = load_conv_weights(ptr, self.conv2d_2)
        for residual_block in self.residualBlock_3:
            ptr = load_conv_weights(ptr, residual_block.conv2d_1)
            ptr = load_conv_weights(ptr, residual_block.conv2d_2)
        ptr = load_conv_weights(ptr, self.conv2d_4)
        for residual_block in self.residualBlock_5:
            ptr = load_conv_weights(ptr, residual_block.conv2d_1)
            ptr = load_conv_weights(ptr, residual_block.conv2d_2)
        ptr = load_conv_weights(ptr, self.conv2d_6)
        for residual_block in self.residualBlock_7:
            ptr = load_conv_weights(ptr, residual_block.conv2d_1)
            ptr = load_conv_weights(ptr, residual_block.conv2d_2)
        ptr = load_conv_weights(ptr, self.conv2d_8)
        for residual_block in self.residualBlock_9:
            ptr = load_conv_weights(ptr, residual_block.conv2d_1)
            ptr = load_conv_weights(ptr, residual_block.conv2d_2)
        ptr = load_conv_weights(ptr, self.conv2d_10)
        for residual_block in self.residualBlock_11:
            ptr = load_conv_weights(ptr, residual_block.conv2d_1)
            ptr = load_conv_weights(ptr, residual_block.conv2d_2)

    def call(self, inputs, training):
        output = self.conv2d_1(inputs, training)
        output = self.conv2d_2(output, training)
        for residual_block in self.residualBlock_3:
            output = residual_block(output, training)
        output = self.conv2d_4(output, training)
        for residual_block in self.residualBlock_5:
            output = residual_block(output, training)
        output = self.conv2d_6(output, training)
        for residual_block in self.residualBlock_7:
            output = residual_block(output, training)
        route1 = output
        output = self.conv2d_8(output, training)
        for residual_block in self.residualBlock_9:
            output = residual_block(output, training)
        route2 = output
        output = self.conv2d_10(output, training)
        for residual_block in self.residualBlock_11:
            output = residual_block(output, training)
        return route1, route2, output
