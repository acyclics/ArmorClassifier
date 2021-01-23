import tensorflow as tf


class Conv2d(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides, activation=None, kernel_regularizer=None, bias_initializer=None, normalize=False):
        super(Conv2d, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.normalize = normalize
        self.conv2d = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                             kernel_regularizer=kernel_regularizer,
                                             bias_initializer=bias_initializer, padding='valid' if strides > 1 else 'same')
        if normalize:
            self.batch_norm = tf.keras.layers.BatchNormalization()
    
    def fixed_padding(self, inputs):
        pad_total = self.kernel_size - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]], mode='CONSTANT')
        return padded_inputs
    
    def call(self, inputs, training):
        if self.strides > 1:
            inputs = self.fixed_padding(inputs)
        outputs = self.conv2d(inputs)
        if self.normalize:
            outputs = self.batch_norm(outputs, training)
        if self.activation != None:
            outputs = self.activation(outputs)
        return outputs



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
