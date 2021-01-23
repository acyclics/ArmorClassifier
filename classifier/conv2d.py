import tensorflow as tf


class Conv2d(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size, strides, activation=None, kernel_regularizer=None, bias_initializer=None, normalize=False, use_dropout=True):
        super(Conv2d, self).__init__()
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation = activation
        self.normalize = normalize
        self.use_dropout = use_dropout
        self.conv2d = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                                             kernel_regularizer=kernel_regularizer,
                                             bias_initializer=bias_initializer, padding='valid' if strides > 1 else 'same')
        #if use_dropout:
        #    self.do = tf.keras.layers.Dropout(0.2)
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
        #if self.use_dropout:
        #    outputs = self.do(outputs, training)
        if self.normalize:
            outputs = self.batch_norm(outputs, training)
        if self.activation != None:
            outputs = self.activation(outputs)
        return outputs
