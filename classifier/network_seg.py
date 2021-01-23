import os

import numpy as np
import tensorflow as tf

from conv2d import Conv2d
from darkNet import DarkNet53
from yolo import YoloBlock


class Network(tf.keras.Model):

    def __init__(self, img_size):
        super(Network, self).__init__()
        # Darknet pre-trained weights
        self.darknet_weights = os.path.join(".", "model", "yolov3.weights")
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.99, epsilon=1e-5)
        # Network
        self.img_size = img_size
        with tf.name_scope("darknet53"):
            self.darknet_1 = DarkNet53()
        self.conv_output = tf.keras.layers.Conv2D(filters=2, kernel_size=1, strides=1)
        self.upsample_1 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=1, strides=2, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                                        kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.upsample_2 = tf.keras.layers.Conv2DTranspose(filters=1024, kernel_size=1, strides=2, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                                        kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.upsample_3 = tf.keras.layers.Conv2DTranspose(filters=1024, kernel_size=1, strides=2, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                                        kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        """
        self.yolo_1 = YoloBlock(256, 512)
        self.conv2d_2_1 = Conv2d(filters=1024, kernel_size=1, strides=1, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.conv2d_2_2 = Conv2d(filters=1024, kernel_size=1, strides=1, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.conv2d_2_3 = Conv2d(filters=1024, kernel_size=1, strides=1, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.conv2d_3 = Conv2d(filters=512, kernel_size=1, strides=1, kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                               bias_initializer=tf.zeros_initializer, normalize=False, use_dropout=False)
        self.conv2d_4 = tf.keras.layers.Conv2DTranspose(filters=512, kernel_size=1, strides=2, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                                        kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.conv2d_5 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=1, strides=2, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                                        kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.conv2d_6 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=1, strides=2, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                                                        kernel_regularizer=tf.keras.regularizers.l2(5e-4))
        self.conv2d_7 = tf.keras.layers.Conv2D(filters=2, kernel_size=1, strides=1)
        """
        
    @tf.function
    def call(self, inputs, training):
        """
        args:
            inputs: images in (height, width) format
        """
        # Forward pass
        route1, route2, route3 = self.darknet_1(inputs, training)
        route1 = self.upsample_1(route1)
        route1 = self.upsample_2(route1)
        route1 = self.upsample_3(route1)
        output = self.conv_output(route1)
        """
        route2 = self.upsample_1(route2)
        route3 = self.upsample_2(route3)
        route3 = self.upsample_3(route3)
        inter1, route1 = self.yolo_1(route1, training)
        output = tf.concat([route1, route2, route3], axis=-1)
        output = self.conv2d_2_1(output, training)
        output = self.conv2d_2_2(output, training)
        output = self.conv2d_2_3(output, training)
        output = self.conv2d_3(output, training)
        output = self.conv2d_4(output)
        output = self.conv2d_5(output)
        output = self.conv2d_6(output)
        output = self.conv2d_7(output)
        """
        return output
    
    @tf.function
    def predict(self, inputs, use_dropout):
        """
        args:
            inputs: images in (height, width) format
        """
        mask = self(inputs, use_dropout)
        mask = tf.nn.softmax(mask, axis=-1)
        return mask

    def focal_loss(self, labels, logits, gamma=2.0, alpha=20.0):
        epsilon = 1.e-9
        labels = tf.dtypes.cast(labels, tf.int64)
        labels = tf.convert_to_tensor(labels, tf.int64)
        logits = tf.convert_to_tensor(logits, tf.float32)
        #num_cls = logits.shape[1]
        model_out = tf.add(logits, epsilon)
        onehot_labels = labels#tf.one_hot(labels, num_cls)
        onehot_labels = tf.dtypes.cast(onehot_labels, tf.float32)
        ce = tf.multiply(onehot_labels, -tf.math.log(model_out))
        weight = tf.multiply(onehot_labels, tf.math.pow(tf.subtract(1., model_out), gamma))
        fl = tf.multiply(alpha, tf.multiply(weight, ce))
        reduced_fl = tf.reduce_max(fl, axis=1)
        return reduced_fl

    @tf.function
    def loss(self, inputs, groundTruths):
        mask = self(inputs, True)
        #loss = tf.nn.softmax_cross_entropy_with_logits(logits=mask, labels=groundTruths)
        #mask = tf.nn.softmax(mask, axis=-1)
        #loss = self.focal_loss(logits=mask, labels=groundTruths)
        #loss = tf.reduce_mean(loss)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=mask,
                                                                              labels=groundTruths))
        return loss
    
    def call_build(self):
        inputs = np.zeros((1, self.img_size[1], self.img_size[0], 1))
        # Build input shapes
        self(inputs, False)
        # Load pretrained darknet weights and freeze it
        #self.darknet_1.load_pretrained_weights(self.darknet_weights)


#nn = Network((416, 416))
#nn.call_build()
