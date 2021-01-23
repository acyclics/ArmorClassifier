"""
The following builds were used as reference:
    1. https://github.com/wizyoung/YOLOv3_TensorFlow
"""
import os

import numpy as np
import tensorflow as tf

from conv2d import Conv2d
from darkNet import DarkNet53
from yolo import YoloBlock


class Network(tf.keras.Model):

    def __init__(self, nclasses, img_size, anchors):
        super(Network, self).__init__()
        # Darknet pre-trained weights
        self.darknet_weights = os.path.join(".", "model", "yolov3.weights")
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.99, epsilon=1e-5)
        # Network
        self.nclasses = nclasses
        self.img_size = img_size
        self.anchors = anchors
        with tf.name_scope("darknet53"):
            self.darknet_1 = DarkNet53()
        self.yolo_2 = YoloBlock(512, 1024)
        self.conv2d_2_1 = Conv2d(filters=1024, kernel_size=1, strides=1, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.conv2d_2_2 = Conv2d(filters=1024, kernel_size=1, strides=1, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.conv2d_2_3 = Conv2d(filters=1024, kernel_size=1, strides=1, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.conv2d_3 = Conv2d(filters=3 * (5 + nclasses), kernel_size=1, strides=1, kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                               bias_initializer=tf.zeros_initializer, normalize=False, use_dropout=False)
        self.conv2d_4 = Conv2d(filters=256, kernel_size=1, strides=1, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.upsample_5 = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')
        self.yolo_6 = YoloBlock(256, 512)
        self.conv2d_6_1 = Conv2d(filters=512, kernel_size=1, strides=1, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.conv2d_6_2 = Conv2d(filters=512, kernel_size=1, strides=1, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.conv2d_6_3 = Conv2d(filters=512, kernel_size=1, strides=1, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.conv2d_7 = Conv2d(filters=3 * (5 + nclasses), kernel_size=1, strides=1, kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                               bias_initializer=tf.zeros_initializer, normalize=False, use_dropout=False)
        self.conv2d_8 = Conv2d(filters=128, kernel_size=1, strides=1, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.upsample_9 = tf.keras.layers.UpSampling2D((2, 2), interpolation='bilinear')
        self.yolo_10 = YoloBlock(128, 256)
        self.conv2d_10_1 = Conv2d(filters=256, kernel_size=1, strides=1, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.conv2d_10_2 = Conv2d(filters=256, kernel_size=1, strides=1, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.conv2d_10_3 = Conv2d(filters=256, kernel_size=1, strides=1, activation=lambda x: tf.nn.leaky_relu(x, alpha=0.1),
                               kernel_regularizer=tf.keras.regularizers.l2(5e-4), normalize=True)
        self.conv2d_11 = Conv2d(filters=3 * (5 + nclasses), kernel_size=1, strides=1, kernel_regularizer=tf.keras.regularizers.l2(5e-4),
                                bias_initializer=tf.zeros_initializer, normalize=False, use_dropout=False)

    @tf.function
    def bound_layer(self, f_map, anchors):
        """
        args:
            f_map: single feature map
            anchors: anchors for each feature map
        """
        # (h, w)
        gridDim = tf.shape(f_map)[1:3]
        f_map = tf.reshape(f_map, [-1, gridDim[0], gridDim[1], 3, 5 + self.nclasses])
        box_centers, box_sizes, conf_logits, class_logits = tf.split(f_map, [2, 2, 1, self.nclasses], axis=-1)
        box_centers = tf.math.sigmoid(box_centers)
        gridX = tf.range(gridDim[1], dtype=tf.int32)
        gridY = tf.range(gridDim[0], dtype=tf.int32)
        gridX, gridY = tf.meshgrid(gridX, gridY)
        x_offset = tf.reshape(gridX, (-1, 1))
        y_offset = tf.reshape(gridY, (-1, 1))
        x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
        x_y_offset = tf.dtypes.cast(tf.reshape(x_y_offset, [gridDim[0], gridDim[1], 1, 2]), tf.float32)
        # Shift boxes from their xy location
        box_centers = box_centers + x_y_offset
        # Scale feature-map up to image size
        ratio = tf.dtypes.cast(self.img_size / gridDim, tf.float32)
        box_centers = box_centers * ratio[::-1]
        # (w, h)
        #rescaled_anchors = [(anchor[0] / ratio[1], anchor[1] / ratio[0]) for anchor in anchors]
        rescaled_anchors = [(anchors[idx][0] / ratio[1], anchors[idx][1] / ratio[0]) for idx in range(3)]
        box_sizes = tf.exp(box_sizes) * rescaled_anchors
        box_sizes = box_sizes * ratio[::-1]
        boxes = tf.concat([box_centers, box_sizes], axis=-1)
        return x_y_offset, boxes, conf_logits, class_logits

    @tf.function
    def call(self, inputs, training):
        """
        args:
            inputs: images in (height, width) format
        """
        # Forward pass
        route1, route2, route3 = self.darknet_1(inputs, training)
        inter1, output = self.yolo_2(route3, training)
        output = self.conv2d_2_1(output, training)
        output = self.conv2d_2_2(output, training)
        output = self.conv2d_2_3(output, training)
        fmap_1 = self.conv2d_3(output, training)
        inter1 = self.conv2d_4(inter1, training)
        inter1 = self.upsample_5(inter1)
        concat1 = tf.concat([inter1, route2], axis=3)
        inter2, output = self.yolo_6(concat1, training)
        output = self.conv2d_6_1(output, training)
        output = self.conv2d_6_2(output, training)
        output = self.conv2d_6_3(output, training)
        fmap_2 = self.conv2d_7(output, training)
        inter2 = self.conv2d_8(inter2, training)
        inter2 = self.upsample_9(inter2)
        concat2 = tf.concat([inter2, route1], axis=3)
        _, output = self.yolo_10(concat2, training)
        output = self.conv2d_10_1(output, training)
        output = self.conv2d_10_2(output, training)
        output = self.conv2d_10_3(output, training)
        fmap_3 = self.conv2d_11(output, training)
        return fmap_1, fmap_2, fmap_3
    
    @tf.function
    def predict(self, inputs, use_dropout):
        """
        args:
            inputs: images in (height, width) format
        """
        fmap_1, fmap_2, fmap_3 = self(inputs, use_dropout)
        fmap_anchors = [(fmap_1, self.anchors[6:9]), (fmap_2, self.anchors[3:6]), (fmap_3, self.anchors[0:3])]
        bounds = [self.bound_layer(fmap, anchor) for (fmap, anchor) in fmap_anchors]
        boxes_list, confs_list, classes_list = [], [], []
        for bound in bounds:
            x_y_offset, boxes, conf_logits, class_logits = bound
            gridDim = tf.shape(x_y_offset)[0:2]
            boxes = tf.reshape(boxes, [-1, gridDim[0] * gridDim[1] * 3, 4])
            conf_logits = tf.reshape(conf_logits, [-1, gridDim[0] * gridDim[1] * 3, 1])
            class_logits = tf.reshape(class_logits, [-1, gridDim[0] * gridDim[1] * 3, self.nclasses])
            confs = tf.sigmoid(conf_logits)
            classes = tf.sigmoid(class_logits)
            boxes_list.append(boxes)
            confs_list.append(confs)
            classes_list.append(classes)
        boxes = tf.concat(boxes_list, axis=1)
        confs = tf.concat(confs_list, axis=1)
        classes = tf.concat(classes_list, axis=1)
        center_x, center_y, width, height = tf.split(boxes, [1, 1, 1, 1], axis=-1)
        x_min = center_x - width / 2
        y_min = center_y - height / 2
        x_max = center_x + width / 2
        y_max = center_y + height / 2
        boxes = tf.concat([y_min, x_min, y_max, x_max], axis=-1)
        return boxes, confs, classes
    
    @tf.function
    def compute_loss(self, fmap, groundTruth, anchors):
        gridDim = tf.shape(fmap)[1:3]
        ratio = tf.dtypes.cast(self.img_size / gridDim, tf.float32)
        N = tf.dtypes.cast(tf.shape(fmap)[0], tf.float32)
        x_y_offset, boxes, conf_logits, class_logits = self.bound_layer(fmap, anchors)
        # Mask getting adapted from https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/model.py
        object_mask = groundTruth[..., 4:5]
        ignore_mask = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        def loop_cond(idx, ignore_mask):
            return tf.less(idx, tf.dtypes.cast(N, tf.int32))
        def loop_body(idx, ignore_mask):
            # shape: [13, 13, 3, 4] & [13, 13, 3]  ==>  [V, 4]
            # V: num of true gt box of each image in a batch
            valid_true_boxes = tf.boolean_mask(groundTruth[idx, ..., 0:4], tf.dtypes.cast(object_mask[idx, ..., 0], tf.bool))
            # shape: [13, 13, 3, 4] & [V, 4] ==> [13, 13, 3, V]
            iou = self.iou(boxes[idx], valid_true_boxes)
            # shape: [13, 13, 3]
            best_iou = tf.reduce_max(iou, axis=-1)
            # shape: [13, 13, 3]
            ignore_mask_tmp = tf.dtypes.cast(best_iou < 0.5, tf.float32)
            # finally will be shape: [N, 13, 13, 3]
            ignore_mask = ignore_mask.write(idx, ignore_mask_tmp)
            return idx + 1, ignore_mask
        _, ignore_mask = tf.while_loop(cond=loop_cond, body=loop_body, loop_vars=[0, ignore_mask])
        ignore_mask = ignore_mask.stack()
        # shape: [N, 13, 13, 3, 1]
        ignore_mask = tf.expand_dims(ignore_mask, -1)
        pred_box_xy = boxes[..., 0:2]
        pred_box_wh = boxes[..., 2:4]
        gt_xy = groundTruth[..., 0:2] / ratio[::-1] - x_y_offset
        pred_xy = pred_box_xy / ratio[::-1] - x_y_offset
        gt_tw_th = groundTruth[..., 2:4] / anchors
        pred_tw_th = pred_box_wh / anchors
        # For numerical stability
        gt_tw_th = tf.where(condition=tf.equal(gt_tw_th, 0), x=tf.ones_like(gt_tw_th), y=gt_tw_th)
        pred_tw_th = tf.where(condition=tf.equal(pred_tw_th, 0), x=tf.ones_like(pred_tw_th), y=pred_tw_th)
        gt_tw_th = tf.math.log(tf.clip_by_value(gt_tw_th, 1e-9, 1e9))
        pred_tw_th = tf.math.log(tf.clip_by_value(pred_tw_th, 1e-9, 1e9))
        # Box size punishment:
        box_loss_scale = 2.0 - (groundTruth[..., 2:3] / tf.dtypes.cast(self.img_size[1], tf.float32)) * (groundTruth[..., 3:4] / tf.dtypes.cast(self.img_size[0], tf.float32))
        # Calculate loss
        mix_w = groundTruth[..., -1:]
        xy_loss = tf.reduce_sum(tf.square(gt_xy - pred_xy) * object_mask * box_loss_scale * mix_w) / N
        wh_loss = tf.reduce_sum(tf.square(gt_tw_th - pred_tw_th) * object_mask * box_loss_scale * mix_w) / N
        conf_pos_mask = object_mask
        conf_neg_mask = (1 - object_mask) * ignore_mask
        conf_loss_pos = conf_pos_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=conf_logits)
        conf_loss_neg = conf_neg_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask, logits=conf_logits)
        conf_loss = conf_loss_pos + conf_loss_neg
        conf_loss = tf.reduce_sum(conf_loss * mix_w) / N
        label_target = groundTruth[..., 5:-1]
        class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_target, logits=class_logits) * mix_w
        class_loss = tf.reduce_sum(class_loss) / N
        return xy_loss, wh_loss, conf_loss, class_loss

    @tf.function
    def iou(self, pred, groundTruth):
        pred_xy = pred[..., 0:2]
        pred_wh = pred[..., 2:4]
        pred_xy = tf.expand_dims(pred_xy, -2)
        pred_wh = tf.expand_dims(pred_wh, -2)
        groundTruth_xy = groundTruth[:, 0:2]
        groundTruth_wh = groundTruth[:, 2:4]
        intersect_mins = tf.maximum(pred_xy - pred_wh / 2.0, groundTruth_xy - groundTruth_wh / 2.0)
        intersect_maxs = tf.minimum(pred_xy + pred_wh / 2.0, groundTruth_xy + groundTruth_wh / 2.0)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.0)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        pred_area = pred_wh[..., 0] * pred_wh[..., 1]
        groundTruth_area = groundTruth_wh[..., 0] * groundTruth_wh[..., 1]
        groundTruth_area = tf.expand_dims(groundTruth_area, axis=0)
        iou = intersect_area / (pred_area + groundTruth_area - intersect_area + 1e-10)
        return iou

    @tf.function
    def loss(self, inputs, groundTruths):
        fmaps = self(inputs, True)
        loss_xy, loss_wh, loss_conf, loss_class = 0.0, 0.0, 0.0, 0.0
        all_anchors = [self.anchors[6:9], self.anchors[3:6], self.anchors[0:3]]
        # Compute losses in all 3 scales
        for fmap, groundTruth, anchors in zip(fmaps, groundTruths, all_anchors):
            losses = self.compute_loss(fmap, groundTruth, anchors)
            loss_xy += losses[0]
            loss_wh += losses[1]
            loss_conf += losses[2]
            loss_class += losses[3]
        total_loss = loss_xy + loss_wh + loss_conf + loss_class
        return [total_loss, loss_xy, loss_wh, loss_conf, loss_class]
    
    def call_build(self):
        inputs = np.zeros((1, self.img_size[1], self.img_size[0], 2))
        # Build input shapes
        self(inputs, False)
        # Load pretrained darknet weights and freeze it
        #self.darknet_1.load_pretrained_weights(self.darknet_weights)
       