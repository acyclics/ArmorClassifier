import os
import numpy as np
import pathlib

import tensorflow as tf
import cv2

from network_seg import Network
from data_seg import Data
from image_slicer import slice_image


class ArmorDetector():
    """
    Labels:
        - 0 = Armor 1
        - 1 = Armor 2
        - 2 = Armor 3
        - 3 = Armor 4
        - 4 = Red
        - 5 = Blue
        - 6 = Random 
    """
    def __init__(self):
        # Network related
        self.nclasses = nclasses = 1
        img_size = (416, 416)
        anchors = np.array([[9, 18], [18, 18], [18, 31], [26, 24], [35, 32], [35,  48], [55, 51], [84, 92], [157,170]], dtype=np.float32)
        #anchors = self.anchors = np.array([[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59,  119], [116, 90], [156, 198], [373,326]], dtype=np.float32)
        self.data_to_labels = data_to_labels = {
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
            'red': 5, 'blue': 6, 'random': 7
        }
        self.labels_to_data = {
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
            5: 'red', 6: 'blue', 7: 'random'
        }
        self.network = Network(img_size=img_size)
        self.network.call_build()
        self.batch_size = 1
        self.dataset = Data(self.batch_size, nclasses, data_to_labels, anchors, img_wh=img_size)
        # Training related
        self.epochs_per_save = 10
        self.save_path = os.path.join(".", "model", "armorDetectorModel_seg")

    def restore(self):
        self.network.load_weights(self.save_path)

    def train(self, epochs):
        # Restore if save exist
        #if os.path.exists(self.save_path + ".index"):
        #    print("Restoring network")
        #    self.network.load_weights(self.save_path)
        # Start training
        for e in range(int(epochs)):
            img_batch, y_true_13_batch, y_true_26_batch, y_true_52_batch = self.dataset.get_batched_groundTruth()
            groundTruths = [y_true_13_batch, y_true_26_batch, y_true_52_batch]
            with tf.GradientTape() as tape:
                losses = self.network.loss(img_batch, groundTruths)
            netVars = self.network.trainable_variables
            #netVars = [netVar for netVar in netVars if "dark_net" not in netVar.name]
            grads = tape.gradient(losses[0], netVars)
            grads, _grad_norm = tf.clip_by_global_norm(grads, 5)
            grads_and_var = zip(grads, netVars)
            self.network.optimizer.apply_gradients(grads_and_var)
            print(f"Epoch: {e}, Loss: {losses[0]}")
            if (e + 1) % self.epochs_per_save == 0:
                self.network.save_weights(self.save_path)

    def train_hsv_bw(self, epochs):
        # Restore if save exist
        #if os.path.exists(self.save_path + ".index"):
        #    print("Restoring network")
        #    self.network.load_weights(self.save_path)
        # Start training
        for e in range(int(epochs)):
            losses = []
            gradients = []
            for _ in range(20):
                img_batch, groundTruths, _ = self.dataset.get_batched_groundTruth()
                with tf.GradientTape() as tape:
                    loss = self.network.loss(img_batch, groundTruths)
                netVars = self.network.trainable_variables
                gradient = tape.gradient(loss, netVars)
                gradients.append(gradient)
                losses.append(loss)
            #netVars = [netVar for netVar in netVars if "dark_net" not in netVar.name]
            netVars = self.network.trainable_variables
            gradients = [sum([gradient[idx] for gradient in gradients if gradient[idx] is not None]) * (1.0 / len(gradients)) for idx in range(len(gradients[0]))]
            grads, _grad_norm = tf.clip_by_global_norm(gradients, 5)
            grads_and_var = zip(grads, netVars)
            self.network.optimizer.apply_gradients(grads_and_var)
            print(f"Epoch: {e}, Loss: {np.mean(losses)}")
            if (e + 1) % self.epochs_per_save == 0:
                self.network.save_weights(self.save_path)

    def visual_eval_rescaled(self):
        images, boxes = self.dataset.get_visual_eval_rescaled()
        for image, box in zip(images, boxes):
            #image = image * 255.0
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for b in box:
                image = cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (36,255,12), 1)
            cv2.imshow("Image", image)
            cv2.waitKey(0)

    def actual_eval(self):
        self.network.load_weights(self.save_path)
        for _ in range(100):
            # Get groundtruth
            image, y_true_13, y_true_26, y_true_52, gt_img = self.dataset.get_groundTruth(self.dataset.sample_gt_idx(),
                                                      [self.dataset.img_wh[0], self.dataset.img_wh[1]],
                                                      self.dataset.anchors)
            # Eval
            boxes, probs, classes = self(np.asarray([image]))
            # Edit image
            image = cv2.resize(gt_img, self.dataset.og_img_wh)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Draw boxes
            for l, b in zip(classes[0], boxes[0]):
                bx_min, by_min = self.dataset.resize_to_og(b[1], b[0])
                bx_max, by_max = self.dataset.resize_to_og(b[3], b[2])
                #bx_min, by_min = b[0], b[1]
                #bx_max, by_max = b[3], b[2]
                b1 = [by_min, bx_min, by_max, bx_max]
                image = cv2.rectangle(image, (b1[1], b1[0]), (b1[3], b1[2]), (0, 255, 0), 1)
                l = self.labels_to_data[np.argmax(l)]
                cv2.putText(image, str(l), (int(b1[1]), int(b1[0]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
            print("Displaying Image")
            cv2.imshow("Image", image)
            cv2.waitKey(0)
    
    def actual_eval_hsv_bw(self):
        self.network.load_weights(self.save_path)
        for _ in range(100):
            # Get groundtruth
            image, mask, og_image = self.dataset.get_groundTruth(self.dataset.sample_gt_idx(),
                                                      [self.dataset.img_wh[0], self.dataset.img_wh[1]],
                                                      self.dataset.anchors)
            # Eval
            mask = np.argmax(mask, axis=-1)[0]
            mask = mask.astype(np.uint8)
            predicted_mask = self.network.predict(np.asarray([image]), False)
            predicted_mask = np.argmax(predicted_mask, axis=-1)[0]
            predicted_mask = predicted_mask.astype(np.uint8)
            #image = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
            #image = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2GRAY)
            image = cv2.resize(og_image, (416, 416))
            # Draw boxes
            masked_image = cv2.bitwise_and(image, image, mask=predicted_mask)
            print("Displaying Image")
            cv2.imshow("Masked image", masked_image)
            cv2.imshow("Original image", image)
            cv2.waitKey(0)

    def run_evaluation_hsv_bw(self):
        self.network.load_weights(self.save_path)
        ''' Load Snapshots '''
        imagedir = os.path.join("..", "data", "evaluation")
        imagedir = pathlib.Path(imagedir)
        image_paths = list(imagedir.glob('*.*'))
        self.image_paths = [str(path) for path in image_paths]
        for image_path in self.image_paths:
            # Get groundtruth
            image = cv2.imread(image_path)
            input_image = cv2.resize(image, (416, 416))
            img = input_image.astype(np.uint8)
            #hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            #mask1 = cv2.inRange(hsv_img, (0, 50, 20), (5, 255, 255))
            #mask2 = cv2.inRange(hsv_img, (175, 50, 20), (180, 255, 255))
            #mask3 = cv2.inRange(hsv_img, (90, 50, 20), (135, 255, 255))
            #mask4 = cv2.inRange(hsv_img, (0, 0, 50), (172, 111, 255))
            #final_mask = mask1 | mask2 | mask3 | mask4
            #final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            #cropped_img = cv2.bitwise_and(img, img, mask=final_mask)
            #cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            #cropped_img = cv2.GaussianBlur(cropped_img, (5, 5), 0)
            cropped_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) / 255.0
            input_image = slice_image(cropped_img)
            #cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_GRAY2BGR)
            #cropped_img = np.stack([cropped_img, cropped_img, cropped_img], axis=-1)
            #input_image = cropped_img / 255.0
            #sobelx = cv2.Sobel(cropped_img, cv2.CV_64F, 1, 0, ksize=5)  # x
            #sobely = cv2.Sobel(cropped_img, cv2.CV_64F, 0, 1, ksize=5)  # y
            #input_image = np.stack([cropped_img, sobelx, sobely], axis=-1)
            #input_image = input_image / 255.0
            # Eval
            boxes, probs, classes = self(np.asarray([input_image]))
            #image = cv2.resize(image, (1280, 720))
            image = cv2.resize(img, (416, 416))
            #cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR)
            cropped_img = cv2.resize(cropped_img, (416, 416))
            # Draw boxes
            for l, b in zip(classes[0], boxes[0]):
                #bx_min, by_min = self.dataset.resize_to_og(b[1], b[0])
                #bx_max, by_max = self.dataset.resize_to_og(b[3], b[2])
                bx_min, by_min = b[0], b[1]
                bx_max, by_max = b[3], b[2]
                b1 = [by_min, bx_min, by_max, bx_max]
                image = cv2.rectangle(image, (int(b1[1]), int(b1[0])), (int(b1[3]), int(b1[2])), (0, 255, 0), 1)
                l = self.labels_to_data[np.argmax(l)]
                cv2.putText(image, str(l), (int(b1[1]), int(b1[0]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
            print("Displaying Image")
            cv2.imshow("Image", image)
            #cv2.imshow("Cropped", cropped_img)
            cv2.waitKey(0)

    def run_evaluation(self):
        self.network.load_weights(self.save_path)
        ''' Load Snapshots '''
        imagedir = os.path.join(".", "data", "evaluation")
        imagedir = pathlib.Path(imagedir)
        image_paths = list(imagedir.glob('*.*'))
        self.image_paths = [str(path) for path in image_paths]
        for image_path in self.image_paths:
            # Get groundtruth
            image = cv2.imread(image_path)
            input_image = cv2.resize(image, (416, 416))
            input_image = cv2.cvtColor(input_image,  cv2.COLOR_BGR2RGB)
            input_image = cv2.cvtColor(input_image,  cv2.COLOR_RGB2GRAY)
            input_image = input_image / 255.0
            input_image = np.stack([input_image, input_image, input_image], axis=-1)
            #image = cv2.resize(image, (1280, 720))
            image = cv2.resize(image, (416, 416))
            # Eval
            boxes, probs, classes = self(np.asarray([input_image]))
            # Draw boxes
            for l, b in zip(classes[0], boxes[0]):
                #bx_min, by_min = self.dataset.resize_to_og(b[1], b[0])
                #bx_max, by_max = self.dataset.resize_to_og(b[3], b[2])
                bx_min, by_min = b[0], b[1]
                bx_max, by_max = b[3], b[2]
                b1 = [by_min, bx_min, by_max, bx_max]
                image = cv2.rectangle(image, (b1[1], b1[0]), (b1[3], b1[2]), (0, 255, 0), 1)
                l = self.labels_to_data[np.argmax(l)]
                cv2.putText(image, str(l), (int(b1[1]), int(b1[0]) - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
            print("Displaying Image")
            cv2.imshow("Image", image)
            cv2.waitKey(0)

    @tf.function
    def __call__(self, inputs, use_dropout=False):
        boxes, confs, classes = self.network.predict(inputs, use_dropout)
        scores = confs * classes
        # Get high scores
        max_scores = tf.math.argmax(scores, axis=2)
        batch_idxs = tf.concat([tf.expand_dims(tf.range(max_scores.shape[0], dtype=max_scores.dtype), axis=-1) for _ in range(max_scores.shape[1])], axis=1)
        boxes_idxs = tf.concat([tf.expand_dims(tf.range(max_scores.shape[1], dtype=max_scores.dtype), axis=0) for _ in range(max_scores.shape[0])], axis=0)
        max_idxs = tf.stack([batch_idxs, boxes_idxs, max_scores], axis=-1)
        scores = tf.gather_nd(scores, max_idxs)
        # Do selection
        selected_boxes, selected_confs, selected_classes = [], [], []
        for idx in range(inputs.shape[0]):
            selected_box_indices = tf.image.non_max_suppression(boxes[idx], scores[idx], max_output_size=5, iou_threshold=0.5, score_threshold=0.1)
            selected_boxes.append(tf.gather(boxes[idx], selected_box_indices))
            selected_confs.append(tf.gather(confs[idx], selected_box_indices))
            selected_classes.append(tf.gather(classes[idx], selected_box_indices))
        return selected_boxes, selected_confs, selected_classes
