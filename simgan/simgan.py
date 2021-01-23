import os
import numpy as np
import tensorflow as tf
import cv2

from discriminator import Discriminator
from refiner import Refiner
from data import Data
from buffer import Buffer


class SimGan(tf.keras.Model):
    
    def __init__(self):
        super(SimGan, self).__init__()
        # SimGan
        self.discriminator = Discriminator()
        self.refiner = Refiner()
        # Data
        nclasses = 1
        img_size = (416, 416)
        self.sim_batch_size = 1
        self.epochs_per_save = 50
        self.dataset = Data(self.sim_batch_size, nclasses, img_wh=img_size)
        self.save_path = os.path.join(".", "model", "simgan")
        self.buffer = Buffer(10000000)
    
    def refine_image(self, image):
        return np.array(self.refiner(image))
    
    def eval_refine(self):
        self.restore_model()
        for _ in range(100):
            # Get groundtruth
            simulated_image = self.dataset.get_simulated_image()
            original_simulated_image = simulated_image.copy()
            simulated_image = np.array([simulated_image], dtype=np.float32)
            simulated_image = np.expand_dims(simulated_image, axis=-1)
            refined_image = self.refine_image(simulated_image)
            original_simulated_image = original_simulated_image * 255.0
            original_simulated_image = original_simulated_image.astype(np.uint8)
            refined_image = refined_image[0] * 255.0
            refined_image = refined_image.astype(np.uint8)
            cv2.imshow("Simulated image", original_simulated_image)
            cv2.imshow("Refined image", refined_image)
            cv2.waitKey(0)
    
    def discriminate_image(self, image):
        return self.discriminator(image)

    def eval_discriminate(self):
        for _ in range(100):
            refined = np.random.randint(0, 2)
            if refined:
                simulated_image = self.dataset.get_batched_simulated_images()
                simulated_image = np.expand_dims(simulated_image, axis=-1)
                dis_img = self.refine_image(simulated_image)
            else:
                dis_img = self.dataset.get_batched_real_images()
                dis_img = np.expand_dims(dis_img, axis=-1)
            discrimination = self.discriminate_image(dis_img)
            discrimination = np.argmax(discrimination, axis=-1)
            discrimination = np.reshape(discrimination, (discrimination.shape[0], -1))
            discrimination = np.mean(discrimination[0])
            discrimination = True if discrimination >= 0.5 else False
            print(f"Is image refined = {bool(refined)} ; Discriminator guess = {discrimination}")
    
    def eval_real_images(self):
        for _ in range(100):
            real_image = self.dataset.get_real_image()
            cv2.imshow("Real image", real_image)
            cv2.waitKey(0)

    def generate_images(self):
        self.restore_model()
        for img_path, box_path in zip(self.dataset.image_paths, self.dataset.box_paths):
            img = cv2.imread(img_path)
            img = cv2.resize(img, (416, 416))
            boxes = self.dataset.get_bounding_box(box_path)
            gt_boxes = []
            for idx, box in enumerate(boxes):
                if box != False:
                    # Add bounding box for armor
                    gt_boxes.append(box)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for box in gt_boxes:
                box[0] = int(box[0])
                box[1] = int(box[1])
                box[2] = int(box[2])
                box[3] = int(box[3])
                cropped_img = img[box[1]:box[3], box[0]:box[2]]
                old_size = cropped_img.shape[:2]
                desired_size = 416
                delta_w = desired_size - old_size[1]
                delta_h = desired_size - old_size[0]
                top, bottom = delta_h//2, delta_h-(delta_h//2)
                left, right = delta_w//2, delta_w-(delta_w//2)
                cropped_img = cv2.copyMakeBorder(cropped_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=127.5)
                cropped_img = np.array([cropped_img], dtype=np.float32)
                cropped_img = np.expand_dims(cropped_img, axis=-1)
                cropped_img = cropped_img / 255.0
                refined_img = self.refine_image(cropped_img)
                refined_img = refined_img[0, bottom:bottom+old_size[0], left:left+old_size[1], 0] * 255.0
                img[box[1]:box[3], box[0]:box[2]] = refined_img
            cv2.imwrite("../data/refined_snapshots/" + img_path[18:], img)

    def train(self, epochs=100000):
        self.restore_model()
        for e in range(epochs):
            # Train discriminator
            discriminator_losses = []
            for _ in range(25):
                refined = np.random.randint(0, 2)
                if refined:
                    simulated_image = self.dataset.get_batched_simulated_images()
                    simulated_image = np.expand_dims(simulated_image, axis=-1)
                    refined_img = self.refine_image(simulated_image)
                    """
                    prev_refined_img = np.random.randint(0, 2)
                    if prev_refined_img and self.buffer.len > 0:
                        dis_img = self.buffer.get()
                    else:
                        dis_img = refined_img
                    self.buffer.store(refined_img.copy())
                    """
                    dis_img = refined_img
                else:
                    dis_img = self.dataset.get_batched_real_images()
                    dis_img = np.expand_dims(dis_img, axis=-1)
                discriminator_loss = self.discriminator.train(dis_img, refined, self.sim_batch_size)
                discriminator_losses.append(discriminator_loss)
            # Train refiner
            refiner_losses = []
            for _ in range(25):
                simulated_image = self.dataset.get_batched_simulated_images()
                simulated_image = np.expand_dims(simulated_image, axis=-1)
                with tf.GradientTape() as tape:
                    refined_imgs = self.refiner(simulated_image)
                    prob_of_refined = self.discriminator(refined_imgs)
                    prob_of_refined = prob_of_refined[:, :, :, 1]
                    refiner_loss1 = -tf.reduce_sum(tf.math.log(1 - prob_of_refined + 1e-8))
                    refiner_loss2 = tf.reduce_mean(tf.norm(simulated_image - refined_imgs, 1))
                    refiner_loss = refiner_loss1 + 0.001 * refiner_loss2
                netVars = self.refiner.trainable_variables
                grads = tape.gradient(refiner_loss, netVars)
                grads, _grad_norm = tf.clip_by_global_norm(grads, 5)
                grads_and_var = zip(grads, netVars)
                self.refiner.optimizer.apply_gradients(grads_and_var)
                refiner_losses.append(refiner_loss)
            # Report losses
            total_loss = np.mean(discriminator_losses) + np.mean(refiner_losses)
            print(f"Epoch {e} ; Total loss = {total_loss} ; Discriminator loss = {np.mean(discriminator_losses)} ; Refiner loss = {np.mean(refiner_losses)}")
            #if (e + 1) % self.epochs_per_save == 0:
            self.save_weights(self.save_path)
    
    def restore_model(self):
        if os.path.exists(self.save_path + ".index"):
            print("Restoring network")
            self.load_weights(self.save_path)


simgan = SimGan()
simgan.generate_images()
