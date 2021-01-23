import os
import numpy as np
import pathlib

import cv2
import imgaug as ia
from image_slicer import slice_image
from imgaug import augmenters as iaa


class Data:

    def __init__(self, batch_size, nclasses, data_to_labels, anchors, og_img_wh=(1280, 720), img_wh=(416, 416)):
        self.og_img_wh = og_img_wh
        self.img_wh = img_wh
        self.load_data()
        self.batch_size = batch_size
        self.nclasses = nclasses
        self.data_to_labels = data_to_labels
        self.anchors = anchors
        # Set-up image augmentation
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        self.seq = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5), # horizontally flip 50% of all images
                iaa.Flipud(0.2), # vertically flip 20% of all images
                # crop images by -5% to 10% of their height/width
                sometimes(iaa.CropAndPad(
                    percent=(-0.05, 0.1),
                    pad_mode=ia.ALL,
                    pad_cval=(0, 255)
                )),
                sometimes(iaa.Affine(
                    scale={"x": (0.5, 1.2), "y": (0.5, 1.2)}, # scale images to 50-120% of their size, individually per axis
                    translate_percent={"x": (-0.4, 0.4), "y": (-0.4, 0.4)}, # translate by -40 to +40 percent (per axis)
                    rotate=(-45, 45), # rotate by -45 to +45 degrees
                    shear=(-16, 16), # shear by -16 to +16 degrees
                    order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                    [
                        #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(1, 7)), # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(1, 7)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        iaa.Sharpen(alpha=(0.5, 1.0), lightness=(0.8, 1.2)), # sharpen images
                        iaa.Emboss(alpha=(0.5, 1.0), strength=(1.0, 1.3)), # emboss images
                        # search either for all edges or for directed edges,
                        # blend the result with the original image using a blobby mask
                        iaa.SimplexNoiseAlpha(iaa.OneOf([
                            iaa.EdgeDetect(alpha=(0.5, 1.0)),
                            iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.5, 1.0)),
                        ])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                            iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                        ]),
                        iaa.Invert(0.05, per_channel=True), # invert color channels
                        iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.AddToHueAndSaturation((-5, 5)), # change hue and saturation
                        # either change the brightness of the whole image (sometimes
                        # per channel) or change the brightness of subareas
                        iaa.OneOf([
                            iaa.Multiply((0.5, 1.0), per_channel=0.5),
                            iaa.FrequencyNoiseAlpha(
                                exponent=(-4, 0),
                                first=iaa.Multiply((0.5, 1.5), per_channel=True),
                                second=iaa.LinearContrast((0.5, 2.0))
                            )
                        ]),
                        iaa.LinearContrast((0.8, 1.0), per_channel=0.5), # improve or worsen the contrast
                        iaa.Grayscale(alpha=(0.8, 1.0)),
                        sometimes(iaa.ElasticTransformation(alpha=(0.5, 1.5), sigma=0.25)), # move pixels locally around (with random strengths)
                        sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )
    
    def load_data(self, base_dir="../data"):
        ''' Load Snapshots '''
        imagedir = os.path.join(base_dir, "snapshots")
        #imagedir = os.path.join(base_dir, "refined_snapshots")
        imagedir = pathlib.Path(imagedir)
        image_paths = list(imagedir.glob('*.*'))
        self.image_paths = [str(path) for path in image_paths]
        ''' Load Snapshots '''
        maskdir = os.path.join(base_dir, "masks")
        maskdir = pathlib.Path(maskdir)
        mask_paths = list(maskdir.glob('*.*'))
        self.mask_paths = [str(path) for path in mask_paths]
        self.n_gts = len(self.mask_paths)

    def sample_gt_idx(self):
        return np.random.randint(0, self.n_gts)

    def process_box(self, boxes, labels, img_size, anchors):
        '''
        Adapted from https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/utils/data_utils.py

        Generate the y_true label, i.e. the ground truth feature_maps in 3 different scales.
        params:
            boxes: [N, 5] shape, float32 dtype. `x_min, y_min, x_max, y_mix, mixup_weight`.
            labels: [N] shape, int64 dtype.
            anchors: [9, 4] shape, float32 dtype.
        '''
        anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        # convert boxes form:
        # shape: [N, 2]
        # (x_center, y_center)
        box_centers = (boxes[:, 0:2] + boxes[:, 2:4]) / 2
        # (width, height)
        box_sizes = boxes[:, 2:4] - boxes[:, 0:2]
        # [13, 13, 3, 5+num_class+1] `5` means coords and labels. `1` means mix up weight. 
        y_true_13 = np.zeros((img_size[1] // 32, img_size[0] // 32, 3, 6 + self.nclasses), np.float32)
        y_true_26 = np.zeros((img_size[1] // 16, img_size[0] // 16, 3, 6 + self.nclasses), np.float32)
        y_true_52 = np.zeros((img_size[1] // 8, img_size[0] // 8, 3, 6 + self.nclasses), np.float32)
        # mix up weight default to 1.
        y_true_13[..., -1] = 1.
        y_true_26[..., -1] = 1.
        y_true_52[..., -1] = 1.
        y_true = [y_true_13, y_true_26, y_true_52]
        # [N, 1, 2]
        box_sizes = np.expand_dims(box_sizes, 1)
        # broadcast tricks
        # [N, 1, 2] & [9, 2] ==> [N, 9, 2]
        mins = np.maximum(- box_sizes / 2, - anchors / 2)
        maxs = np.minimum(box_sizes / 2, anchors / 2)
        # [N, 9, 2]
        whs = maxs - mins
        # [N, 9]
        iou = (whs[:, :, 0] * whs[:, :, 1]) / (
                    box_sizes[:, :, 0] * box_sizes[:, :, 1] + anchors[:, 0] * anchors[:, 1] - whs[:, :, 0] * whs[:, :,
                                                                                                            1] + 1e-10)
        # [N]
        best_match_idx = np.argmax(iou, axis=1)
        ratio_dict = {1.0: 8.0, 2.0: 16.0, 3.0: 32.0}
        for i, idx in enumerate(best_match_idx):
            # idx: 0,1,2 ==> 2; 3,4,5 ==> 1; 6,7,8 ==> 0
            feature_map_group = 2 - idx // 3
            # scale ratio: 0,1,2 ==> 8; 3,4,5 ==> 16; 6,7,8 ==> 32
            ratio = ratio_dict[np.ceil((idx + 1) / 3.0)]
            x = int(np.floor(box_centers[i, 0] / ratio))
            y = int(np.floor(box_centers[i, 1] / ratio))
            k = anchors_mask[feature_map_group].index(idx)
            c = labels[i]
            y_true[feature_map_group][y, x, k, :2] = box_centers[i]
            y_true[feature_map_group][y, x, k, 2:4] = box_sizes[i]
            y_true[feature_map_group][y, x, k, 4] = 1.0
            y_true[feature_map_group][y, x, k, 5 + c] = 1.0
            y_true[feature_map_group][y, x, k, -1] = boxes[i, -1]
        return y_true

    def get_groundTruth(self, index, img_size, anchors):
        """
        args:
            index: random sample from all data
            img_size: image size in (width, height)
            anchors: (width, height) of anchors
        """
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_size[0], img_size[1]))
        og_img = img.copy()
        """
        anchors = np.asarray(anchors, dtype=np.float32)
        while True:
            altered_img, altered_gt_boxes = self.alter_images(img, gt_boxes)
            altered_gt_boxes = np.asarray(altered_gt_boxes, dtype=np.float32)
            altered_gt_boxes = np.reshape(altered_gt_boxes, [altered_gt_boxes.shape[0], 4])
            altered_gt_boxes = np.concatenate((altered_gt_boxes, np.full(shape=(altered_gt_boxes.shape[0], 1), fill_value=1., dtype=np.float32)), axis=-1)
            try:
                y_true_13, y_true_26, y_true_52 = self.process_box(altered_gt_boxes, gt_labels, img_size, anchors)
            except:
                continue
            img = altered_img
            break
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.Canny(img, 100, 200, 3, L2gradient=True)
        #img = slice_image(img)
        mask = self.get_mask(img, mask_path)
        img = np.expand_dims(img, axis=-1)
        return img, mask, og_img
    
    def get_batched_groundTruth(self):
        img_batch, mask_batch, og_img_batch = [], [], []
        for _ in range(self.batch_size):
            img, mask, og_img = self.get_groundTruth(self.sample_gt_idx(), [self.img_wh[0], self.img_wh[1]], self.anchors)
            img_batch.append(img)
            mask_batch.append(mask)
            og_img_batch.append(og_img)
        return np.asarray(img_batch).astype(np.float32), np.asarray(mask_batch).astype(np.int32), np.asarray(og_img_batch)

    def get_visual_eval(self):
        images, boxes, labels = [], [], []
        for idx in range(self.n_gts):
            img_path = self.image_paths[idx]
            box_path = self.box_paths[idx]
            color_path = self.color_paths[idx]
            img = cv2.imread(img_path)
            bboxs = self.get_bounding_box_noRescale(box_path)
            color = self.get_colors(color_path)
            gt_boxes = []
            gt_labels = []
            for bx, bb in enumerate(bboxs):
                if bb != False:
                    gt_boxes.append(bb)
                    gt_labels.append("Armor " + str(bx + 1) + ", Color: " + color)
            if len(gt_boxes) == 0:
                continue
            images.append(img)
            boxes.append(gt_boxes)
            labels.append(gt_labels)
        return images, boxes, labels

    def get_visual_eval_rescaled(self):
        images, boxes, labels = [], [], []
        for idx in range(min(100, self.n_gts)):
            img_path = self.image_paths[idx]
            box_path = self.box_paths[idx]
            img = cv2.imread(img_path)
            img = cv2.resize(img, self.img_wh)
            gt_boxes = self.get_bounding_box(box_path)
            if len(gt_boxes) == 0:
                continue
            img, gt_boxes = self.alter_images(img, gt_boxes)
            images.append(img)
            boxes.append(gt_boxes)
        return images, boxes

    def unity_to_cv_coords(self, unity_coord):
        return [unity_coord[0], self.og_img_wh[1] - unity_coord[1] - 1]

    def resize_to_yolo(self, xcoord, ycoord):
        xcoord = xcoord / self.og_img_wh[0] * self.img_wh[0]
        ycoord = ycoord / self.og_img_wh[1] * self.img_wh[1]
        return xcoord, ycoord

    def resize_to_og(self, xcoord, ycoord):
        xcoord = xcoord / self.img_wh[0] * self.og_img_wh[0]
        ycoord = ycoord / self.img_wh[1] * self.og_img_wh[1]
        return xcoord, ycoord

    def alter_images(self, image, bounding_boxes):
        # Image must be RGB and with max 255.0
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.img_wh)
        bbs = [ia.BoundingBox(x1=bounding_box[0], y1=bounding_box[1], x2=bounding_box[2], y2=bounding_box[3]) for bounding_box in bounding_boxes]
        # Make white number vary in grayness
        """
        for bounding_box in bounding_boxes:
            for pix in range(int(bounding_box[0]), int(bounding_box[2])):
                for piy in range(int(bounding_box[1]), int(bounding_box[3])):
                    if image[piy, pix, 0] >= 240 and image[piy, pix, 1] >= 240 and image[piy, pix, 2] >= 240:
                        scale_fact = np.random.uniform(0.3, 1.0)
                        image[piy, pix, 0] = 255 * scale_fact
                        image[piy, pix, 1] = 255 * scale_fact
                        image[piy, pix, 2] = 255 * scale_fact
                    else:
                        scale_fact = np.random.uniform(0.3, 2.0)
                        image[piy, pix, 0] = image[piy, pix, 0] * scale_fact
                        image[piy, pix, 1] = image[piy, pix, 1] * scale_fact
                        image[piy, pix, 2] = image[piy, pix, 2] * scale_fact
        """
        # Resize image
        image = image.astype(np.uint8)
        image_aug, bbs_aug = self.seq(images=[image], bounding_boxes=bbs)
        if len(bbs_aug) > 0:
            new_bbs = [[bb_aug.x1, bb_aug.y1, bb_aug.x2, bb_aug.y2] for bb_aug in bbs_aug]
        else:
            new_bbs = []
        image_aug = cv2.resize(image_aug[0], self.img_wh)
        image_aug = image_aug.astype(np.float32)
        # img should be in range 0-1
        image_aug = image_aug / 255.0
        return image_aug, new_bbs
    
    def get_mask(self, img, mask_path):
        mask = cv2.imread(mask_path)
        mask = cv2.resize(mask, self.img_wh)
        """
        one_hot_mask = np.zeros([1, 416, 416, 2])
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if mask[i, j, 0] == 0 and mask[i, j, 1] == 0 and mask[i, j, 2] == 255:
                    one_hot_mask[0, i, j, 1] = 1.0
                else:
                    one_hot_mask[0, i, j, 0] = 1.0
        """
        one_hot_mask = np.zeros([416, 416])
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                if img[i, j] == 1:
                    if mask[i, j, 0] == 0 and mask[i, j, 1] == 0 and mask[i, j, 2] == 255:
                        one_hot_mask[i, j] = 1
                    else:
                        one_hot_mask[i, j] = 0
        return one_hot_mask.astype(np.int32)
    
    def get_image(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, self.img_wh)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # img should be in range 0-1
        img = img / 255.0
        return img
    