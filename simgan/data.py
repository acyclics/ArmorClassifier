import os
import numpy as np
import pathlib
import xmltodict

import cv2
import imgaug as ia
from imgaug import augmenters as iaa


class Data:

    def __init__(self, batch_size, nclasses, og_img_wh=(1280, 720), img_wh=(416, 416)):
        self.og_img_wh = og_img_wh
        self.img_wh = img_wh
        self.load_data()
        self.load_real_data()
        self.batch_size = batch_size
        self.nclasses = nclasses
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
                            iaa.GaussianBlur((0, 0.5)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(1, 3)), # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7
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
                        #iaa.Invert(0.05, per_channel=True), # invert color channels
                        iaa.Add((-5, 5), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        #iaa.AddToHueAndSaturation((-5, 5)), # change hue and saturation
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
        imagedir = pathlib.Path(imagedir)
        image_paths = list(imagedir.glob('*.*'))
        self.image_paths = [str(path) for path in image_paths]
        ''' Load boxes '''
        boxdir = os.path.join(base_dir, "armors")
        boxdir = pathlib.Path(boxdir)
        box_paths = list(boxdir.glob('*'))
        self.box_paths = [str(path) for path in box_paths]
        ''' Load colors '''
        colordir = os.path.join(base_dir, "color")
        colordir = pathlib.Path(colordir)
        color_paths = list(colordir.glob('*'))
        self.color_paths = [str(path) for path in color_paths]
        self.n_gts = len(self.image_paths)

    def sample_gt_idx(self):
        return np.random.randint(0, self.n_gts)

    def load_real_data(self, base_dir="../DJI ROCO/robomaster_Final Tournament"):
        ''' Load Snapshots '''
        imagedir = os.path.join(base_dir, "image")
        imagedir = pathlib.Path(imagedir)
        image_paths = list(imagedir.glob('*.*'))
        self.real_image_paths = [str(path) for path in image_paths]
        self.real_image_paths.sort()
        ''' Load annotations '''
        annotationdir = os.path.join(base_dir, "image_annotation")
        annotationdir = pathlib.Path(annotationdir)
        annotation_paths = list(annotationdir.glob('*'))
        self.real_annotation_paths = [str(path) for path in annotation_paths]
        self.real_annotation_paths.sort()
        self.n_real_gts = len(self.real_image_paths)

    def sample_real_gt_idx(self):
        return np.random.randint(0, self.n_real_gts)

    def get_simulated_image(self):
        gt_boxes = []
        while len(gt_boxes) == 0:
            index = self.sample_gt_idx()
            img_path = self.image_paths[index]
            box_path = self.box_paths[index]
            img = cv2.imread(img_path)
            boxes = self.get_bounding_box_noRescale(box_path)
            for idx, box in enumerate(boxes):
                if box != False:
                    # Add bounding box for armor
                    gt_boxes.append(box)
        #img, gt_boxes = self.alter_images(img, gt_boxes)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        box = gt_boxes[0]
        simulated_img = img[box[1]:box[3], box[0]:box[2]]
        random_size = np.random.randint(25, 300)
        simulated_img = cv2.resize(simulated_img, (random_size, random_size))
        old_size = simulated_img.shape[:2]
        desired_size = 416
        delta_w = desired_size - old_size[1]
        delta_h = desired_size - old_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        simulated_img = cv2.copyMakeBorder(simulated_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=127.5)
        """
        simulated_img = cv2.resize(img, (416, 416))
        """
        return simulated_img / 255.0
    
    def get_real_image(self):
        while True:
            try:
                while True:
                    index = self.sample_real_gt_idx()
                    img_path = self.real_image_paths[index]
                    annotation_path = self.real_annotation_paths[index]
                    real_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    gt_boxes = self.get_real_bounding_box(annotation_path)
                    if gt_boxes != None and real_img.size != 0:
                        break
                real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
                box = gt_boxes[0]
                cropped_real_img = real_img[int(float(box['ymin'])):int(float(box['ymax'])), int(float(box['xmin'])):int(float(box['xmax']))]
                #real_img = cv2.resize(real_img, (416, 416))
                break
            except:
                continue
        old_size = cropped_real_img.shape[:2]
        desired_size = 416
        #real_img = cv2.resize(real_img, (new_size[1], new_size[0]))
        delta_w = desired_size - old_size[1]
        delta_h = desired_size - old_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        real_img = real_img[max(-bottom + int(float(box['ymin'])), 0):min(top + int(float(box['ymax'])), real_img.shape[0]),
                            max(-left + int(float(box['xmin'])), 0):min(right + int(float(box['xmax'])), real_img.shape[1])]
        old_size = real_img.shape[:2]
        desired_size = 416
        delta_w = desired_size - old_size[1]
        delta_h = desired_size - old_size[0]
        top, bottom = delta_h//2, delta_h-(delta_h//2)
        left, right = delta_w//2, delta_w-(delta_w//2)
        real_img = cv2.copyMakeBorder(real_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=127.5)
        """
        index = self.sample_real_gt_idx()
        img_path = self.real_image_paths[index]
        annotation_path = self.real_annotation_paths[index]
        real_img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        real_img = cv2.cvtColor(real_img, cv2.COLOR_BGR2GRAY)
        real_img = cv2.resize(real_img, (416, 416))
        """
        return real_img / 255.0

    def get_batched_simulated_images(self):
        img_batch = []
        for _ in range(self.batch_size):
            img = self.get_simulated_image()
            img_batch.append(img)
        return np.array(img_batch, dtype=np.float32)
    
    def get_batched_real_images(self):
        img_batch = []
        for _ in range(self.batch_size):
            img = self.get_real_image()
            img_batch.append(img)
        return np.array(img_batch, dtype=np.float32)

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
        
    def get_bounding_box(self, armors_path):
        with open(armors_path, "r") as file:
            boxes = file.readlines()
        boxes = boxes[0].split(',')
        boxes = [int(b) for b in boxes]
        armor1_xmin, armor1_ymin, armor1_xmax, armor1_ymax = boxes[0], boxes[2], boxes[1], boxes[3]
        armor2_xmin, armor2_ymin, armor2_xmax, armor2_ymax = boxes[4], boxes[6], boxes[5], boxes[7]
        armor3_xmin, armor3_ymin, armor3_xmax, armor3_ymax = boxes[8], boxes[10], boxes[9], boxes[11]
        armor4_xmin, armor4_ymin, armor4_xmax, armor4_ymax = boxes[12], boxes[14], boxes[13], boxes[15]
        # Check for -1
        boxed_armors = [armor1_xmin != -1, armor2_xmin != -1, armor3_xmin != -1, armor4_xmin != -1]
        # Change coordinate frame
        armor1_cv_max = self.unity_to_cv_coords([armor1_xmax, armor1_ymin])
        armor1_cv_min = self.unity_to_cv_coords([armor1_xmin, armor1_ymax])
        armor1_xmin, armor1_ymin = armor1_cv_min[0], armor1_cv_min[1]
        armor1_xmax, armor1_ymax = armor1_cv_max[0], armor1_cv_max[1]
        armor2_cv_max = self.unity_to_cv_coords([armor2_xmax, armor2_ymin])
        armor2_cv_min = self.unity_to_cv_coords([armor2_xmin, armor2_ymax])
        armor2_xmin, armor2_ymin = armor2_cv_min[0], armor2_cv_min[1]
        armor2_xmax, armor2_ymax = armor2_cv_max[0], armor2_cv_max[1]
        armor3_cv_max = self.unity_to_cv_coords([armor3_xmax, armor3_ymin])
        armor3_cv_min = self.unity_to_cv_coords([armor3_xmin, armor3_ymax])
        armor3_xmin, armor3_ymin = armor3_cv_min[0], armor3_cv_min[1]
        armor3_xmax, armor3_ymax = armor3_cv_max[0], armor3_cv_max[1]
        armor4_cv_max = self.unity_to_cv_coords([armor4_xmax, armor4_ymin])
        armor4_cv_min = self.unity_to_cv_coords([armor4_xmin, armor4_ymax])
        armor4_xmin, armor4_ymin = armor4_cv_min[0], armor4_cv_min[1]
        armor4_xmax, armor4_ymax = armor4_cv_max[0], armor4_cv_max[1]
        # Resize bounding boxes
        armor1_xmin, armor1_ymin = self.resize_to_yolo(armor1_xmin, armor1_ymin)
        armor1_xmax, armor1_ymax = self.resize_to_yolo(armor1_xmax, armor1_ymax)
        armor2_xmin, armor2_ymin = self.resize_to_yolo(armor2_xmin, armor2_ymin)
        armor2_xmax, armor2_ymax = self.resize_to_yolo(armor2_xmax, armor2_ymax)
        armor3_xmin, armor3_ymin = self.resize_to_yolo(armor3_xmin, armor3_ymin)
        armor3_xmax, armor3_ymax = self.resize_to_yolo(armor3_xmax, armor3_ymax)
        armor4_xmin, armor4_ymin = self.resize_to_yolo(armor4_xmin, armor4_ymin)
        armor4_xmax, armor4_ymax = self.resize_to_yolo(armor4_xmax, armor4_ymax)
        # Get boxes
        if boxed_armors[0]:
            boxed_armors[0] = [armor1_xmin, armor1_ymin, armor1_xmax, armor1_ymax]
        if boxed_armors[1]:
            boxed_armors[1] = [armor2_xmin, armor2_ymin, armor2_xmax, armor2_ymax]
        if boxed_armors[2]:
            boxed_armors[2] = [armor3_xmin, armor3_ymin, armor3_xmax, armor3_ymax]
        if boxed_armors[3]:
            boxed_armors[3] = [armor4_xmin, armor4_ymin, armor4_xmax, armor4_ymax]
        return boxed_armors
    
    def get_bounding_box_noRescale(self, armors_path):
        with open(armors_path, "r") as file:
            boxes = file.readlines()
        boxes = boxes[0].split(',')
        boxes = [int(b) for b in boxes]
        armor1_xmin, armor1_ymin, armor1_xmax, armor1_ymax = boxes[0], boxes[2], boxes[1], boxes[3]
        armor2_xmin, armor2_ymin, armor2_xmax, armor2_ymax = boxes[4], boxes[6], boxes[5], boxes[7]
        armor3_xmin, armor3_ymin, armor3_xmax, armor3_ymax = boxes[8], boxes[10], boxes[9], boxes[11]
        armor4_xmin, armor4_ymin, armor4_xmax, armor4_ymax = boxes[12], boxes[14], boxes[13], boxes[15]
        # Check for -1
        boxed_armors = [armor1_xmin != -1, armor2_xmin != -1, armor3_xmin != -1, armor4_xmin != -1]
        # Change coordinate frame
        armor1_cv_max = self.unity_to_cv_coords([armor1_xmax, armor1_ymin])
        armor1_cv_min = self.unity_to_cv_coords([armor1_xmin, armor1_ymax])
        armor1_xmin, armor1_ymin = armor1_cv_min[0], armor1_cv_min[1]
        armor1_xmax, armor1_ymax = armor1_cv_max[0], armor1_cv_max[1]
        armor2_cv_max = self.unity_to_cv_coords([armor2_xmax, armor2_ymin])
        armor2_cv_min = self.unity_to_cv_coords([armor2_xmin, armor2_ymax])
        armor2_xmin, armor2_ymin = armor2_cv_min[0], armor2_cv_min[1]
        armor2_xmax, armor2_ymax = armor2_cv_max[0], armor2_cv_max[1]
        armor3_cv_max = self.unity_to_cv_coords([armor3_xmax, armor3_ymin])
        armor3_cv_min = self.unity_to_cv_coords([armor3_xmin, armor3_ymax])
        armor3_xmin, armor3_ymin = armor3_cv_min[0], armor3_cv_min[1]
        armor3_xmax, armor3_ymax = armor3_cv_max[0], armor3_cv_max[1]
        armor4_cv_max = self.unity_to_cv_coords([armor4_xmax, armor4_ymin])
        armor4_cv_min = self.unity_to_cv_coords([armor4_xmin, armor4_ymax])
        armor4_xmin, armor4_ymin = armor4_cv_min[0], armor4_cv_min[1]
        armor4_xmax, armor4_ymax = armor4_cv_max[0], armor4_cv_max[1]
        # Get boxes
        if boxed_armors[0]:
            boxed_armors[0] = [armor1_xmin, armor1_ymin, armor1_xmax, armor1_ymax]
        if boxed_armors[1]:
            boxed_armors[1] = [armor2_xmin, armor2_ymin, armor2_xmax, armor2_ymax]
        if boxed_armors[2]:
            boxed_armors[2] = [armor3_xmin, armor3_ymin, armor3_xmax, armor3_ymax]
        if boxed_armors[3]:
            boxed_armors[3] = [armor4_xmin, armor4_ymin, armor4_xmax, armor4_ymax]
        return boxed_armors
    
    def get_real_bounding_box(self, real_xml):
        with open(real_xml) as fd:
            doc = xmltodict.parse(fd.read())
            if 'object' in doc['annotation']:
                for obj in doc['annotation']['object']:
                    if not isinstance(obj, str) and obj['name'] == 'armor':
                        bb = obj['bndbox']
                        return [bb]
        return None
    