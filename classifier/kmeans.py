"""
Method of getting kmeans adapted from https://github.com/wizyoung/YOLOv3_TensorFlow/blob/master/get_kmeans.py
"""
import os
import pathlib
import numpy as np
import cv2


def get_bounding_box(armors_path):
    with open(armors_path, "r") as file:
        boxes = file.readlines()
    boxes = boxes[0].split(',')
    boxes = [int(b) for b in boxes]
    armor1_xmin, armor1_ymin, armor1_xmax, armor1_ymax = boxes[0], boxes[2], boxes[1], boxes[3]
    armor2_xmin, armor2_ymin, armor2_xmax, armor2_ymax = boxes[4], boxes[6], boxes[5], boxes[7]
    armor3_xmin, armor3_ymin, armor3_xmax, armor3_ymax = boxes[8], boxes[10], boxes[9], boxes[11]
    armor4_xmin, armor4_ymin, armor4_xmax, armor4_ymax = boxes[12], boxes[14], boxes[13], boxes[15]
    boxed_armors = [None, None, None, None]
    if armor1_xmin != -1:
        boxed_armors[0] = [armor1_xmin, armor1_ymin, armor1_xmax, armor1_ymax]
    if armor2_xmin != -1:
        boxed_armors[1] = [armor2_xmin, armor2_ymin, armor2_xmax, armor2_ymax]
    if armor3_xmin != -1:
        boxed_armors[2] = [armor3_xmin, armor3_ymin, armor3_xmax, armor3_ymax]
    if armor4_xmin != -1:
        boxed_armors[3] = [armor4_xmin, armor4_ymin, armor4_xmax, armor4_ymax]
    return boxed_armors


def parse_boxes(base_dir, og_img_size, final_img_size):
    """
        Args:
            base_dir: data dir
            og_img_size: original image size (w, h)
            final_img_size: neural network input image size (w, h)
    """
    boxdir = os.path.join(base_dir, "armors")
    boxdir = pathlib.Path(boxdir)
    box_paths = list(boxdir.glob('*'))
    box_paths = [str(path) for path in box_paths]
    parsed_boxes = []
    for box_path in box_paths:
        boxed_armors = get_bounding_box(box_path)
        for boxed_armor in boxed_armors:
            if boxed_armor != None:
                xmin, ymin, xmax, ymax = boxed_armor[0], boxed_armor[1], boxed_armor[2], boxed_armor[3]
                width = xmax - xmin
                height = ymax - ymin
                resize_ratio = min(final_img_size[0] / og_img_size[0], final_img_size[1] / og_img_size[1])
                width *= resize_ratio
                height *= resize_ratio
                if width * height != 0:
                    parsed_boxes.append([width, height])
    return np.asarray(parsed_boxes)


def iou(box, clusters):
    """
    Calculates the Intersection over Union (IoU) between a box and k clusters.
    param:
        box: tuple or array, shifted to the origin (i. e. width and height)
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        numpy array of shape (k, 0) where k is the number of clusters
    """
    x = np.minimum(clusters[:, 0], box[0])
    y = np.minimum(clusters[:, 1], box[1])
    if np.count_nonzero(x == 0) > 0 or np.count_nonzero(y == 0) > 0:
        raise ValueError("Box has no area")
    intersection = x * y
    box_area = box[0] * box[1]
    cluster_area = clusters[:, 0] * clusters[:, 1]
    iou_ = np.true_divide(intersection, box_area + cluster_area - intersection + 1e-10)
    return iou_


def avg_iou(boxes, clusters):
    """
    Calculates the average Intersection over Union (IoU) between a numpy array of boxes and k clusters.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        clusters: numpy array of shape (k, 2) where k is the number of clusters
    return:
        average IoU as a single float
    """
    return np.mean([np.max(iou(boxes[i], clusters)) for i in range(boxes.shape[0])])


def translate_boxes(boxes):
    """
    Translates all the boxes to the origin.
    param:
        boxes: numpy array of shape (r, 4)
    return:
    numpy array of shape (r, 2)
    """
    new_boxes = boxes.copy()
    for row in range(new_boxes.shape[0]):
        new_boxes[row][2] = np.abs(new_boxes[row][2] - new_boxes[row][0])
        new_boxes[row][3] = np.abs(new_boxes[row][3] - new_boxes[row][1])
    return np.delete(new_boxes, [0, 1], axis=1)


def kmeans(boxes, k, dist=np.median):
    """
    Calculates k-means clustering with the Intersection over Union (IoU) metric.
    param:
        boxes: numpy array of shape (r, 2), where r is the number of rows
        k: number of clusters
        dist: distance function
    return:
        numpy array of shape (k, 2)
    """
    rows = boxes.shape[0]
    distances = np.empty((rows, k))
    last_clusters = np.zeros((rows,))
    np.random.seed()
    # the Forgy method will fail if the whole array contains the same rows
    clusters = boxes[np.random.choice(rows, k, replace=False)]
    while True:
        for row in range(rows):
            distances[row] = 1 - iou(boxes[row], clusters)
        nearest_clusters = np.argmin(distances, axis=1)
        if (last_clusters == nearest_clusters).all():
            break
        for cluster in range(k):
            clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)
        last_clusters = nearest_clusters
    return clusters


def get_kmeans(boxes, cluster_num=9):
    anchors = kmeans(boxes, cluster_num)
    ave_iou = avg_iou(boxes, anchors)
    anchors = anchors.astype('int').tolist()
    anchors = sorted(anchors, key=lambda x: x[0] * x[1])
    return anchors, ave_iou


def get_anchors():
    base_dir = "../data"
    og_img_size = [1280, 720]    # (w, h)
    final_img_size = [416, 416]
    boxes = parse_boxes(base_dir, og_img_size, final_img_size)
    anchors, ave_iou = get_kmeans(boxes, 9)
    anchor_string = ''
    for anchor in anchors:
        anchor_string += '{},{}, '.format(anchor[0], anchor[1])
    anchor_string = anchor_string[:-2]
    print('anchors are:')
    print(anchor_string)
    print('the average iou is:')
    print(ave_iou)


get_anchors()
