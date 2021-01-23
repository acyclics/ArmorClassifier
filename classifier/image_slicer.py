import numpy as np
import cv2


def slice_image(img):
    """
        image should be (h, w)
        range should be 0 - 255
    """
    img = img.copy() * 255.0
    img = img / 15.0
    img = img.astype(np.int32)
    img = img * 15
    sliced_image = (np.arange(17) == img[..., None]).astype(int)
    sliced_image = sliced_image.astype(np.float32)
    return sliced_image
