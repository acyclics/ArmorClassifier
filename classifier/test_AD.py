import numpy as np
import tensorflow as tf
import cv2

from armorDetector_seg import ArmorDetector


def test_AD():
    armorDetector = ArmorDetector()
    armorDetector.actual_eval()


def test_AD_hsv_bw():
    armorDetector = ArmorDetector()
    armorDetector.actual_eval_hsv_bw()


def test_Train():
    armorDetector = ArmorDetector()
    armorDetector.train(1e10)


def test_Train_hsv_bw():
    armorDetector = ArmorDetector()
    armorDetector.train_hsv_bw(1e10)


def test_VisualEval():
    armorDetector = ArmorDetector()
    armorDetector.visual_eval_rescaled()


def test_eval():
    armorDetector = ArmorDetector()
    armorDetector.run_evaluation()


def test_eval_hsv_bw():
    armorDetector = ArmorDetector()
    armorDetector.run_evaluation_hsv_bw()


def test_live():
    armorDetector = ArmorDetector()
    armorDetector.network.call_build()
    armorDetector.network.load_weights(armorDetector.save_path)
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    while True:
        ret, frame = cam.read()
        inputs = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = cv2.resize(inputs, (416, 416)) / 255.0
        boxes, probs, classes = armorDetector(np.asarray([inputs]).astype(np.float32))
        # Draw boxes
        for b in boxes[0]:
            bx_min, by_min = armorDetector.dataset.resize_to_og(b[1], b[0])
            bx_max, by_max = armorDetector.dataset.resize_to_og(b[3], b[2])
            b1 = [by_min, bx_min, by_max, bx_max]
            frame = cv2.rectangle(frame, (b1[1], b1[0]), (b1[3], b1[2]), (0, 255, 0), 1)
        print("Displaying Image")
        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()


#test_AD()
#test_Train()
#test_VisualEval()
#test_live()
#test_eval()
test_Train_hsv_bw()
#test_AD_hsv_bw()
#test_eval_hsv_bw()
