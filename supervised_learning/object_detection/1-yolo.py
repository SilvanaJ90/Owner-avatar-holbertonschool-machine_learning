#!/usr/bin/env python3
""" Doc """
import tensorflow.keras as K
import numpy as np


class Yolo:
    """ Doc """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """ The initializes the Yolo class """
        self.class_t = class_t
        self.nms_t = nms_t
        self.model = K.models.load_model(model_path)
        self.anchors = anchors
        with open(classes_path) as f:
            self.class_names = [line.strip() for line in f.readlines()]


    def process_outputs(self, outputs, image_size):
        """ Doc """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for i, output in enumerate(outputs):
            anchors = self.anchors[i]
            g_h, g_w = output.shape[:2]

            t_xy = output[..., :2]
            t_wh = output[..., 2:4]

            obj_prob = output[..., 4]
            class_prob = output[..., 5:]

            c_y = np.arange(g_h).reshape(-1, 1, 1)
            c_x = np.arange(g_w).reshape(1, -1, 1)

            p_wh = anchors * np.exp(t_wh)
            p_wh[..., 0] *= image_size[1] / self.model.input.shape[1]
            p_wh[..., 1] *= image_size[0] / self.model.input.shape[2]

            b_y = (1 / (1 + np.exp(-t_xy[..., 1]))) + c_y
            b_x = (1 / (1 + np.exp(-t_xy[..., 0]))) + c_x
            b_y /= g_h
            b_x /= g_w

            b_h = p_wh[..., 1] / 2
            b_w = p_wh[..., 0] / 2

            x1 = (b_x - b_w) * image_size[1]
            y1 = (b_y - b_h) * image_size[0]
            x2 = (b_x + b_w) * image_size[1]
            y2 = (b_y + b_h) * image_size[0]

            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))
            box_confidences.append(obj_prob[..., np.newaxis])
            box_class_probs.append(class_prob)

        return boxes, box_confidences, box_class_probs
