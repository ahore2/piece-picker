import itertools as it
import numpy as np
import cv2
import datetime
from shapely import Polygon
from pathlib import Path
from piece_picker.sam_utils.model_utils import get_sam2_predictor

class PiecePicker:
    def __init__(self, step_size = 50, upper_threshold = 0.1, lower_threshold = 0.01, overlap_threshold = 0.2, is_save_inference=False):
        self.step_size = step_size
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.overlap_threshold = overlap_threshold
        self.is_save_inference = is_save_inference
        self.predictor = get_sam2_predictor()

    def predict(self, image):

        self.predictor.set_image(image)
        x_min = y_min = 1
        x_max, y_max = image.shape[1], image.shape[0]
        area = x_max * y_max

        bboxes = []

        xvalues = np.arange(x_min, x_max + self.step_size - 1, self.step_size)
        yvalues = np.arange(y_min, y_max + self.step_size - 1, self.step_size)

        for x, y in it.product(xvalues, yvalues):
            prompt = [x, y]

            input_point = np.array([prompt])
            input_label = np.array([1])

            masks, scores, logits = self.predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=False)

            sorted_indices = np.argsort(scores)[::-1]
            masks = masks[sorted_indices]
            scores = scores[sorted_indices]
            logits = logits[sorted_indices]

            box = self.get_bounding_box(masks)
            box_area = cv2.contourArea(box)

            if self.lower_threshold < (box_area / area) < self.upper_threshold:
                sides_ratio = np.sqrt((box[1][0] - box[0][0]) ** 2 + (box[1][1] - box[0][1]) ** 2) / np.sqrt((box[3][0] - box[0][0]) ** 2 + (box[3][1] - box[0][1]) ** 2)
                if sides_ratio < 1:
                    sides_ratio = 1 / sides_ratio
                bboxes.append({"bbox": box, "score": scores[0], "sides_ratio": sides_ratio, "area": box_area})

        non_overlapping_bboxes = self.get_non_overlapping_bboxes(bboxes)

        if self.is_save_inference:
            self.save_inference(image, non_overlapping_bboxes)

        return non_overlapping_bboxes

    def get_bounding_box(self, masks):
        mask = masks[0]
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        return np.int0(box)
    
    def save_inference(self, image, bboxes):
        for bbox in bboxes:
            cv2.drawContours(image, [bbox["bbox"]], 0, (255, 0, 0), 5)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        Path("inference").mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"inference/inference_{timestamp}.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

    def get_non_overlapping_bboxes(self, bboxes):
        boxes = [bbox["bbox"] for bbox in bboxes]
        scores = [bbox["score"] for bbox in bboxes]

        keep_indices = self.non_max_suppression(boxes, scores)

        return [bboxes[i] for i in keep_indices]
    
    def non_max_suppression(self, boxes, scores):
        indices = np.argsort(scores)[::-1]
        keep_indices = []

        while len(indices) > 0:
            best_index = indices[0]
            keep_indices.append(best_index)

            temp_indices = []
            for index in indices[1:]:
                if self.get_polygon_overlap(boxes[best_index], boxes[index]) < self.overlap_threshold:
                    temp_indices.append(index)
            indices = np.array(temp_indices)

        return keep_indices
    
    def get_polygon_overlap(self, box1, box2):
        box1 = Polygon(box1)
        box2 = Polygon(box2)

        intersection_area = box1.intersection(box2).area
        return max(intersection_area / box1.area, intersection_area / box2.area)