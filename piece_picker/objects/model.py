import itertools as it
import numpy as np
import cv2
import datetime
from shapely import Polygon
from pathlib import Path
from piece_picker.sam_utils.model_utils import get_sam2_predictor


class PiecePicker:
    """
    This class is responsible for using a segmentation predictor to detect and localize pieces
    within an image. It performs a grid search over the image, predicts segmentation masks,
    extracts bounding boxes, filters them based on area and aspect ratio thresholds, and finally
    applies non-maximum suppression to remove overlapping detections. Optionally, it can also save
    the inference result with the detected bounding boxes drawn on the input image.

    Attributes:
        step_size (int): The number of pixels to move between successive grid points while scanning the image.
        upper_threshold (float): The maximum allowed ratio (bounding box area / image area) for a valid detection.
        lower_threshold (float): The minimum allowed ratio (bounding box area / image area) for a valid detection.
        overlap_threshold (float): The maximum acceptable overlap ratio between detected bounding boxes.
        is_save_inference (bool): Flag indicating whether to save an image with drawn bounding boxes.
        predictor (SAM2ImagePredictor): The segmentation predictor instance used to generate masks and scores.
    """

    def __init__(
        self,
        step_size=50,
        upper_threshold=0.1,
        lower_threshold=0.01,
        overlap_threshold=0.2,
        is_save_inference=False,
    ):
        """
        Initialize the model object with the given thresholds and configuration options.
        Parameters:
            step_size (int, optional): The step size used for processing. Default is 50.
            upper_threshold (float, optional): The threshold representing the upper confidence limit. Default is 0.1.
            lower_threshold (float, optional): The threshold representing the lower confidence limit. Default is 0.01.
            overlap_threshold (float, optional): The threshold to determine acceptable prediction overlap. Default is 0.2.
            is_save_inference (bool, optional): A flag to indicate whether inference results should be saved. Default is False.
        Side Effects:
            Initializes the predictor with a SAM2 predictor instance via the get_sam2_predictor function.
        """
        self.step_size = step_size
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        self.overlap_threshold = overlap_threshold
        self.is_save_inference = is_save_inference
        self.predictor = get_sam2_predictor()

    def predict(self, image):
        """
        Predict objects in the provided image using a grid approach.
        This method iterates over the image using a grid defined by self.step_size and performs
        the following steps for each grid coordinate:
            - Sets the current image in the predictor.
            - Constructs a point prompt based on the current (x, y) coordinate.
            - Uses the predictor to generate masks, scores, and logits.
            - Sorts the outputs by score and extracts the corresponding bounding box using the
              get_bounding_box method.
            - Computes the area of the bounding box and normalizes it with respect to the full image area.
            - Filters the bounding box based on preset lower and upper area thresholds.
            - Calculates a sides ratio to measure dimensional proportions of the bounding box.
            - If the bounding box meets the criteria, it is added to the list of candidates.
        After processing all grid coordinates, the method merges overlapping boxes using get_non_overlapping_bboxes.
        If self.is_save_inference is enabled, the inference results are saved with the input image.
        Finally, it returns a list of dictionaries, each representing a valid bounding box.
        Parameters:
            image (ndarray): The input image as a NumPy array. The image dimensions are used to
                             determine the area and limits for generating prompt points.
        Returns:
            List[dict]: A list of dictionaries, where each dictionary contains:
                        - "bbox": The coordinates of the predicted bounding box.
                        - "score": The prediction confidence score.
                        - "sides_ratio": The ratio describing the relative side lengths of the bounding box.
                        - "area": The area of the bounding box.
        """
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

            masks, scores, logits = self.predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=False,
            )

            sorted_indices = np.argsort(scores)[::-1]
            masks = masks[sorted_indices]
            scores = scores[sorted_indices]
            logits = logits[sorted_indices]

            box = self.get_bounding_box(masks)
            box_area = cv2.contourArea(box)

            if self.lower_threshold < (box_area / area) < self.upper_threshold:
                sides_ratio = np.sqrt(
                    (box[1][0] - box[0][0]) ** 2 + (box[1][1] - box[0][1]) ** 2
                ) / np.sqrt((box[3][0] - box[0][0]) ** 2 + (box[3][1] - box[0][1]) ** 2)
                if sides_ratio < 1:
                    sides_ratio = 1 / sides_ratio
                bboxes.append(
                    {
                        "bbox": box,
                        "score": scores[0],
                        "sides_ratio": sides_ratio,
                        "area": box_area,
                    }
                )

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
        cv2.imwrite(
            f"inference/inference_{timestamp}.jpg",
            cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        )

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
                if (
                    self.get_polygon_overlap(boxes[best_index], boxes[index])
                    < self.overlap_threshold
                ):
                    temp_indices.append(index)
            indices = np.array(temp_indices)

        return keep_indices

    def get_polygon_overlap(self, box1, box2):
        box1 = Polygon(box1)
        box2 = Polygon(box2)

        intersection_area = box1.intersection(box2).area
        return max(intersection_area / box1.area, intersection_area / box2.area)
