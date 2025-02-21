import os
from typing import Dict, List, Union

import cv2
import numpy as np
import supervisely as sly
from ultralytics import YOLO

# we will use `sly.nn.PredictionAlphaMask` class for storing probability maps.
# But you can implement your own class if needed, e.g.:
# class ProbabilityHeatmap(sly.nn.Prediction):
#     def __init__(self, heatmap: np.ndarray):
#         self.heatmap = heatmap


class CustomModel(sly.nn.inference.InstanceSegmentation):
    FRAMEWORK_NAME = "Custom YOLO"  # ⬅︎
    MODELS = "src/demo_data/models_data.json"  # ⬅︎ path to the pretrained models data
    INFERENCE_SETTINGS = "src/custom_settings.yaml"  # Additional inference settings

    def load_model_meta(self):
        """Create a ProjectMeta object that describes the classes and geometry of the model."""
        self.classes = list(self.model.names.values())
        obj_classes = []
        for name in self.classes:
            obj_classes.append(sly.ObjClass(name, sly.Bitmap))  # binary mask
            obj_classes.append(sly.ObjClass(f"{name}_heatmap", sly.AlphaMask))  # probability map
        self._model_meta = sly.ProjectMeta(obj_classes=obj_classes)

    def load_model(self, *arg, **kwargs):
        """Initialize the model and load the weights into memory."""
        checkpoint_path = kwargs.get("model_files", {}).get("checkpoint")
        if checkpoint_path is None:
            checkpoint_path = os.path.join(sly.app.get_data_dir(), "models", "best.pt")
        if not sly.fs.file_exists(checkpoint_path):
            checkpoint_url = "https://github.com/supervisely-ecosystem/tutorial-custom-inference/releases/download/v0.0.1/best.pt"
            sly.fs.download(checkpoint_url, checkpoint_path)

        self.model = YOLO(checkpoint_path)
        self.model.to(kwargs.get("device", "cpu"))
        self.load_model_meta()

    def _create_label(
        self, dto: Union[sly.nn.PredictionAlphaMask, sly.nn.PredictionMask]
    ) -> sly.Label:
        if not dto.mask.any():
            sly.logger.debug(f"Mask of class {name} is empty and will be skipped")
            return None

        name = dto.class_name
        if isinstance(dto, sly.nn.PredictionMask):
            geometry = sly.Bitmap(dto.mask, extra_validation=False)
        elif isinstance(dto, sly.nn.PredictionAlphaMask):
            name = f"{name}_heatmap"
            geometry = sly.AlphaMask(dto.mask, extra_validation=False)
        obj_class = self.model_meta.get_obj_class(name)
        return sly.Label(geometry, obj_class)

    def to_dto(self, predictions: List, settings: Dict) -> List[List[sly.nn.Prediction]]:
        """Convert predictions to PredictionHeatmap objects."""

        # Check if we want to return probability maps
        return_heatmaps = settings.get("return_heatmaps", False)

        results = []
        for prediction in predictions:
            if not prediction.masks:
                continue
            temp_results = []
            for data, mask in zip(prediction.boxes.data, prediction.masks.data):
                mask_class_name = self.classes[int(data[5])]
                mask = mask.cpu().numpy()
                mask = np.where(mask > 0.5, 255, 0).astype(np.uint8)

                dto = sly.nn.PredictionMask(mask_class_name, mask)
                temp_results.append(dto)
                if return_heatmaps:  # If we want to return probability maps
                    mask = cv2.GaussianBlur(mask, (91, 91), 0)  # only for example purposes
                    heatmap_dto = sly.nn.PredictionAlphaMask(mask_class_name, mask)
                    temp_results.append(heatmap_dto)
            results.append(temp_results)
        return results

    def predict_batch(
        self, images_np: List[np.ndarray], settings: Dict
    ) -> List[List[sly.nn.Prediction]]:
        """
        Make predictions on a batch of images.
        For each image, return a list of predictions.
        Each prediction is a list of DTOs (Data Transfer Objects) that represent the detected objects.
        """
        # RGB to BGR
        images_np = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in images_np]
        # Predict
        predictions = self.model(
            source=images_np,
            conf=settings["conf"],
            iou=settings["iou"],
            half=settings["half"],
            device=self.device,
            max_det=settings["max_det"],
            agnostic_nms=settings["agnostic_nms"],
            retina_masks=True,
        )

        # Convert predictions to DTO (Data Transfer Object)
        results = self.to_dto(predictions, settings)
        return results
