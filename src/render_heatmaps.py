import cv2
import numpy as np
import supervisely as sly


def color_map(img_size, data: np.ndarray, origin: sly.PointLocation) -> np.ndarray:
    mask = np.zeros(img_size, dtype=np.uint8)
    x, y = origin.col, origin.row
    h, w = data.shape[:2]
    mask[y : y + h, x : x + w] = data
    cv2.normalize(mask, mask, 0, 255, cv2.NORM_MINMAX)
    mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    BG_COLOR = np.array([128, 0, 0], dtype=np.uint8)
    mask = np.where(mask == BG_COLOR, 0, mask)
    return mask


def render_heatmaps_on_image(img_path: str, ann: sly.Annotation) -> np.ndarray:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    temp = img.copy()
    for label in ann.labels[::-1]:
        if isinstance(label.geometry, sly.AlphaMask):
            mask = color_map(ann.img_size, label.geometry.data, label.geometry.origin)
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
            temp = np.where(np.any(mask > 0, axis=-1, keepdims=True), mask, temp)
    result = cv2.addWeighted(img, 0.5, temp, 0.5, 0).astype(np.uint8)
    return result
