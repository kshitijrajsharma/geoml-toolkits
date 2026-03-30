import numpy as np
import rasterio
from scipy import ndimage

from .._logging import get_logger

log = get_logger(__name__)


def morphological_opening(mask: np.ndarray, kernel_size: int = 3, iterations: int = 2) -> np.ndarray:
    """Apply morphological opening (erosion then dilation) to remove small objects."""
    structure = np.ones((kernel_size, kernel_size), dtype=bool)
    result = mask.copy()
    for _ in range(iterations):
        result = ndimage.binary_opening(result, structure=structure)
    return result.astype(mask.dtype) * np.iinfo(mask.dtype).max if np.issubdtype(mask.dtype, np.integer) else result


def clear_border(mask: np.ndarray) -> np.ndarray:
    """Remove connected components that touch any border of the image."""
    labeled, num_features = ndimage.label(mask > 0)
    if num_features == 0:
        return mask

    border_labels = set()
    border_labels.update(labeled[0, :].ravel())
    border_labels.update(labeled[-1, :].ravel())
    border_labels.update(labeled[:, 0].ravel())
    border_labels.update(labeled[:, -1].ravel())
    border_labels.discard(0)

    cleaned = mask.copy()
    for label_id in border_labels:
        cleaned[labeled == label_id] = 0

    return cleaned


def morphological_cleaning(prediction_merged_mask_path: str) -> str:
    """Apply morphological opening and border clearing to a prediction mask raster."""
    with rasterio.open(prediction_merged_mask_path) as src:
        img = src.read(1)
        profile = src.profile.copy()

    opened = morphological_opening(img, kernel_size=3, iterations=2)
    cleaned = clear_border(opened)

    with rasterio.open(prediction_merged_mask_path, "w", **profile) as dst:
        dst.write(cleaned, 1)

    log.info(f"Cleaned mask: {prediction_merged_mask_path}")
    return prediction_merged_mask_path


def extract_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Extract contour boundaries from a binary mask using scipy label + boundary detection."""
    labeled, num_features = ndimage.label(mask > 0)
    contours = []
    for i in range(1, num_features + 1):
        component = (labeled == i).astype(np.uint8)
        eroded = ndimage.binary_erosion(component)
        boundary = component - eroded.astype(np.uint8)
        coords = np.argwhere(boundary > 0)
        if len(coords) > 2:
            contours.append(coords)
    return contours
