import cv2
import numpy as np


def remove_artificial_filters_adaptive(image: np.ndarray) -> np.ndarray:
    """
    Убирает искусственные фильтры только если фото имеет большой засвет.
    В остальных случаях — оставляет без изменений.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or invalid.")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    hist, _ = np.histogram(l, bins=256, range=(0, 256))
    bright_pixels = np.sum(hist[230:])
    total_pixels = l.size
    bright_ratio = bright_pixels / total_pixels

    mean_brightness = np.mean(l)
    has_extreme_bright = (
        bright_ratio > 0.150 or
        mean_brightness > 150
    )

    if not has_extreme_bright:
        return image.copy()
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_norm = clahe.apply(l)
    saturation_factor = 0.5
    a = (a.astype(np.float32) - 128) * saturation_factor + 128
    b = (b.astype(np.float32) - 128) * saturation_factor + 128
    a = np.clip(a, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)

    lab_norm = cv2.merge([l_norm, a, b])
    result = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)
    yuv = cv2.cvtColor(result, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)

    u_centered = (u.astype(np.float32) - 128) * 0.7 + 128
    v_centered = (v.astype(np.float32) - 128) * 0.7 + 128
    u_centered = np.clip(u_centered, 0, 255).astype(np.uint8)
    v_centered = np.clip(v_centered, 0, 255).astype(np.uint8)

    yuv_balanced = cv2.merge([y, u_centered, v_centered])
    result = cv2.cvtColor(yuv_balanced, cv2.COLOR_YUV2BGR)
    gamma = 1.4
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    result = cv2.LUT(result, table)

    return result