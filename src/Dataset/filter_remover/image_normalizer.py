"""
Модуль для адаптивного удаления искусственных фильтров с изображений.

Этот модуль содержит функции для обработки изображений с целью 
удаления чрезмерных световых и цветовых эффектов, применяемых 
в социальных сетях и приложениях знакомств.
"""

import cv2
import numpy as np

from src.Сonfigs.dv_constants import (
    BRIGHTNESS_HISTOGRAM_THRESHOLD,
    BRIGHT_RATIO_THRESHOLD,
    MEAN_BRIGHTNESS_THRESHOLD,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    SATURATION_FACTOR,
    COLOR_CORRECTION_FACTOR,
    GAMMA_CORRECTION,
)


def remove_artificial_filters_adaptive(image: np.ndarray) -> np.ndarray:
    """
    Адаптивно убирает искусственные фильтры только если фото имеет большой засвет.

    Функция анализирует яркость изображения и применяет коррекцию только 
    если содержание ярких участков превышает пороговые значения. 
    В противном случае возвращает копию исходного изображения.

    Args:
        image (np.ndarray): Входное изображение в формате BGR

    Returns:
        np.ndarray: Обработанное изображение (если требовалась обработка) 
                   или копия исходного изображения (если обработка не нужна)

    Raises:
        ValueError: Если входное изображение пустое или недействительно
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or invalid.")
    
    # Преобразуем изображение в цветовое пространство LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Вычисляем гистограмму канала L (яркость)
    hist, _ = np.histogram(l, bins=256, range=(0, 256))
    bright_pixels = np.sum(hist[BRIGHTNESS_HISTOGRAM_THRESHOLD:])
    total_pixels = l.size
    bright_ratio = bright_pixels / total_pixels

    mean_brightness = np.mean(l)
    has_extreme_bright = (
        bright_ratio > BRIGHT_RATIO_THRESHOLD or
        mean_brightness > MEAN_BRIGHTNESS_THRESHOLD
    )

    # Если нет чрезмерной яркости, возвращаем копию исходного изображения
    if not has_extreme_bright:
        return image.copy()
    
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
    l_norm = clahe.apply(l)
    
    saturation_factor = SATURATION_FACTOR
    a = (a.astype(np.float32) - 128) * saturation_factor + 128
    b = (b.astype(np.float32) - 128) * saturation_factor + 128
    a = np.clip(a, 0, 255).astype(np.uint8)
    b = np.clip(b, 0, 255).astype(np.uint8)

    # Собираем каналы обратно и преобразуем в BGR
    lab_norm = cv2.merge([l_norm, a, b])
    result = cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)
    
    # Преобразуем в цветовое пространство YUV для дальнейшей коррекции
    yuv = cv2.cvtColor(result, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv)

    u_centered = (u.astype(np.float32) - 128) * COLOR_CORRECTION_FACTOR + 128
    v_centered = (v.astype(np.float32) - 128) * COLOR_CORRECTION_FACTOR + 128
    u_centered = np.clip(u_centered, 0, 255).astype(np.uint8)
    v_centered = np.clip(v_centered, 0, 255).astype(np.uint8)

    # Собираем YUV каналы обратно и преобразуем в BGR
    yuv_balanced = cv2.merge([y, u_centered, v_centered])
    result = cv2.cvtColor(yuv_balanced, cv2.COLOR_YUV2BGR)
    
    gamma = GAMMA_CORRECTION
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    result = cv2.LUT(result, table)

    return result
