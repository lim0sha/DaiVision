"""
Константы для проекта DaiVision.

Содержит все магические числа и конфигурационные параметры,
используемые в различных модулях проекта.
"""

# Face Detection (MediaPipe)
MIN_DETECTION_CONFIDENCE = 0.5
MIN_CROP_CONFIDENCE = 0.6
MIN_CROP_SIZE = 100
MIN_CROP_SIZE_FALLBACK = 80
FACE_SCALE_RETRY = 2.0

# Video Processing
VIDEO_FRAME_STEP = 5

# Image Normalization (Filter Removal)
BRIGHTNESS_HISTOGRAM_THRESHOLD = 230
BRIGHT_RATIO_THRESHOLD = 0.150
MEAN_BRIGHTNESS_THRESHOLD = 150
CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID_SIZE = (8, 8)
SATURATION_FACTOR = 0.5
COLOR_CORRECTION_FACTOR = 0.7
GAMMA_CORRECTION = 1.4