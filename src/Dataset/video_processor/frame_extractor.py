"""
Модуль для извлечения лучшего кадра с лицом из видеофайла.

Этот модуль предоставляет функции для анализа видео и выбора 
кадра с наиболее выраженным лицом, используя MediaPipe для 
детекции лиц и расчет остроты изображения.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions, RunningMode
from src.Сonfigs.common_paths import CV2_MODELS_DIR

from src.Dataset.video_processor.sharpness_calculator import get_sharpness_score


def extract_best_face_frame(video_path, step=5):
    """
    Извлекает лучший кадр с лицом из видеофайла.

    Функция анализирует видео и находит кадр, содержащий наиболее 
    заметное и четкое лицо, используя комбинацию площади лица и 
    остроты изображения как критерий качества.

    Args:
        video_path (str or Path): Путь к видеофайлу
        step (int): Интервал между кадрами для анализа (по умолчанию 5)

    Returns:
        numpy.ndarray or None: Кадр с лучшим лицом или None, если лицо не найдено

    Raises:
        FileNotFoundError: Если модель детекции лиц не найдена
    """
    # Открываем видеофайл
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Проверяем существование модели детекции лиц
    model_path = CV2_MODELS_DIR / "blaze_face_short_range.tflite"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Default face detection model not found at {model_path}. "
            "Please ensure 'blaze_face_short_range.tflite' is placed in the 'models' subdirectory."
        )

    # Создаем опции для детектора лиц
    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        min_detection_confidence=0.5
    )

    # Переменные для отслеживания лучшего кадра
    best_score = 0
    best_frame = None
    frame_idx = 0

    # Создаем детектор лиц и начинаем анализ видео
    with FaceDetector.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            # Пропускаем кадры в соответствии с шагом
            if frame_idx % step != 0:
                continue

            # Подготавливаем изображение для детектора
            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            # Вычисляем временные метки для видео-детекции
            timestamp_ms = int(frame_idx / fps * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            # Пропускаем кадры без обнаруженных лиц
            if not result.detections:
                continue

            h, w = frame.shape[:2]
            # Обрабатываем все обнаруженные лица на кадре
            for detection in result.detections:
                bbox = detection.bounding_box
                # Рассчитываем площадь лица
                face_area = bbox.width * bbox.height
                # Рассчитываем остроту кадра
                sharp = get_sharpness_score(frame)
                # Комбинируем площадь и остроту в один показатель
                score = face_area * sharp

                # Обновляем лучший кадр, если текущий лучше
                if score > best_score:
                    best_score = score
                    best_frame = frame.copy()

    cap.release()
    return best_frame
