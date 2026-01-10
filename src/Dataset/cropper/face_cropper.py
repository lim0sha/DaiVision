"""
Модуль для обрезки изображений до области лица с использованием MediaPipe.

Этот модуль предоставляет функции для обнаружения лиц на изображениях
и последующего вырезания областей с лицами. Используется для подготовки
датасета с изображениями лиц.
"""

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetectorOptions, RunningMode, FaceDetector

from src.Сonfigs.common_paths import CV2_MODELS_DIR


def crop_face_from_image(image_path: str, output_path: str, min_size=100):
    """
    Обрезает изображение до области лица и сохраняет результат.

    Функция использует MediaPipe BlazeFace для обнаружения лица на изображении,
    затем вырезает область лица и сохраняет в указанный файл. Если лицо не найдено
    или размер области меньше минимального, функция возвращает False.

    Args:
        image_path (str): Путь к входному изображению
        output_path (str): Путь для сохранения обрезанного изображения
        min_size (int): Минимальный размер стороны обрезанного изображения (по умолчанию 100)

    Returns:
        bool: True, если лицо успешно обнаружено и сохранено, иначе False

    Raises:
        FileNotFoundError: Если модель BlazeFace не найдена
    """
    # Загружаем изображение
    image = cv2.imread(image_path)
    if image is None:
        return False

    h, w = image.shape[:2]

    # Путь к модели BlazeFace
    model_path = CV2_MODELS_DIR / "blaze_face_short_range.tflite"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    def detect_on_image(img, scale=1.0):
        """
        Обнаруживает лица на изображении с заданным масштабом.

        Args:
            img: Изображение OpenCV
            scale (float): Масштаб изображения для детекции

        Returns:
            list: Список обнаруженных лиц с координатами и оценками
        """
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        result = detector.detect(mp_img)
        detections = []
        for d in result.detections:
            bbox = d.bounding_box
            detections.append({
                'x': int(bbox.origin_x / scale),
                'y': int(bbox.origin_y / scale),
                'width': int(bbox.width / scale),
                'height': int(bbox.height / scale),
                'score': d.categories[0].score
            })
        return detections

    try:
        # Создаем опции для детектора лиц
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            min_detection_confidence=0.5
        )

        # Создаем детектор и выполняем обнаружение
        with FaceDetector.create_from_options(options) as detector:
            # Сначала пробуем обнаружить лица на оригинальном изображении
            detections = detect_on_image(image, scale=1.0)

            # Если лица не найдены, увеличиваем изображение и пробуем снова
            if not detections:
                scaled = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                detections = detect_on_image(scaled, scale=2.0)

            # Если так и не нашли лиц, возвращаем False
            if not detections:
                return False

            # Выбираем лицо с наибольшим показателем уверенности
            best = max(detections, key=lambda d: d['score'])

            # Проверяем минимальный порог уверенности
            if best['score'] < 0.6:  # fine-tune если необходимо и в cropped-датасете много мусора (не лиц)
                return False

            # Вычисляем границы области с лицом
            x_min = max(0, best['x'])
            y_min = max(0, best['y'])
            x_max = min(w, x_min + best['width'])
            y_max = min(h, y_min + best['height'])

            # Проверяем, удовлетворяет ли размер области минимальным требованиям
            if (x_max - x_min) < min_size or (y_max - y_min) < min_size:
                return False

            # Вырезаем область с лицом и сохраняем
            cropped = image[y_min:y_max, x_min:x_max]
            cv2.imwrite(output_path, cropped)
            return True

    except Exception as e:
        print(f"[ERROR] Cropping failed: {e}")
        return False
