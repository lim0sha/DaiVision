"""
Модуль для обработки строк датасета, содержащих видеофайлы.

Этот модуль предоставляет функции для обработки датасета,
в котором некоторые записи могут содержать пути к видеофайлам.
Видео обрабатываются для извлечения кадра с лучшим лицом,
который затем заменяет видео в датасете.
"""

from pathlib import Path

import cv2
import pandas as pd

from src.Сonfigs.common_paths import (
    DV_VIDEO_DIR,
    DV_PHOTOS_EXTRACTED_DIR,
    DV_RAW_CSV,
    DV_FRAMES_CSV,
)

from src.Dataset.video_processor.frame_extractor import extract_best_face_frame


def process_video_rows():
    """
    Обрабатывает строки датасета, содержащие видеофайлы.

    Процесс:
    1. Создает директорию для извлеченных фото
    2. Читает исходный датасет из CSV-файла
    3. Для каждой строки:
       - Проверяет, является ли путь к изображению видеофайлом (.mp4)
       - Если да, извлекает лучший кадр с лицом из видео
       - Сохраняет кадр как изображение
       - Обновляет путь в датасете на путь к изображению
       - Если это не видео или лицо не найдено, оставляет строку без изменений
    4. Сохраняет обновленный датасет в CSV-файл

    Заменяет видеофайлы на изображения с лучшим кадром, содержащим лицо.
    """
    # Создаем директорию для извлеченных фото
    DV_PHOTOS_EXTRACTED_DIR.mkdir(exist_ok=True)

    # Читаем исходный датасет
    df = pd.read_csv(DV_RAW_CSV)
    new_rows = []

    # Обрабатываем каждую строку датасета
    for _, row in df.iterrows():
        image_path = row["image_path"]

        # Если путь не является видеофайлом, просто добавляем строку в результат
        if not isinstance(image_path, str) or not image_path.lower().endswith(".mp4"):
            new_rows.append(row)
            continue

        # Формируем путь к видеофайлу
        video_path = DV_VIDEO_DIR / Path(image_path).name

        # Проверяем существование видеофайла
        if not video_path.exists():
            print(f"[WARN] Видео не найдено: {video_path}")
            continue

        # Извлекаем лучший кадр с лицом из видео
        best_frame = extract_best_face_frame(video_path)

        # Если лицо не найдено, пропускаем
        if best_frame is None:
            print(f"[WARN] Лицо не найдено: {video_path}")
            continue

        # Генерируем имя и путь для сохранения извлеченного кадра
        photo_name = video_path.stem + ".jpg"
        photo_path = DV_PHOTOS_EXTRACTED_DIR / photo_name

        # Сохраняем кадр как изображение
        cv2.imwrite(str(photo_path), best_frame)

        # Создаем новую строку с обновленным путем к изображению
        new_row = row.copy()
        new_row["image_path"] = f"photos_extracted/{photo_name}"
        new_row["image_index"] = 0

        new_rows.append(new_row)

    # Сохраняем обновленный датасет в CSV-файл
    pd.DataFrame(new_rows, columns=df.columns).to_csv(
        DV_FRAMES_CSV,
        index=False,
        encoding="utf-8"
    )
