"""
Модуль для удаления искусственных фильтров из изображений датасета.

Этот модуль предоставляет функцию для обработки датасета с изображениями,
применяя адаптивное удаление искусственных фильтров к изображениям с чрезмерными эффектами.
Обработанные изображения сохраняются в отдельной директории, а обновленный датасет
записывается в новый CSV-файл.
"""

from pathlib import Path

import cv2
import pandas as pd
import numpy as np

from src.Сonfigs.common_paths import (
    DV_PHOTOS_EXTRACTED_DIR,
    DV_PHOTOS_UNFILTERED_DIR,
    DV_FRAMES_CSV,
    DV_FRAMES_UNFILTERED_CSV
)
from src.Dataset.filter_remover.image_normalizer import remove_artificial_filters_adaptive


def process_dataset_with_filter_removal():
    """
    Обрабатывает датасет из DV_FRAMES_CSV с удалением искусственных фильтров.

    Обрабатывает датасет из файла DV_FRAMES_CSV:
    - сначала ищет изображения в <DV_DATASET>/photos/,
    - если не найдено — пробует в <DV_DATASET>/photos_extracted/,
    - применяет remove_artificial_filters_adaptive ТОЛЬКО если фото имеет "дохера засвета",
    - сохраняет результат в DV_PHOTOS_UNFILTERED_DIR ТОЛЬКО если обработано,
    - записывает новый CSV в DV_FRAMES_UNFILTERED_CSV, меняя путь только у обработанных фото.

    Процесс:
    1. Создает директорию для сохранения обработанных изображений
    2. Читает датасет из CSV-файла
    3. Для каждого изображения:
       - определяет правильный путь к файлу
       - загружает изображение
       - применяет нормализацию фильтров
       - если изображение было изменено, сохраняет его с новым именем
    4. Сохраняет обновленный датасет в новый CSV-файл
    """
    # Создаем директорию для обработанных изображений
    DV_PHOTOS_UNFILTERED_DIR.mkdir(parents=True, exist_ok=True)

    # Читаем исходный датасет
    df = pd.read_csv(DV_FRAMES_CSV)

    # Проверяем наличие необходимой колонки
    if "image_path" not in df.columns:
        raise ValueError("CSV must contain 'image_path' column.")

    new_image_paths = []
    # Определяем директорию с оригинальными фото
    DV_PHOTOS_DIR = DV_PHOTOS_EXTRACTED_DIR.parent / "photos"

    # Обрабатываем каждую строку датасета
    for _, row in df.iterrows():
        rel_path = row["image_path"]
        clean_rel_path = rel_path
        
        # Определяем относительный путь в зависимости от типа изображения
        if rel_path.startswith("photos/"):
            clean_rel_path = rel_path[len("photos/"):]
        elif rel_path.startswith("photos_extracted/"):
            clean_rel_path = rel_path[len("photos_extracted/"):]

        # Пытаемся найти изображение в оригинальной директории
        src_path = DV_PHOTOS_DIR / clean_rel_path

        if not src_path.exists():
            # Если не нашли, пробуем в директории извлеченных фото
            src_path = DV_PHOTOS_EXTRACTED_DIR / clean_rel_path
            if not src_path.exists():
                print(f"[WARNING] Image not found in 'photos' nor 'photos_extracted': {rel_path}")
                new_image_paths.append(rel_path)
                continue

        # Загружаем изображение
        image = cv2.imread(str(src_path))
        if image is None:
            print(f"[WARNING] Failed to read image: {src_path}")
            new_image_paths.append(rel_path)
            continue

        try:
            # Применяем адаптивное удаление фильтров
            normalized_image = remove_artificial_filters_adaptive(image)

        except Exception as e:
            print(f"[ERROR] Failed to process {src_path}: {e}")
            new_image_paths.append(rel_path)
            continue

        # Если изображение не изменилось после обработки, оставляем старый путь
        if np.array_equal(image, normalized_image):
            new_image_paths.append(rel_path)
            continue

        # Генерируем новое имя файла для обработанного изображения
        filename = Path(clean_rel_path).name
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        new_filename = f"{stem}_unfiltered{suffix}"
        dst_path = DV_PHOTOS_UNFILTERED_DIR / new_filename

        # Сохраняем обработанное изображение
        cv2.imwrite(str(dst_path), normalized_image)
        new_rel_path = f"photos_unfiltered/{new_filename}"
        new_image_paths.append(new_rel_path)

    # Обновляем пути к изображениям в датафрейме
    df_out = df.copy()
    df_out["image_path"] = new_image_paths
    df_out.to_csv(DV_FRAMES_UNFILTERED_CSV, index=False)

    print(f"[INFO] Processed dataset saved to: {DV_FRAMES_UNFILTERED_CSV}")
    print(f"[INFO] Processed images saved to: {DV_PHOTOS_UNFILTERED_DIR}")
