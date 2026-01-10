"""
Модуль для обработки датасета с обрезкой изображений до лиц.

Этот модуль предоставляет функции для обработки датасета, 
содержащего пути к изображениям, и обрезки этих изображений 
до областей с лицами с помощью детектора лиц.
"""

import pandas as pd
from pathlib import Path

from src.Сonfigs.common_paths import (
    DV_FRAMES_UNFILTERED_CSV,
    DV_CROPPED_FACES_DIR,
    DV_DATASET
)
from src.Dataset.cropper.face_cropper import crop_face_from_image


def process_dataset_with_face_cropping():
    """
    Обрабатывает датасет: обрезает фото до лиц.

    Обрабатывает датасет: обрезает фото до лиц.
    - читает CSV с unfiltered фото,
    - для каждого фото строит правильный путь (учитывая, что image_path может быть:
        - photos/...
        - photos_extracted/...
        - photos_unfiltered/...
    ),
    - если найдено лицо — сохраняет обрезанное фото в DV_CROPPED_FACES_DIR,
    - если лицо не найдено, то запись не добавляется в итоговый датасет
    - записывает новый CSV с путями к обрезанным фото.

    Процесс:
    1. Создает директорию для обрезанных изображений
    2. Читает CSV-файл с необработанными изображениями
    3. Для каждой строки датасета:
       - определяет правильный путь к исходному изображению
       - обрезает изображение до области с лицом
       - сохраняет обрезанное изображение
       - добавляет строку в выходной датасет (если лицо найдено)
    4. Сохраняет обновленный датасет в CSV-файл
    """
    # Создаем директорию для обрезанных лиц
    DV_CROPPED_FACES_DIR.mkdir(parents=True, exist_ok=True)

    # Читаем исходный датасет
    df = pd.read_csv(DV_FRAMES_UNFILTERED_CSV)

    # Проверяем наличие обязательной колонки
    if "image_path" not in df.columns:
        raise ValueError("CSV must contain 'image_path' column.")

    kept_rows = []

    # Обрабатываем каждую строку датасета
    for _, row in df.iterrows():
        rel_path = row["image_path"]
        clean_rel_path = rel_path
        base_dir = None

        # Определяем базовую директорию в зависимости от префикса пути
        if rel_path.startswith("photos_unfiltered/"):
            clean_rel_path = rel_path[len("photos_unfiltered/"):]
            base_dir = DV_DATASET / "photos_unfiltered"
        elif rel_path.startswith("photos_extracted/"):
            clean_rel_path = rel_path[len("photos_extracted/"):]
            base_dir = DV_DATASET / "photos_extracted"
        elif rel_path.startswith("photos/"):
            clean_rel_path = rel_path[len("photos/"):]
            base_dir = DV_DATASET / "photos"
        else:
            # Если префикс не определен, пробуем все возможные директории
            for candidate_name in ["photos", "photos_extracted", "photos_unfiltered"]:
                candidate_path = DV_DATASET / candidate_name / rel_path
                if candidate_path.exists():
                    base_dir = DV_DATASET / candidate_name
                    clean_rel_path = rel_path
                    break
            if base_dir is None:
                continue

        # Формируем полный путь к исходному изображению
        src_path = base_dir / clean_rel_path
        if not src_path.exists():
            continue

        # Генерируем новое имя файла для обрезанного изображения
        filename = Path(clean_rel_path).name
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        new_filename = f"{stem}_cropped{suffix}"
        dst_path = DV_CROPPED_FACES_DIR / new_filename
        
        # Обрезаем изображение до области с лицом
        success = crop_face_from_image(str(src_path), str(dst_path), min_size=80)

        # Если лицо успешно найдено и обрезано, добавляем строку в выходной датасет
        if success:
            new_row = row.copy()
            new_row["image_path"] = f"photos_cropped/{new_filename}"
            kept_rows.append(new_row)
    
    # Создаем выходной датафрейм и сохраняем в CSV
    df_out = pd.DataFrame(kept_rows)
    output_csv = DV_FRAMES_UNFILTERED_CSV.parent / "dv_dataset_frames_cropped_filtered.csv"
    df_out.to_csv(output_csv, index=False)

    print(f"[INFO] Filtered dataset saved to: {output_csv}")
    print(f"[INFO] Kept {len(df_out)} rows out of {len(df)}")
    print(f"[INFO] Cropped faces saved to: {DV_CROPPED_FACES_DIR}")
