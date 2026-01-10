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
    DV_PHOTOS_EXTRACTED_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DV_RAW_CSV)
    new_rows = []

    for _, row in df.iterrows():
        image_path = row["image_path"]

        if not isinstance(image_path, str) or not image_path.lower().endswith(".mp4"):
            new_rows.append(row)
            continue

        video_path = DV_VIDEO_DIR / Path(image_path).name

        if not video_path.exists():
            print(f"[WARN] Видео не найдено: {video_path}")
            continue

        best_frame = extract_best_face_frame(video_path)

        if best_frame is None:
            print(f"[WARN] Лицо не найдено: {video_path}")
            continue

        photo_name = video_path.stem + ".jpg"
        photo_path = DV_PHOTOS_EXTRACTED_DIR / photo_name

        cv2.imwrite(str(photo_path), best_frame)

        new_row = row.copy()
        new_row["image_path"] = f"photos_extracted/{photo_name}"
        new_row["image_index"] = 0

        new_rows.append(new_row)

    pd.DataFrame(new_rows, columns=df.columns).to_csv(
        DV_FRAMES_CSV,
        index=False,
        encoding="utf-8"
    )
