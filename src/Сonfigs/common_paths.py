from pathlib import Path

from src.Dataset.utils.dv_dataset_finder import find_dv_dataset
from src.Dataset.utils.dv_json_finder import find_result_json


# Корень проекта: DaiVision/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Общая папка с датасетами
DATASETS_DIR = PROJECT_ROOT / "datasets"

# Папка с ресурсами
RESOURCES_DIR = PROJECT_ROOT / "resources"

# Папка с моделями cv2
CV2_MODELS_DIR = RESOURCES_DIR / "models"

# Датасет из Дайвинчика
DV_DATASET = find_dv_dataset(DATASETS_DIR)

# results.json из Дайвинчика
DV_RESULTS_JSON_PATH = find_result_json(DATASETS_DIR)

# Внутренняя структура DV-датасета
DV_VIDEO_DIR = DV_DATASET / "video_files"
DV_PHOTOS_EXTRACTED_DIR = DV_DATASET / "photos_extracted"
DV_PHOTOS_UNFILTERED_DIR = DV_DATASET / "photos_unfiltered"
DV_CROPPED_FACES_DIR = DV_DATASET / "photos_cropped"

# Папка для процессинга датасетов
PROCESSED_DIR = PROJECT_ROOT / "files" / "processed"

# CSV-файлы
DV_RAW_CSV = PROCESSED_DIR / "dv_dataset_raw.csv"
DV_FRAMES_CSV = PROCESSED_DIR / "dv_dataset_frames.csv"
DV_FRAMES_UNFILTERED_CSV = PROCESSED_DIR / "dv_dataset_frames_unfiltered.csv"
DV_FRAMES_CROPPED_CSV = PROCESSED_DIR / "dv_dataset_frames_cropped.csv"
