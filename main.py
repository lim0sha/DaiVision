from src.Сonfigs.common_paths import DV_RESULTS_JSON_PATH
from src.Dataset.cropper.dv_dataset_cropper import process_dataset_with_face_cropping
from src.Dataset.dataset_builder.dv_dataset_builder import DatasetBuilder
from src.Dataset.filter_remover.dv_dataset_filter_remover import process_dataset_with_filter_removal
from src.Dataset.video_processor.dv_video_rows_processor import process_video_rows

if __name__ == '__main__':
    buildDatasetFromDV = DatasetBuilder(DV_RESULTS_JSON_PATH)
    buildDatasetFromDV.export_to_csv()

    process_video_rows()
    process_dataset_with_filter_removal()
    process_dataset_with_face_cropping()
