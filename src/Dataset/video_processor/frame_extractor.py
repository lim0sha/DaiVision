import cv2
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import FaceDetector, FaceDetectorOptions, RunningMode
from src.Сonfigs.common_paths import CV2_MODELS_DIR

from src.Dataset.video_processor.sharpness_calculator import get_sharpness_score


def extract_best_face_frame(video_path, step=5):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    model_path = CV2_MODELS_DIR / "blaze_face_short_range.tflite"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Default face detection model not found at {model_path}. "
            "Please ensure 'blaze_face_short_range.tflite' is placed in the 'models' subdirectory."
        )

    options = FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.VIDEO,
        min_detection_confidence=0.5
    )

    best_score = 0
    best_frame = None
    frame_idx = 0

    with FaceDetector.create_from_options(options) as detector:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if frame_idx % step != 0:
                continue

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            )

            timestamp_ms = int(frame_idx / fps * 1000)
            result = detector.detect_for_video(mp_image, timestamp_ms)

            if not result.detections:
                continue

            h, w = frame.shape[:2]
            for detection in result.detections:
                bbox = detection.bounding_box
                face_area = bbox.width * bbox.height
                sharp = get_sharpness_score(frame)
                score = face_area * sharp

                if score > best_score:
                    best_score = score
                    best_frame = frame.copy()

    cap.release()
    return best_frame