import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceDetectorOptions, RunningMode, FaceDetector

from src.Сonfigs.common_paths import CV2_MODELS_DIR


def crop_face_from_image(image_path: str, output_path: str, min_size=100):
    image = cv2.imread(image_path)
    if image is None:
        return False

    h, w = image.shape[:2]

    model_path = CV2_MODELS_DIR / "blaze_face_short_range.tflite"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    def detect_on_image(img, scale=1.0):
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
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            min_detection_confidence=0.5
        )

        with FaceDetector.create_from_options(options) as detector:
            detections = detect_on_image(image, scale=1.0)
            if not detections:
                scaled = cv2.resize(image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                detections = detect_on_image(scaled, scale=2.0)

            if not detections:
                return False
            best = max(detections, key=lambda d: d['score'])
            if best['score'] < 0.6:  # fine-tune если необходимо и в cropped-датасете много мусора (не лиц)
                return False

            x_min = max(0, best['x'])
            y_min = max(0, best['y'])
            x_max = min(w, x_min + best['width'])
            y_max = min(h, y_min + best['height'])

            if (x_max - x_min) < min_size or (y_max - y_min) < min_size:
                return False

            cropped = image[y_min:y_max, x_min:x_max]
            cv2.imwrite(output_path, cropped)
            return True

    except Exception as e:
        print(f"[ERROR] Cropping failed: {e}")
        return False
