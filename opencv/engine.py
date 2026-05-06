"""
YOLOv8 inference utilities for the Flask OpenCV project page.
"""

import base64
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

_MODEL = None


def _get_model():
    global _MODEL
    if _MODEL is None:
        # Uses a lightweight model for a responsive dashboard demo.
        _MODEL = YOLO("yolov8n.pt")
    return _MODEL


def detect_objects_from_base64(
    image_base64,
    confidence=0.4,
    max_dim=960,
    person_only=False,
    return_annotated=True,
):
    """Run YOLOv8 detection on a base64 image with dashboard-focused performance options."""
    try:
        if not image_base64:
            return {"error": "No image payload provided."}

        payload = image_base64.split(",", 1)[1] if "," in image_base64 else image_base64
        image_bytes = base64.b64decode(payload)
        pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        h, w = image.shape[:2]
        if max(h, w) > int(max_dim):
            scale = float(max_dim) / float(max(h, w))
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        model = _get_model()
        conf = float(confidence)
        classes = [0] if bool(person_only) else None
        results = model(image, conf=conf, classes=classes, verbose=False)
        result = results[0]

        annotated_image = None
        if return_annotated:
            annotated = result.plot()
            success, encoded = cv2.imencode(".jpg", annotated)
            if not success:
                return {"error": "Failed to encode annotated image."}
            annotated_image = "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("utf-8")

        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                conf_score = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                label = model.names.get(cls_id, str(cls_id)) if isinstance(model.names, dict) else model.names[cls_id]
                detections.append(
                    {
                        "label": label,
                        "confidence": conf_score,
                        "box": [int(x1), int(y1), int(x2), int(y2)],
                    }
                )

        person_count = sum(1 for d in detections if d["label"] == "person")

        return {
            "success": True,
            "annotated_image": annotated_image,
            "detections": detections,
            "object_count": len(detections),
            "person_count": person_count,
            "model": "YOLOv8n",
            "confidence": conf,
            "image_shape": list(image.shape[:2]),
            "person_only": bool(person_only),
        }
    except Exception as exc:
        return {"error": f"Detection failed: {str(exc)}"}
