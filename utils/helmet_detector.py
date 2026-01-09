from ultralytics import YOLO
import cv2
import numpy as np
# Load YOLO model once (global)
model = YOLO("models/best.pt")


def detect_helmet(image_np):
    """
    Detects helmet in the given image.
    Returns True if helmet is detected, else False.
    Handles images with 1, 3, or 4 channels.
    """
    # Ensure image has 3 channels (BGR)
    if len(image_np.shape) == 2:
        # Grayscale -> convert to BGR
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.shape[2] == 4:
        # RGBA -> convert to BGR
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)
    elif image_np.shape[2] == 3:
        pass  # already BGR
    else:
        raise ValueError(f"Unexpected number of channels: {image_np.shape[2]}")
    # Run YOLO detection
    results = model(image_np, conf=0.5)
    for r in results:
        if r.boxes is None:
            continue
        for cls in r.boxes.cls:
            if int(cls) == 0:  # helmet class
                return True
    return False
