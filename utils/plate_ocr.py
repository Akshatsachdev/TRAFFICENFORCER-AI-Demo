from paddleocr import PaddleOCR
import cv2
import re
import numpy as np
import pandas as pd
from datetime import datetime
import os
from typing import Union

# Initialize PaddleOCR once (global)
ocr = PaddleOCR(use_angle_cls=True, lang='en')
# File paths - constants
CHALLAN_FILE = "data/challan_details.xlsx"
DATA_DIR = "data"


def clean_plate_text(text):
    """
    Cleans OCR output to improve license plate accuracy.
    """
    if not text:
        return ""

    text = str(text).upper()

    # More aggressive common OCR mistakes correction for Indian plates
    replacements = {
        'O': '0', 'Q': '0', 'D': '0',
        'I': '1', 'L': '1',
        'S': '5', 'Z': '2',
        'B': '8', 'G': '6',
        'T': '7', 'Y': '4', 'A': '4'
    }
    for wrong, correct in replacements.items():
        text = text.replace(wrong, correct)

    plate = re.sub(r'[^A-Z0-9]', '', text)

    # Flexible Indian plate format validation
    if re.match(r'^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$', plate):
        return plate

    return plate if len(plate) >= 7 else ""


def extract_plate(image_np):
    """
    Safe version with clipping, dtype enforcement and size limit
    """
    if not isinstance(image_np, np.ndarray):
        raise ValueError("Input must be a numpy array")

    # Channel correction
    if len(image_np.shape) == 2:
        img = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    elif image_np.shape[2] == 4:
        img = cv2.cvtColor(image_np, cv2.COLOR_BGRA2BGR)
    elif image_np.shape[2] == 3:
        img = image_np.copy()
    else:
        raise ValueError(f"Unsupported channels: {image_np.shape}")

    # Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Light sharpening
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    # Upscale (moderate – too high often causes primitive error on CPU)
    scale = 2.5  # ← lowered from 3.0 to prevent huge tensors
    h, w = sharpened.shape
    upscaled = cv2.resize(sharpened, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)

    # Denoise
    denoised = cv2.bilateralFilter(upscaled, d=7, sigmaColor=55, sigmaSpace=55)

    # Very important: clip values and force uint8
    prepared_img = np.clip(denoised, 0, 255).astype(np.uint8)

    # Force 3 channels
    if len(prepared_img.shape) == 2:
        prepared_img = cv2.cvtColor(prepared_img, cv2.COLOR_GRAY2BGR)

    # Critical: limit maximum dimension to prevent memory/primitive errors
    max_side = 960  # ← Paddle CPU works best under 1000px usually
    h, w = prepared_img.shape[:2]
    if max(h, w) > max_side:
        scale_down = max_side / max(h, w)
        prepared_img = cv2.resize(prepared_img, None, fx=scale_down, fy=scale_down,
                                  interpolation=cv2.INTER_AREA)

    # Debug info (remove later if you want)
    print(f"OCR input → shape: {prepared_img.shape}, dtype: {prepared_img.dtype}, "
          f"min/max: {prepared_img.min()}/{prepared_img.max()}")

    # Run OCR
    try:
        result = ocr.ocr(prepared_img, cls=True)
    except Exception as e:
        print(f"OCR failed: {str(e)}")
        return "OCR Error"

    if not result or len(result) == 0 or not result[0]:
        return "Not Detected"

    best_plate = "Not Detected"
    best_conf = 0.75

    for line in result[0]:
        recognition = line[1]

        if isinstance(recognition, (list, tuple)) and len(recognition) == 2:
            text, conf = recognition
        elif isinstance(recognition, (int, float)):
            text = ""
            conf = float(recognition)
        elif isinstance(recognition, str):
            text = recognition
            conf = 0.7
        else:
            continue

        if conf > best_conf:
            plate = clean_plate_text(text)
            if plate:
                best_plate = plate
                best_conf = conf

    return best_plate


def log_challan(plate: str, helmet_status: Union[str, bool]) -> bool:
    """Logs challan to Excel file"""
    if isinstance(helmet_status, bool):
        helmet_display = "With Helmet" if helmet_status else "Without Helmet"
    else:
        helmet_display = str(helmet_status)

    new_entry = {
        "Plate Number": plate,
        "Helmet Status": helmet_display,
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    os.makedirs(DATA_DIR, exist_ok=True)

    try:
        if os.path.exists(CHALLAN_FILE):
            df_existing = pd.read_excel(CHALLAN_FILE, engine='openpyxl')
            df = pd.concat([df_existing, pd.DataFrame(
                [new_entry])], ignore_index=True)
        else:
            df = pd.DataFrame([new_entry])

        df.to_excel(CHALLAN_FILE, index=False, engine='openpyxl')
        return True
    except Exception as e:
        print(f"Challan save failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("Testing log_challan...")
    success = log_challan("TN59AB1234", False)
    print("Success" if success else "Failed")
