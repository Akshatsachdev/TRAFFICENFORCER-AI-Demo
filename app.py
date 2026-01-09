import streamlit as st
import numpy as np
import cv2
from PIL import Image
from utils.helmet_detector import detect_helmet
# Ensure log_challan is imported
from utils.plate_ocr import extract_plate, log_challan

# IMPORTANT: Disable MKL-DNN to avoid "could not execute a primitive" on CPU
# Add this before any Paddle import / OCR call (best at top of file)
import os
os.environ["OMP_NUM_THREADS"] = "1"          # Limit threads
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_DYNAMIC"] = "false"
# If still issues, also try in plate_ocr.py: ocr = PaddleOCR(..., enable_mkldnn=False)

st.set_page_config(page_title="TrafficEnforcer-AI Demo", layout="wide")

st.title("🚦 TRAFFICENFORCER-AI")
st.caption(
    "AI-Based Helmet & Number Plate Violation Detection Demo (Indian Vehicles)")
st.markdown("---")

# ── Sidebar for settings ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    upscale_factor = st.slider("OCR Upscale Factor", 1.5, 3.0, 2.0, step=0.5,
                               help="Start low (2.0) to avoid CPU errors")
    padding = st.slider("Plate Crop Padding (px)", 10, 60, 20)
    min_crop_height = st.slider("Min Plate Height (px)", 80, 200, 100)

# ── Main content ─────────────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload a vehicle image (jpg, jpeg, png)",
    type=["jpg", "jpeg", "png"],
    help="Best results with clear daylight images showing helmet & plate clearly."
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)  # RGB

    col1, col2 = st.columns([3, 1])
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("🚀 Run AI Detection", type="primary"):
        with st.spinner("Analyzing image..."):
            # 1. Helmet on full image
            helmet_detected = detect_helmet(image_np)

            # 2. Plate crop – REPLACE THIS WITH REAL YOLO PLATE BOX WHEN READY!
            # For now: use full image as fallback (will be poor, but prevents crash)
            h, w, _ = image_np.shape
            padding_px = padding

            # Safe padding
            y1, y2 = max(0, 0 - padding_px), min(h, h + padding_px)
            x1, x2 = max(0, 0 - padding_px), min(w, w + padding_px)
            plate_crop_rgb = image_np[y1:y2, x1:x2]

            # Resize if too small
            if plate_crop_rgb.shape[0] < min_crop_height:
                scale = min_crop_height / plate_crop_rgb.shape[0]
                plate_crop_rgb = cv2.resize(plate_crop_rgb, None,
                                            fx=scale, fy=scale,
                                            interpolation=cv2.INTER_CUBIC)

            # Convert to BGR for OpenCV processing
            plate_crop_bgr = cv2.cvtColor(plate_crop_rgb, cv2.COLOR_RGB2BGR)

            # Preprocessing
            gray = cv2.cvtColor(plate_crop_bgr, cv2.COLOR_BGR2GRAY)
            # Lower clip to be safer
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)

            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(enhanced, -1, kernel)

            # Upscale – capped to avoid huge tensors
            max_side = 960  # Safe limit for CPU
            scale_factor = upscale_factor
            up_h, up_w = int(
                sharpened.shape[0] * scale_factor), int(sharpened.shape[1] * scale_factor)
            if max(up_h, up_w) > max_side:
                scale_factor = max_side / max(sharpened.shape)
                up_h, up_w = int(
                    sharpened.shape[0] * scale_factor), int(sharpened.shape[1] * scale_factor)

            upscaled = cv2.resize(sharpened, (up_w, up_h),
                                  interpolation=cv2.INTER_CUBIC)

            denoised = cv2.bilateralFilter(
                upscaled, d=5, sigmaColor=50, sigmaSpace=50)

            prepared_plate = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

            # Safety: clip & uint8
            prepared_plate = np.clip(prepared_plate, 0, 255).astype(np.uint8)

            # Debug info
            st.sidebar.write(
                f"OCR input shape: {prepared_plate.shape}, dtype: {prepared_plate.dtype}")

            # Show preprocessed
            with col2:
                st.image(prepared_plate, caption="Preprocessed Plate (OCR Input)",
                         use_container_width=True, channels="BGR")

            # OCR – with fallback
            try:
                plate_number = extract_plate(prepared_plate)
            except Exception as e:
                st.error(f"OCR failed: {str(e)}")
                plate_number = "OCR Error – Try smaller upscale or different image"

        # ── Results ──────────────────────────────────────────────────────────────
        st.subheader("🔍 Detection Results")
        col_res1, col_res2 = st.columns(2)

        with col_res1:
            st.metric("🪖 Helmet Status",
                      "✅ With Helmet" if helmet_detected else "❌ Without Helmet")

        with col_res2:
            st.metric("🚗 Number Plate",
                      plate_number if plate_number != "Not Detected" else "❓ Not Detected")

        if not helmet_detected and plate_number not in ["Not Detected", "OCR Error – Try smaller upscale or different image"]:
            st.error("⚠️ **HELMET VIOLATION DETECTED** — Challan Applicable!")
            st.info(f"Plate: **{plate_number}** | Action: Issue challan")

            # Log challan
            try:
                success = log_challan(plate_number, helmet_detected)
                if success:
                    st.success("Challan logged!")
                else:
                    st.warning("Failed to log challan")
            except Exception as e:
                st.info(f"Logging skipped: {e}")
        elif plate_number in ["Not Detected", "OCR Error – Try smaller upscale or different image"]:
            st.warning(
                "No clear plate detected or OCR issue. Try image with larger/ clearer plate.")
        else:
            st.success("✅ No Violation – Safe Ride!")
