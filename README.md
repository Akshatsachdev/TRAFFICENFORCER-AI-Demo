# 🚦 TrafficEnforcer-AI

**TrafficEnforcer-AI** is an AI-powered vehicle monitoring and challan generation system.  
It automatically detects **helmet violations** and **reads vehicle license plates** from images or video frames using **YOLOv8** and **PaddleOCR**.

---

## **Features**

- **Helmet Detection:** Detects riders wearing helmets using YOLOv8.
- **License Plate Recognition:** Extracts license plate numbers using PaddleOCR.
- **Real-time Processing:** Works on images and video streams.
- **Confidence Thresholds:** Only detects high-confidence results for accuracy.
- **Clean Output:** Automatically formats and cleans license plate numbers.

---

## **Screenshots**

![Helmet Detection](assets/helmet_demo.png)  
*Helmet detected on a rider.*

![License Plate OCR](assets/plate_demo.png)  
*License plate successfully recognized.*

---

## **Installation**

Clone the repository:

```bash
git clone https://github.com/Akshatsachdev/TRAFFICENFORCER-AI-Demo-.git
cd TRAFFICENFORCER-AI-Demo-
````

Create a virtual environment (optional but recommended):

```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux
```

Install dependencies:

```bash
pip install -r requirements.txt
```

> Make sure your `models/best.pt` YOLOv8 model is in place.

---

## **Usage**

Run the Streamlit app:

```bash
streamlit run app.py
```

* Upload an image or video frame.
* The app will detect helmets and extract license plate numbers.
* Results will be displayed along with processed images.

---

## **Requirements**

* Python 3.8+
* Streamlit
* OpenCV
* PaddleOCR
* Ultralytics YOLOv8
* Numpy
* Pillow
* Torch

Example `requirements.txt`:

```
streamlit
numpy
pillow
opencv-python
opencv-python-headless
torch
ultralytics
paddleocr
```

---

## **Project Structure**

```
TrafficEnforcer-AI/
│
├─ app.py                  # Streamlit main app
├─ requirements.txt        # Project dependencies
├─ models/
│   └─ best.pt             # YOLOv8 helmet detection model
├─ utils/
│   ├─ helmet_detector.py  # YOLO detection code
│   └─ plate_ocr.py        # PaddleOCR license plate extraction
├─ assets/                 # Demo images/screenshots
└─ README.md
```

---

## **Notes**

* PaddleOCR automatically downloads necessary model weights to `~/.paddleocr`.
* YOLOv8 model must be present in `models/best.pt`.
* Handles RGBA, RGB, and grayscale images automatically.

---

## **Future Enhancements**

* Live video stream support for traffic cameras.
* Automatic challan PDF generation.
* GPS tagging for violations.
* Integration with government traffic systems.

---

## **License**

This project is open-source and available under the MIT License.

---

## **Author**

**Akshat Sachdeva**

* GitHub: [Akshatsachdev](https://github.com/Akshatsachdev)
* LinkedIn: [Ak0011](https://www.linkedin.com/in/ak0011)

---

