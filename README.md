# 😴 Fatigue Detection using YOLO11

A computer vision project for real-time **fatigue detection** using **YOLO11**. The model detects facial features associated with drowsiness by identifying **eye** and **mouth** states, enabling applications such as driver monitoring systems, workplace safety, and operator alertness monitoring.

---
# 📌 Overview

This project develops an object detection model capable of recognizing facial indicators of fatigue using **YOLO11**. Two publicly available Roboflow datasets were merged into a unified dataset with synchronized class labels before training.

The model detects four facial conditions:
- 👁️ Open Eye
- 😴 Closed Eye
- 👄 Open Mouth 
- 🤐 Closed Mouth

These detections can be used as the foundation for higher-level fatigue analysis, where prolonged eye closure or frequent yawning (open mouth) indicates driver drowsiness or reduced alertness. 

---
# 📊 Dataset

This project combines two Roboflow datasets:

### Dataset 1
[Fatigue Detection Dataset](https://universe.roboflow.com/athena-ba1wv/fatigue-detection-loo95/dataset/1)

### Dataset 2
[Tired Detect Dataset](https://universe.roboflow.com/smoke-tcfs5/tired-detect/dataset/3) 

Before training, both datasets were processed to ensure:
- identical class names
- identical class ordering
- compatible YOLO annotations
- merged into a single unified dataset

The final dataset contains four classes:

| Class ID | Class |
|----------|-------|
| 0 | closed_eye |
| 1 | closed_mouth |
| 2 | open_eye |
| 3 | open_mouth |

---

# 🧠 Model

The project uses **YOLO11 Nano (YOLO11n)** from the Ultralytics framework.

**Base Model**
```
yolo11n.pt
```

Training configuration:

| Parameter | Value |
|-----------|-------|
| Epochs | 50 |
| Image Size | 640 |
| Batch Size | 16 |
| Optimizer | Default Ultralytics |
| Early Stopping | Patience = 10 |
| Random Seed | 42 |

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Fatigue-Detection-YOLO11.git
cd Fatigue-Detection-YOLO11
```

Install dependencies

```bash
pip install ultralytics
pip install opencv-python
pip install matplotlib
pip install pandas
pip install pyyaml
```

---

# 🚀 Training

Train the model using:

```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(
    data="final/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10
)
```

---

# 🔍 Inference

```python
from ultralytics import YOLO
model = YOLO("best.pt")
results = model.predict(
    source="image.jpg",
    conf=0.5,
    save=True
)
```

---

# 📈 Training Outputs

YOLO automatically generates:
- training loss curves
- validation loss curves
- precision
- recall
- mAP50
- mAP50-95
- confusion matrix
- prediction samples

---

# 🧪 Classes Detected

| Class | Description |
|--------|-------------|
| Open Eye | Eyes are open |
| Closed Eye | Eyes are closed |
| Open Mouth | Mouth is open (possible yawning) |
| Closed Mouth | Mouth is closed |

---

# 💡 Applications
- Driver Drowsiness Detection
- Smart Vehicle Safety Systems
- Workplace Fatigue Monitoring
- Industrial Operator Monitoring
- Heavy Equipment Safety
- Human Alertness Monitoring
- AI-based Smart Surveillance

---

# 📖 Future Improvements
- Real-time webcam detection
- Eye Aspect Ratio (EAR) integration
- Mouth Aspect Ratio (MAR) integration
- Fatigue score calculation
- Blink frequency analysis
- Yawn duration analysis
- Driver alert system
- Edge deployment on Raspberry Pi or Jetson Nano
