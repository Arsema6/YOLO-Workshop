📦 Drug Detection Object Detector
Final Workshop Assignment – Build, Compare, and Deploy Your Own Object Detector
📌 Project Overview

This project focuses on training, evaluating, and comparing three versions of YOLOv8 object detection models on a custom Drug Detection dataset.

The objectives were to:

Train three YOLOv8 models (n, s, m)

Compare performance using evaluation metrics

Analyze trade-offs between speed and accuracy

Deploy the best-performing model for real-time inference

🗂 Dataset

Source: Roboflow

Format: YOLOv8

Classes: 14 drug categories

Validation Images: 180

Task: Object Detection (Bounding Boxes)

🧠 Models Trained

YOLOv8n (Nano)

YOLOv8s (Small)

YOLOv8m (Medium)

All models were fine-tuned from pretrained weights using the same dataset and training configuration.

⚙️ Training Configuration

Framework: Ultralytics YOLOv8

Python: 3.12

GPU: Tesla T4

Image Size: 640

Epochs: 20

Batch Size: Auto

📊 Model Performance Comparison
Model	Parameters	mAP50	mAP50–95	Precision	Recall	Inference Speed (ms)	Model Size (MB)
YOLOv8n	3.0M	0.961	0.922	0.934	0.931	4.4 ms	~6 MB
YOLOv8s	11.1M	0.966	0.925	0.956	0.946	9.2 ms	~22 MB
YOLOv8m	25.8M	0.956	0.917	0.946	0.927	21.6 ms	~50 MB
🔎 Key Observations

Highest Accuracy: YOLOv8s achieved the highest mAP50 (0.966) and mAP50–95 (0.925).

Fastest Model: YOLOv8n achieved the fastest inference time (4.4 ms per image).

Best Overall Trade-Off: YOLOv8s provides the best balance between speed and detection accuracy.

Larger Model ≠ Better Performance: Despite having more parameters, YOLOv8m did not outperform YOLOv8s.

📈 Metrics Analysis
🔹 Precision vs Recall

Precision measures how many predicted detections are correct.

Recall measures how many actual objects were successfully detected.

High precision reduces false positives.
High recall reduces missed detections.

YOLOv8s achieved the best balance between precision and recall, making it the most reliable detector for this dataset.

⚖️ Speed vs Accuracy Trade-Off
Model	Strength	Best Use Case
YOLOv8n	Extremely fast	Edge devices / strict real-time
YOLOv8s	Best accuracy-speed balance	Production systems
YOLOv8m	Larger capacity	High-compute environments

For real-time deployment with high reliability, YOLOv8s is recommended.

🎥 Real-Time Deployment

The best-performing model (YOLOv8s) was deployed locally using webcam inference.

Example:

from ultralytics import YOLO

model = YOLO("yolov8s_best.pt")
model.predict(source=0, show=True)
Deployment Results

Real-time detection achieved

Stable bounding box predictions

Low latency on GPU hardware

📸 Inference Results

Sample detection outputs are available in:

results/detection_outputs/

These images demonstrate accurate classification and localization across multiple drug classes.

🚀 How to Reproduce Results

Install dependencies:

pip install ultralytics roboflow

Download dataset from Roboflow (YOLOv8 format)

Train a model:

from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="data.yaml", epochs=20)

Validate:

model.val()

Run inference:

model.predict(source="image.jpg", show=True)
🎯 Conclusion

This project demonstrates that:

Model size does not always guarantee better accuracy.

YOLOv8s achieved the best overall performance.

YOLOv8n is ideal for low-latency systems.

Proper evaluation metrics are critical when selecting a deployment model.
