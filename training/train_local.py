"""
train_local.py - Train the campus YOLO model on THIS laptop (CPU). Slower than Colab
(~1.5-2 h) but needs nothing else. Run from the VisionSecurityRobot folder:

    python training/train_local.py

When it finishes, the model is copied to models/campus_yolo.pt automatically.
"""
import os
import shutil

from ultralytics import YOLO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data = os.path.join(ROOT, "dataset", "data.yaml")
model = YOLO(os.path.join(ROOT, "models", "yolov8n.pt"))
model.train(data=data, epochs=40, imgsz=416, batch=16, patience=15, device="cpu", workers=2,
            hsv_v=0.6, hsv_s=0.6, fliplr=0.5, mosaic=1.0, close_mosaic=8,
            project=os.path.join(ROOT, "training", "results"), name="campus", exist_ok=True, seed=7)
best = os.path.join(ROOT, "training", "results", "campus", "weights", "best.pt")
shutil.copy(best, os.path.join(ROOT, "models", "campus_yolo.pt"))
print("\nSaved trained model to models/campus_yolo.pt")
m = YOLO(best).val(data=data, split="test", imgsz=416)
print("night test set: mAP50 %.3f  mAP50-95 %.3f" % (m.box.map50, m.box.map))
