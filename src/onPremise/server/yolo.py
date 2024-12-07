from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Perform object detection on an image
results = model("ps/project/TheObservatory/src/onPremise/server/static/frames/9abab5e2-afe5-4991-8470-7ad5fd6a316f.jpg")
results[0].show()
