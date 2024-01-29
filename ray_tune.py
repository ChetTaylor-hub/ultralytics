from ultralytics import YOLO
from ray import train, tune

# Define a YOLO model
model = YOLO("yolov8n.pt")

# Run Ray Tune on the model
result_grid = model.tune(data="coco128.yaml",
                         space={"lr0": tune.uniform(1e-5, 1e-1)},
                         epochs=50,
                         use_ray=True)