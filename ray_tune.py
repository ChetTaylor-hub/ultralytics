from ultralytics import YOLO
from ray import train, tune

# Define a YOLO model
model = YOLO("/data/ct/Code/ultralytics/yolov8n.pt")

# Run Ray Tune on the model
result_grid = model.tune(data="/data/ct/Code/ultralytics/ultralytics/cfg/datasets/my_yaml/Ceramic_Bowl/Ceramic_Bowl.yaml",
                        #  space={"lr0": tune.uniform(1e-5, 1e-1)},
                         iterations=1,
                         device='0',
                         epochs=300,
                         workers=4,
                         batch=32,
                         imgsz=320,
                         save_period=50,
                         cache=False,
                         use_ray=True)