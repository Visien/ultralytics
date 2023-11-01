from ultralytics import YOLO


# Load a model
# model = YOLO('./models/v8/yolov8-p6.yaml')
model = YOLO('yolov8x.yaml')

# Train the model
model.train(data='./datasets/coco-loaf.yaml', epochs=50, batch=12, device='0,1,2,3',workers=12, imgsz=2048, name='yolov8-loaf',
            save=True, save_period=10)