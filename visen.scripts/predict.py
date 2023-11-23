from ultralytics import YOLO
import os

# model = YOLO('pretrain_models/yolov8x.pt')
# model.predict(source='datasets/VOC2028/JPEGImages', save_txt=True, classes=0, imgsz=640, device=[1,2,3,4,5,6,7])

# Load a model
model = YOLO('pretrain_models/3-v8n-100.pt')  # load an official model

img_path = 'visen.scripts/test_example/'
# model.predict(source=img_path, save=True, imgsz=640, device=[0,1])

img_ls = [img_path + i for i in os.listdir(img_path)]
print(f'img_ls: {img_ls}')
class_dic = {0: 'person', 1: 'head', 2: 'helmet'}

results = model(img_ls, stream=True, device='cpu')
for i, result in zip(range(len(img_ls)), results):
    boxes = result.boxes
    cls = [class_dic[i] for i in boxes.cls.tolist()]
    print(img_ls[i])
    print(cls)
    print(boxes.xywh)

    print('-----' * 20)
