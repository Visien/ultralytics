from ultralytics import YOLO

def train_v8n():
    model = YOLO('pretrain_yaml/yolov8n.yaml').load('pretrain_models/yolov8n.pt')
    # model = YOLO('pretrain_models/yolov8n.pt')


    # Train the model with 2 GPUs
    results = model.train(data='datasets/Safety_Helmet_Train_dataset/SHT.yaml', 
                        epochs=100, batch=560, workers=40, imgsz=640, 
                        val=False, patience=50,
                        device=[0,1,2,3,4,5,6,7])


def train_v8m():
    model = YOLO('pretrain_yaml/yolov8m.yaml').load('pretrain_models/yolov8m.pt')
    # model = YOLO('pretrain_models/yolov8n.pt')


    # Train the model with 2 GPUs
    results = model.train(data='datasets/Safety_Helmet_Train_dataset/SHT.yaml', 
                        epochs=100, batch=64, workers=16, imgsz=640, 
                        val=False, patience=50,
                        device=[2,3])
    

def train_v8x():
    model = YOLO('pretrain_yaml/yolov8x.yaml').load('pretrain_models/yolov8x.pt')
    # model = YOLO('pretrain_models/yolov8n.pt')


    # Train the model with 2 GPUs
    results = model.train(data='datasets/Safety_Helmet_Train_dataset/SHT.yaml', 
                        epochs=100, batch=80, workers=20, imgsz=640, 
                        val=False, patience=50,
                        device=[4,5,6,7])
    
if __name__ == '__main__':
    train_v8x()
    