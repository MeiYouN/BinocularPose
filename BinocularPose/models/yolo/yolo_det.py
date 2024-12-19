import cv2
from torchvision.ops import boxes
from ultralytics import YOLO

class Yolo_Det:
    def __init__(self, model='./weights/yolo11n.pt'):
        self.model = YOLO(model)

    def __call__(self, img):
        results = self.model.predict(img,classes=[0])
        boxe = results[0].boxes.xyxy.cpu().numpy()
        if len(boxe) == 0 :
            return None
        else:
            return boxe


def train():
    model = Yolo_Det()

    left_video = "D:\Desktop\EveryThing\WorkProject\ThreeD_demo\data10\pose\\01.mp4"
    capL = cv2.VideoCapture(left_video)
    check, frame = capL.read()
    while check:
        check, frame = capL.read()
        results = model(frame)
        if results:
            print(len(results[0]))
            print(results)



if __name__ == '__main__':
    train()