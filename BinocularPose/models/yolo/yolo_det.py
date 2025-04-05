import cv2
from ultralytics import YOLO

class Yolo_Det:
    def __init__(self, model='./weights/yolo11n.pt'):
        self.model = YOLO(model)

    def ret_all_box(self,results):
        boxes = []
        for res in results:
            boxe = res.boxes.xyxy.cpu().numpy()
            boxes.append(boxe)

        return boxes

    def __call__(self, images):
        results = self.model.predict(images,classes=[0],half=True)
        # boxe = results[0].boxes.xyxy.cpu().numpy()
        if len(results) == 0 :
            return None
        else:
            return self.ret_all_box(results)


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