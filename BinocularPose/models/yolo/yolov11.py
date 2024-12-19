from ultralytics import YOLO

class YoloV11:
    def __init__(self, model='./weights/yolo11m-pose.pt'):
        self.model = YOLO(model)

    def __call__(self, img):
        results = self.model.predict(img)
        keypoint = results[0].keypoints.data.cpu().numpy()[0]
        return keypoint



def train():
    model = YOLO("./weights/yolo11m-pose.pt")  # load a pretrained model (recommended for training)

    results = model.train(data="coco-pose.yaml", epochs=100, imgsz=640)

if __name__ == '__main__':
    train()