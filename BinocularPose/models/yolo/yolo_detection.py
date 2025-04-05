import os.path
import random
import cv2
import numpy as np
import torch
import torchvision

from ultralytics.nn.autobackend import AutoBackend
import torch.nn.functional as F


class YOLODetectionInfer:
    def __init__(self, weights, cuda, conf_thres, iou_thres) -> None:
        self.imgsz = 640
        self.device = cuda
        self.model = AutoBackend(weights, device=torch.device(cuda))
        self.model.eval()
        self.names = self.model.names
        self.conf = conf_thres
        self.iou = iou_thres
        self.color = {"font": (255, 255, 255)}
        self.color.update(
            {self.names[i]: (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
             for i in range(len(self.names))})

    def infer(self, img_path, save_path):
        img_src = cv2.imread(img_path)
        img_array = np.array([img_src])

        img = self.tensor_process(img_array)
        preds = self.model(img)
        results = self.non_max_suppression(preds, img.shape[2:], img_src.shape, self.conf, self.iou, nc=len(self.names))

        for result in results:
            self.draw_box(img_array[int(result[6])], result[:4], result[4], self.names[result[5]])

        for i in range(img_array.shape[0]):
            cv2.imwrite(os.path.join(save_path, f"{i}.jpg"), img_array[i])

    def draw_box(self, img_src, box, conf, cls_name):
        lw = max(round(sum(img_src.shape) / 2 * 0.003), 2)  # line width
        tf = max(lw - 1, 1)  # font thickness
        sf = lw / 3  # font scale

        color = self.color[cls_name]
        label = f'{cls_name} {conf:.4f}'
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        # 绘制矩形框
        cv2.rectangle(img_src, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        # text width, height
        w, h = cv2.getTextSize(label, 0, fontScale=sf, thickness=tf)[0]
        # label fits outside box
        outside = box[1] - h - 3 >= 0
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        # 绘制矩形框填充
        cv2.rectangle(img_src, p1, p2, color, -1, cv2.LINE_AA)
        # 绘制标签
        cv2.putText(img_src, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                    0, sf, self.color["font"], thickness=2, lineType=cv2.LINE_AA)

    def tensor_process(self, image_cv):
        img_shape = image_cv.shape[1:]
        new_shape = [640, 640]
        r = min(new_shape[0] / img_shape[0], new_shape[1] / img_shape[1])
        # Compute padding
        new_unpad = int(round(img_shape[0] * r)), int(round(img_shape[1] * r))

        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        dw /= 2  # divide padding into 2 sides
        dh /= 2

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        padding_value = 114

        # Convert
        image_cv = image_cv[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
        image_cv = np.ascontiguousarray(image_cv)  # contiguous
        image_tensor = torch.from_numpy(image_cv).float()
        image_tensor = image_tensor.to(self.device)

        resized_tensor = F.interpolate(image_tensor, size=new_unpad, mode='bilinear', align_corners=False)
        padded_tensor = F.pad(resized_tensor, (top, bottom, left, right), mode='constant', value=padding_value)
        infer_tensor = padded_tensor / 255.0

        return infer_tensor

    def non_max_suppression(self, prediction, inferShape, orgShape, conf_thres=0.25, iou_thres=0.45, agnostic=True,
                            multi_label=False,
                            max_wh=7680, nc=0):
        prediction = prediction[0]  # select only inference output

        nc = nc  # number of classes
        nm = prediction.shape[1] - nc - 4
        mi = 4 + nc  # mask start index
        xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

        # Settings
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

        prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
        prediction[..., :4] = self.xywh2xyxy(prediction[..., :4])  # xywh to xyxy

        true_indices = torch.nonzero(xc)
        selected_rows = prediction[true_indices[:, 0], true_indices[:, 1]]
        new_prediction = torch.cat((selected_rows, true_indices[:, 0].unsqueeze(1).float()), dim=1)

        if new_prediction.shape[0] == 0:
            return

        box, cls, mask, idxs = new_prediction.split((4, nc, nm, 1), 1)
        conf, j = cls.max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.squeeze(-1) > conf_thres]
        if not x.shape[0]:  # no boxes
            return

        cls = x[:, 5]  # classes
        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        idxs = idxs.t().squeeze(0)

        keep = torchvision.ops.batched_nms(boxes, scores, idxs, iou_thres)

        boxes[keep] = self.scale_boxes(inferShape, boxes[keep], orgShape)

        boxes = boxes[keep].cpu().numpy()
        scores = scores[keep].cpu().numpy()
        cls = cls[keep].cpu().numpy()
        idxs = idxs[keep].cpu().numpy()

        results = np.hstack((boxes, np.expand_dims(scores, axis=1)))
        results = np.hstack((results, np.expand_dims(cls, axis=1)))
        results = np.hstack((results, np.expand_dims(idxs, axis=1)))
        return results

    def xywh2xyxy(self, x):
        assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
        y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
        dw = x[..., 2] / 2  # half-width
        dh = x[..., 3] / 2  # half-height
        y[..., 0] = x[..., 0] - dw  # top left x
        y[..., 1] = x[..., 1] - dh  # top left y
        y[..., 2] = x[..., 0] + dw  # bottom right x
        y[..., 3] = x[..., 1] + dh  # bottom right y
        return y

    def clip_boxes(self, boxes, shape):
        if isinstance(boxes, torch.Tensor):  # faster individually (WARNING: inplace .clamp_() Apple MPS bug)
            boxes[..., 0] = boxes[..., 0].clamp(0, shape[1])  # x1
            boxes[..., 1] = boxes[..., 1].clamp(0, shape[0])  # y1
            boxes[..., 2] = boxes[..., 2].clamp(0, shape[1])  # x2
            boxes[..., 3] = boxes[..., 3].clamp(0, shape[0])  # y2
        else:  # np.array (faster grouped)
            boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])  # x1, x2
            boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])  # y1, y2
        return boxes

    def scale_boxes(self, img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
            pad = (
                round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
                round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
            )  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        if padding:
            boxes[..., 0] -= pad[0]  # x padding
            boxes[..., 1] -= pad[1]  # y padding
            if not xywh:
                boxes[..., 2] -= pad[0]  # x padding
                boxes[..., 3] -= pad[1]  # y padding
        boxes[..., :4] /= gain
        return self.clip_boxes(boxes, img0_shape)


if __name__ == '__main__':
    weights = r'yolov8n.pt'
    cuda = 'cuda:0'
    save_path = "./runs"

    if not os.path.exists(save_path):
        os.mkdir(save_path)

    model = YOLODetectionInfer(weights, cuda, 0.25, 0.45)

    img_path = r'./ultralytics/assets/bus.jpg'
    model.infer(img_path, save_path)