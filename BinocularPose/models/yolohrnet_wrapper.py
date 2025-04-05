from glob import glob
from os.path import join
from tqdm import tqdm
import os
import cv2
import numpy as np


def extract_yolo_hrnet(image_root, annot_root, ext, config_yolo, config_hrnet):
    config_yolo.pop('ext', None)
    imgnames = sorted(glob(join(image_root, '*{}'.format(ext))))
    import torch
    device = torch.device('cuda')
    from .yolo.yolo_det import Yolo_Det
    device = torch.device('cuda') \
            if torch.cuda.is_available() else torch.device('cpu')
    detector = Yolo_Det(device=device, **config_yolo)
    from .hrnet import SimpleHRNet
    estimator = SimpleHRNet(device=device, **config_hrnet)

    for nf, imgname in enumerate(tqdm(imgnames, desc=os.path.basename(image_root))):
        detections = detector(image_rgb)
        # forward_hrnet
        points2d = estimator.predict(image_rgb, detections)
        annots = []
        for i in range(len(detections)):
            annot_ = {
                'bbox': [float(d) for d in detections[i]],
                'keypoints': points2d[i],
                'isKeyframe': False
            }
            annot_['area'] = max(annot_['bbox'][2] - annot_['bbox'][0], annot_['bbox'][3] - annot_['bbox'][1])**2
            annots.append(annot_)
        annots.sort(key=lambda x:-x['area'])