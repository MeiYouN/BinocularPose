import os.path
import time
from argparse import ArgumentParser
import numpy as np
from BinocularPose.models.yolo.yolo_det import Yolo_Det
import cv2
from mmpose.apis import inference_topdown, init_model,MMPoseInferencer
from mmpose.utils import register_all_modules


filter_args = dict(bbox_thr=0.3, nms_thr=0.3, pose_based_nms=False)
POSE2D_SPECIFIC_ARGS = dict(
    yoloxpose=dict(bbox_thr=0.01, nms_thr=0.65, pose_based_nms=True),
    rtmo=dict(bbox_thr=0.1, nms_thr=0.65, pose_based_nms=True),
)

def parse_args():
    parser = ArgumentParser()

    # init args
    parser.add_argument(
        '--pose2d',
        type=str,
        default=None,
        help='Pretrained 2D pose estimation algorithm. It\'s the path to the '
        'config file or the model name defined in metafile.')
    parser.add_argument(
        '--pose2d-weights',
        type=str,
        default=None,
        help='Path to the custom checkpoint file of the selected pose model. '
        'If it is not specified and "pose2d" is a model name of metafile, '
        'the weights will be loaded from metafile.')
    parser.add_argument(
        '--det-model',
        type=str,
        default=None,
        help='Config path or alias of detection model.')
    parser.add_argument(
        '--det-weights',
        type=str,
        default=None,
        help='Path to the checkpoints of detection model.')
    parser.add_argument(
        '--det-cat-ids',
        type=int,
        nargs='+',
        default=0,
        help='Category id for detection model.')
    parser.add_argument(
        '--scope',
        type=str,
        default='mmpose',
        help='Scope where modules are defined.')
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
        help='Device used for inference. '
        'If not specified, the available device will be automatically used.')

    # The default arguments for prediction filtering differ for top-down
    # and bottom-up models. We assign the default arguments according to the
    # selected pose2d model
    args, _ = parser.parse_known_args()
    for model in POSE2D_SPECIFIC_ARGS:
        if args.pose2d is not None and model in args.pose2d:
            filter_args.update(POSE2D_SPECIFIC_ARGS[model])
            break

    # call args
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display the image/video in a popup window.')
    parser.add_argument(
        '--draw-bbox',
        action='store_true',
        help='Whether to draw the bounding boxes.')
    parser.add_argument(
        '--draw-heatmap',
        action='store_true',
        default=False,
        help='Whether to draw the predicted heatmaps.')
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=filter_args['bbox_thr'],
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=filter_args['nms_thr'],
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--pose-based-nms',
        type=lambda arg: arg.lower() in ('true', 'yes', 't', 'y', '1'),
        default=filter_args['pose_based_nms'],
        help='Whether to use pose-based NMS')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--use-oks-tracking',
        action='store_true',
        default=True,
        help='Whether to use OKS as similarity in tracking')
    parser.add_argument(
        '--disable-norm-pose-2d',
        action='store_true',
        help='Whether to scale the bbox (along with the 2D pose) to the '
        'average bbox scale of the dataset, and move the bbox (along with the '
        '2D pose) to the average bbox center of the dataset. This is useful '
        'when bbox is small, especially in multi-person scenarios.')
    parser.add_argument(
        '--disable-rebase-keypoint',
        action='store_true',
        default=False,
        help='Whether to disable rebasing the predicted 3D pose so its '
        'lowest keypoint has a height of 0 (landing on the ground). Rebase '
        'is useful for visualization when the model do not predict the '
        'global position of the 3D pose.')
    parser.add_argument(
        '--num-instances',
        type=int,
        default=1,
        help='The number of 3D poses to be visualized in every frame. If '
        'less than 0, it will be set to the number of pose results in the '
        'first frame.')
    parser.add_argument(
        '--radius',
        type=int,
        default=3,
        help='Keypoint radius for visualization.')
    parser.add_argument(
        '--thickness',
        type=int,
        default=1,
        help='Link thickness for visualization.')
    parser.add_argument(
        '--skeleton-style',
        default='mmpose',
        type=str,
        choices=['mmpose', 'openpose'],
        help='Skeleton style selection')
    parser.add_argument(
        '--black-background',
        action='store_true',
        help='Plot predictions on a black image')
    parser.add_argument(
        '--vis-out-dir',
        type=str,
        default='',
        help='Directory for saving visualized results.')
    parser.add_argument(
        '--pred-out-dir',
        type=str,
        default='',
        help='Directory for saving inference results.')

    call_args = vars(parser.parse_args())

    init_kws = [
        'pose2d', 'pose2d_weights', 'scope', 'device', 'det_model',
        'det_weights', 'det_cat_ids', 'pose3d', 'pose3d_weights',
        'show_progress'
    ]
    init_args = {}
    for init_kw in init_kws:
        init_args[init_kw] = call_args.pop(init_kw)


    return init_args, call_args

def parse_args_call():
    parser = ArgumentParser()
    parser.add_argument(
        '--bbox-thr',
        type=float,
        default=filter_args['bbox_thr'],
        help='Bounding box score threshold')
    parser.add_argument(
        '--nms-thr',
        type=float,
        default=filter_args['nms_thr'],
        help='IoU threshold for bounding box NMS')
    parser.add_argument(
        '--pose-based-nms',
        type=lambda arg: arg.lower() in ('true', 'yes', 't', 'y', '1'),
        default=filter_args['pose_based_nms'],
        help='Whether to use pose-based NMS')
    parser.add_argument(
        '--kpt-thr', type=float, default=0.3, help='Keypoint score threshold')
    parser.add_argument(
        '--tracking-thr', type=float, default=0.3, help='Tracking threshold')
    parser.add_argument(
        '--use-oks-tracking',
        action='store_true',
        default=True,
        help='Whether to use OKS as similarity in tracking')

    return  vars(parser.parse_args())

class MyMMP:
    def __init__(self, model_path):
        register_all_modules()
        config_file = os.path.join(model_path, 'rtmo-m_16xb16-600e_body7-640x640.py')
        checkpoint_file = os.path.join(model_path, 'rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth')
        self.model = init_model(config_file, checkpoint_file, device='cuda:0')  # or device='cuda:0'
        # self.model = init_model(**self.init_args)


    def select_bbox(self,result):
        keypoints_data = result[0].pred_instances
        max_key_index = np.argmax(keypoints_data.bbox_scores)
        if keypoints_data.bbox_scores[max_key_index] <= 0.50 :
            return None
        keypoint = keypoints_data.keypoints[max_key_index]
        keypoint_score = keypoints_data.keypoint_scores[max_key_index]
        return np.c_[keypoint, keypoint_score]

    def __call__(self, frame, bbox):
        # output = self.inference_topdown(inputs=frame ,**self.call_args)
        results = inference_topdown(self.model, frame, bbox, 'xyxy')
        return self.select_bbox(results)

class MyMMP1:
    def __init__(self):
        self.call_args = parse_args_call()
        register_all_modules()
        config_file = 'rtmo-m_16xb16-600e_body7-640x640.py'
        checkpoint_file = 'rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth'
        self.model = MMPoseInferencer(config_file, checkpoint_file, device='cuda:0', det_cat_ids=[0], )

    def __call__(self, frame):
        # output = self.inference_topdown(inputs=frame ,**self.call_args)
        results = self.model(frame, **self.call_args)

        return next(results)[0][0]


def main():
    left_video = "D:\Desktop\EveryThing\WorkProject\ThreeD_demo\data10\pose\\01.mp4"
    capL = cv2.VideoCapture(left_video)

    yolo = Yolo_Det()
    model = MyMMP()

    while True:
        retl, framel = capL.read()
        start_time = time.time()
        # 执行模型训练
        bbox = yolo(framel)
        if bbox is not None:
            print(bbox)
            results = model(framel,bbox)
            # print(results['predictions'])
            print(results)

        end_time = time.time()
        runtime = end_time - start_time
        print(runtime)




if __name__ == '__main__':
    main()
