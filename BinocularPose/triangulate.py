import numpy as np

from BinocularPose.mytools.camera_utils import Undistort


def batch_triangulate(keypoints_, Pall, min_view=2)->np.ndarray:
    """ triangulate the keypoints of whole body

    Args:
        keypoints_ (nViews, nJoints, 3): 2D detections
        Pall (nViews, 3, 4): projection matrix of each view
        min_view (int, optional): min view for visible points. Defaults to 2.

    Returns:
        keypoints3d: (nJoints, 4)
    """
    # keypoints: (nViews, nJoints, 3)
    # Pall: (nViews, 3, 4)
    # A: (nJoints, nViewsx2, 4), x: (nJoints, 4, 1); b: (nJoints, nViewsx2, 1)
    v = (keypoints_[:, :, -1]>0).sum(axis=0)
    valid_joint = np.where(v >= min_view)[0]
    keypoints = keypoints_[:, valid_joint]
    conf3d = keypoints[:, :, -1].sum(axis=0)/v[valid_joint]
    # P2: P矩阵的最后一行：(1, nViews, 1, 4)
    P0 = Pall[None, :, 0, :]
    P1 = Pall[None, :, 1, :]
    P2 = Pall[None, :, 2, :]
    # uP2: x坐标乘上P2: (nJoints, nViews, 1, 4)
    uP2 = keypoints[:, :, 0].T[:, :, None] * P2
    vP2 = keypoints[:, :, 1].T[:, :, None] * P2
    conf = keypoints[:, :, 2].T[:, :, None]
    Au = conf * (uP2 - P0)
    Av = conf * (vP2 - P1)
    A = np.hstack([Au, Av])
    u, s, v = np.linalg.svd(A)
    X = v[:, -1, :]
    X = X / X[:, 3:]
    # out: (nJoints, 4)
    result = np.zeros((keypoints_.shape[1], 4))
    result[valid_joint, :3] = X[:, :3]
    result[valid_joint, 3] = conf3d #* (conf[..., 0].sum(axis=-1)>min_view)
    return result


class SimpleTriangulate:
    def __init__(self, mode='naive'):
        self.mode = mode

    @staticmethod
    def undistort(points, cameras):
        nViews = len(points)
        pelvis_undis = []
        for nv in range(nViews):
            camera = {key: cameras[key][nv] for key in ['R', 'T', 'K', 'dist']}
            if points[nv].shape[0] > 0:
                pelvis = Undistort.points(points[nv], camera['K'], camera['dist'])
            else:
                pelvis = points[nv].copy()
            pelvis_undis.append(pelvis)
        return pelvis_undis

    def __call__(self, keypoints:np.ndarray, cameras:dict)->np.ndarray:
        '''
            keypoints: [nViews, nJoints, 3]
        output:
            keypoints3d: (nJoints, 4)
        '''
        # print(keypoints.shape)
        keypoints = self.undistort(keypoints, cameras)
        keypoints = np.stack(keypoints)
        if self.mode == 'naive':
            keypoints3d = batch_triangulate(keypoints, cameras['P'])
            # print(1)
        else:
            keypoints3d, k2d = iterative_triangulate(keypoints, cameras['P'], dist_max=25)
            print(2)
        # print(keypoints3d)
        return keypoints3d






