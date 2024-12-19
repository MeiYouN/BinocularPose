import yaml
import numpy as np
from BinocularPose.mytools.camera_utils import *
from plotly.graph_objs.icicle import Pathbar


def load_cameras(file_path):
    return read_cameras(file_path)

def convert_camera_data(camera_data):
    # 初始化结果字典
    result = {
        'K': [],
        'R': [],
        'T': [],
        'dist': [],
        'P': []
    }
    # 遍历每个摄像头的数据
    for name, data in camera_data.items():
        result['K'].append(data['K'])
        result['R'].append(data['R'])
        result['T'].append(data['T'])
        result['dist'].append(data['dist'])
        result['P'].append(data['P'])
    # 将列表转换为 NumPy 数组
    result['K'] = np.array(result['K'])
    result['R'] = np.array(result['R'])
    result['T'] = np.array(result['T'])
    result['dist'] = np.array(result['dist'])
    result['P'] = np.array(result['P'])
    return result

def compute_projection_matrix(K, R, T):
    RT = np.hstack((R, T))
    P = K @ RT
    return P

def load_yml(Path):
    cam_data = load_cameras(Path)
    cam = convert_camera_data(cam_data)
    return cam

def main():
    # 读取配置文件
    cam_data= load_cameras('D:\Desktop\EveryThing\EasyMocap\mydata\data10\\video')
    cam = convert_camera_data(cam_data)

    print(cam)


if __name__ == '__main__':
    main()