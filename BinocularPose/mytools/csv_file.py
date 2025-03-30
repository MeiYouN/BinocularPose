
from datetime import datetime
import os
import numpy as np
import pandas as pd

COCO_JOINTS = [
    "Nose",          # 0
    "L_Eye",         # 1
    "R_Eye",         # 2
    "L_Ear",         # 3
    "R_Ear",         # 4
    "L_Shoulder",    # 5
    "R_Shoulder",    # 6
    "L_Elbow",       # 7
    "R_Elbow",       # 8
    "L_Wrist",       # 9
    "R_Wrist",       # 10
    "L_Hip",         # 11
    "R_Hip",         # 12
    "L_Knee",        # 13
    "R_Knee",        # 14
    "L_Ankle",       # 15
    "R_Ankle"        # 16
]

class CsvFile:

    data = []
    data_title = ['Nose_x', 'Nose_y', 'Nose_z', 'Nose_score',
                  'Leye_x', 'Leye_y', 'Leye_z', 'Leye_score',
                  'Reye_x', 'Reye_y', 'Reye_z', 'Reye_score',
                  'Lear_x', 'Lear_y', 'Lear_z', 'Lear_score',
                  'Rear_x', 'Rear_y', 'Rear_z', 'Rear_score',
                  'Lshoulder_x', 'Lshoulder_y', 'Lshoulder_z', 'Lshoulder_score',
                  'Rshoulder_x', 'Rshoulder_y', 'Rshoulder_z', 'Rshoulder_score',
                  'Lelbow_x', 'Lelbow_y', 'Lelbow_z', 'Lelbow_score',
                  'Relbow_x', 'Relbow_y', 'Relbow_z', 'Relbow_score',
                  'Lwrist_x', 'Lwrist_y', 'Lwrist_z', 'Lwrist_score',
                  'Rwrist_x', 'Rwrist_y', 'Rwrist_z', 'Rwrist_score',
                  'Lhip_x', 'Lhip_y', 'Lhip_z', 'Lhip_score',
                  'Rhip_x', 'Rhip_y', 'Rhip_z', 'Rhip_score',
                  'Lknee_x', 'Lknee_y', 'Lknee_z', 'Lknee_score',
                  'Rknee_x', 'Rknee_y', 'Rknee_z', 'Rknee_score',
                  'Lankle_x', 'Lankle_y', 'Lankle_z', 'Lankle_score',
                  'Rankle_x', 'Rankle_y', 'Rankle_z', 'Rankle_score',]

    def __init__(self, folder_path=None, save_path=None, fps=0):
        current_datetime = datetime.now()
        if save_path is not None:
            self.save_path = save_path
        else:
            self.save_path = os.path.join(folder_path, f"run_{current_datetime.strftime('%Y%m%d%H%M%S')}.csv")


    def update(self, pose_data:np.ndarray):
        dataone = [j for i in pose_data for j in i]
        self.data.append(dataone)


    def save(self):
        test = pd.DataFrame(columns=self.data_title, data=self.data)
        test.to_csv(self.save_path, index=False)


