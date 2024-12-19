import queue
import threading
from datetime import datetime
import json
import os
import numpy as np


class JsonFile:

    data = {
        'folder_path': '',
        'framers': 0,
        'datetime': '',
        'fps': 0,
        'pose_data': [],
    }

    def __init__(self,folder_path ,save_path, fps=0):
        current_datetime = datetime.now()
        self.save_path =  os.path.join(save_path, f"run_{current_datetime.strftime('%Y%m%d%H%M%S')}.json")
        self.data['datetime'] = current_datetime.strftime('%Y-%m-%d %H:%M:%S')
        self.data['folder_path'] = folder_path
        self.data['fps'] = fps
        self.index = 0
        # self.data_queue = queue.Queue()
        # self.save_thread = threading.Thread(target=self._save_loop, daemon=True)
        # self.save_thread.start()

    def update(self, pose_data=None):
        isvis = True
        self.index += 1
        if pose_data is None:
            pose_data = []
            isvis = False
        pose_data_node = {
            'id': self.index,
            'isvis': isvis,
            'pose': pose_data,
        }

        self.data['pose_data'].append(pose_data_node)
        # self.data_queue.put(pose_data_node)

    def _save_loop(self):
        while True:
            if not self.data_queue.empty():
                pose_data = self.data_queue.get()
                self.data['pose_data'].append(pose_data)
                self.save()

    def save(self):
        with open(self.save_path, "w", encoding="utf-8") as fp:
            json.dump(self.data, fp, indent=4)
