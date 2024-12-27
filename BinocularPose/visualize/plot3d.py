import matplotlib.pyplot as plt


connnection = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
               [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4]]

class vis_plot:

    def __init__(self):
        self.fig = plt.figure(figsize=(5, 5))
        self.ax = self.fig.add_subplot(111, projection="3d")


    def show(self, keypoints3d):
        self.ax.clear()
        self.ax.set_xlim3d(-1, 1)
        self.ax.set_ylim3d(-1, 1)
        self.ax.set_zlim3d(0, 1)
        self.ax.scatter(keypoints3d[:, 0], keypoints3d[:, 1], keypoints3d[:, 2], s=10)
        for _c in connnection:
            self.ax.plot([keypoints3d[_c[0], 0], keypoints3d[_c[1], 0]],
                    [keypoints3d[_c[0], 1], keypoints3d[_c[1], 1]],
                    [keypoints3d[_c[0], 2], keypoints3d[_c[1], 2]], 'g')
        plt.pause(0.01)

