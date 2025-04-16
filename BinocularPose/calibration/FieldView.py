# 简单的例子，展示如何使用open3d的create_camera_visualization函数来生成相机的可视化线段.
# 这里我hardcode了一组随机的相机参数。
import open3d as o3d
import numpy as np

WIDTH = 1280
HEIGHT = 720

img_w = 2048
img_h = 1536

def extend_camera_cone_lines(camera_lineset, extension_factor=5.0):
    # 获取 LineSet 的顶点和边
    points = np.asarray(camera_lineset.points)
    lines = np.asarray(camera_lineset.lines)

    # 提取原始顶点：假设有 5 个顶点 (相机中心 + 4个角投影点)
    camera_center = points[0]
    corner_points = points[1:]

    # 计算延伸后的顶点位置
    extended_corner_points = []
    for corner in corner_points:
        # 计算从相机中心到每个角点的方向向量，并按比例延长
        direction_vector = corner - camera_center
        extended_corner = camera_center + extension_factor * direction_vector
        extended_corner_points.append(extended_corner)

    return camera_center, np.array(corner_points), np.array(extended_corner_points)

def compute_intersections(corner_points, extended_corner_points):
    lines = [[0,1], [1,2], [2,3], [3,0], [4,5], [5,6], [6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
    newlines = []
    points = np.concatenate((corner_points, extended_corner_points))
    count = points.shape[0]

    epsilon = 1e-9  # 浮点误差容忍度
    intersections = []
    lenl = len(lines)
    for idx in range(lenl):
        line = lines[idx]
        i, j = line
        p0, p1 = points[i], points[j]
        z0, z1 = p0[2], p1[2]
        # 检查是否同侧
        if z0 * z1 >= 0:
            continue
        else:
            t = -z0 / (z1 - z0)
            if 0 <= t <= 1:
                x = p0[0] + t * (p1[0] - p0[0])
                y = p0[1] + t * (p1[1] - p0[1])
                pz0 = np.array([x, y, 0.0])
                points = np.append(points, [pz0], axis=0)
                intersections.append(count)

                if z0 > 0:
                    lines.append([i, count])
                elif z0 < 0:
                    lines.append([j, count])
                count += 1

    for idx,p in enumerate(points):
        if p[2] < 0:
            lines = [sub for sub in lines if idx not in sub]

    comb_manual = []
    n = len(intersections)
    for i in range(n):
        for j in range(i+1, n):
            comb_manual.append([intersections[i], intersections[j]])
    lines += comb_manual

    return lines, points

def create_extend_lineset(corner_points, extended_corner_points):
    lines, points = compute_intersections(corner_points, extended_corner_points)

    colors = [[1, 0, 0] for i in range(len(lines))]

    extend_lineset = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    extend_lineset.colors = o3d.utility.Vector3dVector(colors)
    return extend_lineset

def create_scene():
    vertices = np.array([
        [-5.0, -5.0, 0.0],  # 顶点 0
        [5.0, -5.0, 0.0],  # 顶点 1
        [5.0, 5.0, 0.0],  # 顶点 2
        [-5.0, 5.0, 0.0]  # 顶点 3
    ])
    triangles = np.array([
        [0, 1, 2],  # 第一个三角形
        [0, 2, 3]  # 第二个三角形
    ])
    # 创建一个 TriangleMesh 对象
    mesh_plane = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), o3d.utility.Vector3iVector(triangles))
    coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])

    return mesh_plane, coor

def make_one_view(w, h, intri, extri):
    CamLineset = o3d.geometry.LineSet.create_camera_visualization(view_width_px=w, view_height_px=h,
                                                                    intrinsic=intri[:3, :3], extrinsic=extri)
    camera_center, corner_points, extended_corners = extend_camera_cone_lines(CamLineset, extension_factor=5.0)
    extend_lineset = create_extend_lineset(corner_points, extended_corners)
    return extend_lineset, CamLineset


# an random camera intrinsic matrix
intrinsics1 = np.array([
    [938.153000, 0.00000000e+00, 1031.972935, 0.00000000e+00],
    [0.00000000e+00, 934.152574, 789.459426, 0.00000000e+00],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

# an random camera extrinsic matrix, i.e., w2c matrix
extrinsic1 = np.array([
    [0.972707, -0.220402, -0.072554 ,  -0.351848],
    [-0.108824, -0.157157, -0.981560 , 1.040511],
    [0.204935, 0.962665, -0.176852 , 2.138623],
    [ 0.      ,  0.       ,  0.       , 1.      ]])


intrinsics2 = np.array([
    [951.644933, 0.00000000e+00, 991.163254, 0.00000000e+00],
    [0.00000000e+00, 948.381818, 776.141048, 0.00000000e+00],
    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
    [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
    ])

# an random camera extrinsic matrix, i.e., w2c matrix
extrinsic2 = np.array([
    [0.873040, 0.487591, -0.007467 ,  -0.102270],
    [0.098889, -0.192016, -0.976397 , 0.775004],
    [-0.477516, 0.851695, -0.215856 , 2.283383],
    [ 0.      ,  0.       ,  0.       , 1.      ]])


def main():
    extend_lineset1, cameraLines1 = make_one_view(img_w,img_h,intrinsics1, extrinsic1)
    extend_lineset2, cameraLines2 = make_one_view(img_w,img_h,intrinsics2, extrinsic2)

    mesh_plane, coor = create_scene()

    vizualizer = o3d.visualization.Visualizer()
    vizualizer.create_window(width=WIDTH, height=HEIGHT)

    vizualizer.add_geometry(mesh_plane)
    vizualizer.add_geometry(cameraLines1)
    vizualizer.add_geometry(cameraLines2)
    vizualizer.add_geometry(extend_lineset1)
    vizualizer.add_geometry(extend_lineset2)
    vizualizer.add_geometry(coor)

    vizualizer.run()
    vizualizer.destroy_window()

if __name__ == '__main__':
    main()
