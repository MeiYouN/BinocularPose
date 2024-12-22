# detect the corner of chessboard
from torch.utils.tensorboard.summary import image

from BinocularPose.mytools.file_utils import getFileList, read_json, save_json
from tqdm import tqdm
from BinocularPose.mytools.basic_utils import ImageFolder
import numpy as np
from os.path import join
import cv2
import os
import func_timeout
import threading
from func_timeout import func_set_timeout


def _findChessboardCorners(img, pattern, debug):
    "basic function"
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    retval, corners = cv2.findChessboardCorners(img, pattern,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_FILTER_QUADS)
    if not retval:
        return False, None
    corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
    corners = corners.squeeze()
    return True, corners

def _findChessboardCornersAdapt(img, pattern, debug):
    "Adapt mode"
    img = cv2.adaptiveThreshold(img, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY, 21, 2)
    return _findChessboardCorners(img, pattern, debug)

@func_set_timeout(5)
def findChessboardCorners(img, annots, pattern, debug=False):
    conf = sum([v[2] for v in annots['keypoints2d']])
    if annots['visited'] and conf > 0:
        return True
    elif annots['visited']:
        return None
    annots['visited'] = True
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Find the chess board corners
    for func in [_findChessboardCorners, _findChessboardCornersAdapt]:
        ret, corners = func(gray, pattern, debug)
        if ret:break
    else:
        return None
    # found the corners
    show = img.copy()
    show = cv2.drawChessboardCorners(show, pattern, corners, ret)
    assert corners.shape[0] == len(annots['keypoints2d'])
    corners = np.hstack((corners, np.ones((corners.shape[0], 1))))
    annots['keypoints2d'] = corners.tolist()
    return show

def getChessboard3d(pattern, gridSize, axis='yx'):
    # 注意：这里为了让标定板z轴朝上，设定了短边是x，长边是y
    template = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1,2)
    object_points = np.zeros((pattern[1]*pattern[0], 3), np.float32)
    # 长边是x,短边是z
    if axis == 'xz':
        object_points[:, 0] = template[:, 0]
        object_points[:, 2] = template[:, 1]
    elif axis == 'yx':
        object_points[:, 0] = template[:, 1]
        object_points[:, 1] = template[:, 0]
    else:
        raise NotImplementedError
    object_points = object_points * gridSize
    return object_points

def create_chessboard(path, image, pattern, gridSize, ext, axis, overwrite=True):
    print('Create chessboard {}'.format(pattern))
    keypoints3d = getChessboard3d(pattern, gridSize=gridSize, axis=axis)
    keypoints2d = np.zeros((keypoints3d.shape[0], 3))
    imgnames = getFileList(join(path, image), ext=ext)
    template = {
        'keypoints3d': keypoints3d.tolist(),
        'keypoints2d': keypoints2d.tolist(),
        'pattern': pattern,
        'grid_size': gridSize,
        'visited': False
    }
    for imgname in tqdm(imgnames, desc='create template chessboard'):
        annname = imgname.replace(ext, '.json')
        annname = join(path, 'chessboard', annname)
        if os.path.exists(annname) and overwrite:
            # 覆盖keypoints3d
            data = read_json(annname)
            data['keypoints3d'] = template['keypoints3d']
            save_json(annname, data)
        elif os.path.exists(annname) and not overwrite:
            continue
        else:
            save_json(annname, template)

def _detect_chessboard(datas, path, image, out, pattern):
    for imgname, annotname in datas:
        # imgname, annotname = dataset[i]
        # detect the 2d chessboard
        img = cv2.imread(imgname)
        annots = read_json(annotname)
        try:
            show = findChessboardCorners(img, annots, pattern)
        except func_timeout.exceptions.FunctionTimedOut:
            show = None
        save_json(annotname, annots)
        if show is None:
            print('[Info] Cannot find chessboard in {}'.format(imgname))
            continue
        outname = join(out, imgname.replace(path + '/{}/'.format(image), ''))
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        if isinstance(show, np.ndarray):
            cv2.imwrite(outname, show)

def detect_chessboard(path, image, out, pattern, gridSize, args):
    create_chessboard(path, image, pattern, gridSize, ext=args.ext, axis=args.axis ,overwrite=args.overwrite3d)
    dataset = ImageFolder(path, image=image, annot='chessboard', ext=args.ext)
    dataset.isTmp = False
    trange = list(range(len(dataset)))
    threads = []
    for i in range(args.mp):
        ranges = trange[i::args.mp]
        datas = [dataset[t] for t in ranges]
        thread = threading.Thread(target=_detect_chessboard, args=(datas, path, image, out, pattern)) # 应该不存在任何数据竞争
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
   

def _detect_by_search(path, image, out, pattern, sub, args):
    dataset = ImageFolder(path, sub=sub, annot='chessboard', ext=args.ext)
    dataset.isTmp = False
    nFrames = len(dataset)
    found = np.zeros(nFrames, dtype=bool)
    visited = np.zeros(nFrames, dtype=bool)
    proposals = []
    init_step = args.max_step
    min_step = args.min_step
    for nf in range(0, nFrames, init_step):
        if nf + init_step < len(dataset):
            proposals.append([nf, nf+init_step])
    while len(proposals) > 0:
        left, right = proposals.pop(0)
        print('[detect] {} {:4.1f}% Check [{:5d}, {:5d}]'.format(
            sub, visited.sum()/visited.shape[0]*100, left, right), end=' ')
        for nf in [left, right]:
            if not visited[nf]:
                visited[nf] = True
                imgname, annotname = dataset[nf]
                # detect the 2d chessboard
                img = cv2.imread(imgname)
                annots = read_json(annotname)
                try:
                    show = findChessboardCorners(img, annots, pattern)
                except func_timeout.exceptions.FunctionTimedOut:
                    show = None
                save_json(annotname, annots)
                if show is None:
                    if args.debug:
                        print('[Info] Cannot find chessboard in {}'.format(imgname))
                    found[nf] = False
                    continue
                found[nf] = True
                outname = join(out, imgname.replace(path + '{}{}{}'.format(os.sep, image, os.sep), ''))
                os.makedirs(os.path.dirname(outname), exist_ok=True)
                if isinstance(show, np.ndarray):
                    cv2.imwrite(outname, show)
        print('{}-{}'.format('o' if found[left] else 'x', 'o' if found[right] else 'x'))
        if not found[left] and not found[right]:
            visited[left:right] = True
            continue
        mid = (left+right)//2
        if mid == left or mid == right:
            continue
        if mid - left > min_step:
            proposals.append((left, mid))
        if right - mid > min_step:
            proposals.append((mid, right))

def detect_chessboard_sequence(path, out, pattern, gridSize, args, image="images"):
    create_chessboard(path, image, pattern, gridSize, ext=args.ext, overwrite=args.overwrite3d)
    subs = sorted(os.listdir(join(path, image)))
    subs = [s for s in subs if os.path.isdir(join(path, image, s))]
    if len(subs) == 0:
        subs = [None]
    from multiprocessing import Process
    tasks = []
    for sub in subs:
        task = Process(target=_detect_by_search, args=(path, image, out, pattern, sub, args))
        task.start()
        tasks.append(task)
    for task in tasks:
        task.join()
    for sub in subs:
        dataset = ImageFolder(path, sub=sub, annot='chessboard', ext=args.ext)
        dataset.isTmp = False
        count, visited = 0, 0
        for nf in range(len(dataset)):
            imgname, annotname = dataset[nf]
            # detect the 2d chessboard
            annots = read_json(annotname)
            if annots['visited']:
                visited += 1
            if annots['keypoints2d'][0][-1] > 0.01:
                count += 1
        print('{}: found {:4d}/{:4d} frames'.format(sub, count, visited))

def check_chessboard(path, out):
    subs_notvalid = []
    for sub in sorted(os.listdir(join(path, 'images'))):
        if os.path.exists(join(out, sub)):
            continue
        subs_notvalid.append(sub)
    print(subs_notvalid)
    print('Cannot find chessboard in view {}'.format(subs_notvalid))
    print('Please annot them manually:')
    print(f'python3 apps/annotation/annot_calib.py {path} --mode chessboard --annot chessboard --sub {" ".join(subs_notvalid)}')

dir = 'D:\Desktop\EveryThing\WorkProject\ThreeD_demo\data11\dimian/'
dir2 = 'D:\Desktop\EveryThing\WorkProject\ThreeD_demo\data11\dimian/output'

def parser_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str,default='')
    parser.add_argument('--image', type=str, default='images')
    parser.add_argument('--out', type=str,default='')
    parser.add_argument('--ext', type=str, default='.jpg', choices=['.jpg', '.png'])
    parser.add_argument('--pattern', type=lambda x: (int(x.split(',')[0]), int(x.split(',')[1])),
                        help='The pattern of the chessboard', default=(7, 5))
    parser.add_argument('--grid', type=float, default=0.1,
                        help='The length of the grid size (unit: meter)')
    parser.add_argument('--max_step', type=int, default=60)
    parser.add_argument('--min_step', type=int, default=50)
    parser.add_argument('--mp', type=int, default=4)
    parser.add_argument('--axis', type=str, default='yx')

    parser.add_argument('--silent', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--overwrite3d', action='store_true')
    parser.add_argument('--seq', action='store_true')
    parser.add_argument('--check', action='store_true')

    return parser.parse_args()

def det_board(path, pattern, grid, seq=False):
    # args = parser_args()
    import argparse
    args_path = path
    args_out = path + 'output'
    args_pattern = pattern
    args_grid = grid
    args_seq = seq
    args_image = 'images'
    parser = argparse.ArgumentParser()
    parser.add_argument('--mp', type=int, default=4)
    parser.add_argument('--overwrite3d', action='store_true')
    parser.add_argument('--ext', type=str, default='.jpg', choices=['.jpg', '.png'])
    parser.add_argument('--axis', type=str, default='yx')
    args = parser.parse_args()

    if args_seq:
        detect_chessboard_sequence(args_path, args_out, pattern=args_pattern, gridSize=args_grid, args=args, image=args.image)
    else:
        detect_chessboard(args_path, args_image, args_out, pattern=args_pattern, gridSize=args_grid, args=args)

    if True:
        check_chessboard(args_path, args_out)



if __name__ == "__main__":

    args = parser_args()

    if args.seq:
        detect_chessboard_sequence(args.path, args.out, pattern=args.pattern, gridSize=args.grid, args=args, image=args.image)
    else:
        detect_chessboard(args.path, args.image, args.out, pattern=args.pattern, gridSize=args.grid, args=args)
    
    if args.check:
        check_chessboard(args.path, args.out)