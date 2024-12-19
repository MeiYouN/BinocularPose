'''
  @ Date: 2021-01-13 20:38:33
  @ Author: Qing Shuai
  @ LastEditors: Qing Shuai
  @ LastEditTime: 2021-04-13 21:43:52
  @ FilePath: /EasyMocapRelease/scripts/preprocess/extract_video.py
'''
import os
import cv2
from os.path import join
from tqdm import tqdm
from glob import glob

mkdir = lambda x: os.makedirs(x, exist_ok=True)

def _extract_video(videoname, path, start, end, step):
    base = os.path.basename(videoname).replace('.mp4', '')
    if not os.path.exists(videoname):
        return base
    outpath = join(path, 'images', base)
    if os.path.exists(outpath) and len(os.listdir(outpath)) > 0:
        num_images = len(os.listdir(outpath))
        print('>> exists {} frames'.format(num_images))
        return base
    else:
        os.makedirs(outpath, exist_ok=True)
    video = cv2.VideoCapture(videoname)
    totalFrames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for cnt in tqdm(range(totalFrames), desc='{:10s}'.format(os.path.basename(videoname))):
        ret, frame = video.read()
        if cnt < start:continue
        if cnt >= end:break
        if not ret:continue
        if (cnt % step ==0):
            cv2.imwrite(join(outpath, '{:06d}.jpg'.format(cnt)), frame)
    video.release()
    return base

def parser_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str, help="the path of data")
    parser.add_argument('--start', type=int, default=0,
                        help='frame start')
    parser.add_argument('--end', type=int, default=2000,
                        help='frame end')
    parser.add_argument('--step', type=int, default=2,
                        help='frame step')
    return parser.parse_args()


def extract_video(path,step):
    args = parser_args()
    args.path = path
    args.step = step

    if os.path.isdir(args.path):
        image_path = join(args.path, 'images')
        os.makedirs(image_path, exist_ok=True)
        subs_image = sorted(os.listdir(image_path))
        subs_videos = sorted(glob(join(args.path, 'videos', '*.mp4')))
        if len(subs_videos) > len(subs_image):
            videos = sorted(glob(join(args.path, 'videos', '*.mp4')))
            subs = []
            for video in videos:
                basename = _extract_video(video, args.path, start=args.start, end=args.end, step=args.step)
                subs.append(basename)
        else:
            subs = sorted(os.listdir(image_path))
        print('cameras: ', ' '.join(subs))
    else:
        print(args.path, ' not exists')


if __name__ == "__main__":

    args = parser_args()

    if os.path.isdir(args.path):
        image_path = join(args.path, 'images')
        os.makedirs(image_path, exist_ok=True)
        subs_image = sorted(os.listdir(image_path))
        subs_videos = sorted(glob(join(args.path, 'videos', '*.mp4')))
        if len(subs_videos) > len(subs_image):
            videos = sorted(glob(join(args.path, 'videos', '*.mp4')))
            subs = []
            for video in videos:
                basename = _extract_video(video, args.path, start=args.start, end=args.end, step=args.step)
                subs.append(basename)
        else:
            subs = sorted(os.listdir(image_path))
        print('cameras: ', ' '.join(subs))
    else:
        print(args.path, ' not exists')