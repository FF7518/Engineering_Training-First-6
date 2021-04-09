# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""Perform inference on a single video or all videos with a certain extension
(e.g., .mp4) in a folder.
"""

import detectron2
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

import subprocess as sp
import numpy as np
import time
import argparse
import sys
import os
import glob

def parse_args():#根据函数内容创建对象（参数配置说明）
    parser = argparse.ArgumentParser(description='End-to-end inference')#在参数文档之前显示End-to-end inference
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: mp4)',
        default='mp4',
        type=str
    )
    parser.add_argument(
        'im_or_folder',
        help='image or folder of images',
        default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def get_resolution(filename):#获取文件宽度和高度
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    for line in pipe.stdout:
        w, h = line.decode().strip().split(',')
        return int(w), int(h)

def read_video(filename):#读取文件
    w, h = get_resolution(filename)

    command = ['ffmpeg',
            '-i', filename,
            '-f', 'image2pipe',
            '-pix_fmt', 'bgr24',
            '-vsync', '0',
            '-vcodec', 'rawvideo', '-']

    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=-1)
    while True:
        data = pipe.stdout.read(w*h*3)
        if not data:
            break
        yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))#数据重塑


def main(args):

    cfg = get_cfg()#获取detectron2的默认配置
    cfg.merge_from_file(model_zoo.get_config_file(args.cfg))#从文件加载值
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(args.cfg)#加载检测点
    predictor = DefaultPredictor(cfg)#创建一个端到端检测器
    
    #路径操作
    if os.path.isdir(args.im_or_folder):#读取路径
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    for video_name in im_list:
        out_name = os.path.join(args.output_dir, os.path.basename(video_name))#输出路径
        print('Processing {}'.format(video_name))

        boxes = []
        segments = []
        keypoints = []
        #提取每一帧的骨骼
        for frame_i, im in enumerate(read_video(video_name)):#frame_i为下标，im为数据
            t = time.time()
            outputs = predictor(im)['instances'].to('cpu')
            
            print('Frame {} processed in {:.3f}s'.format(frame_i, time.time() - t))

            has_bbox = False
            if outputs.has('pred_boxes'):
                bbox_tensor = outputs.pred_boxes.tensor.numpy()
                if len(bbox_tensor) > 0:
                    has_bbox = True
                    scores = outputs.scores.numpy()[:, None]
                    bbox_tensor = np.concatenate((bbox_tensor, scores), axis=1)
            if has_bbox:#数据格式化
                kps = outputs.pred_keypoints.numpy()
                kps_xy = kps[:, :, :2]
                kps_prob = kps[:, :, 2:3]
                kps_logit = np.zeros_like(kps_prob) # Dummy
                kps = np.concatenate((kps_xy, kps_logit, kps_prob), axis=2)
                kps = kps.transpose(0, 2, 1)
            else:
                kps = []
                bbox_tensor = []
                
            #模拟成detectron1的格式
            cls_boxes = [[], bbox_tensor]
            cls_keyps = [[], kps]
            
            boxes.append(cls_boxes)
            segments.append(None)
            keypoints.append(cls_keyps)

        
        #输出视频的分辨率
        metadata = {
            'w': im.shape[1],
            'h': im.shape[0],
        }
        
        #保存为.npz文件
        np.savez_compressed(out_name, boxes=boxes, segments=segments, keypoints=keypoints, metadata=metadata)

#当前模块仅作为脚本执行
if __name__ == '__main__':
    setup_logger()
    args = parse_args()
    main(args)
