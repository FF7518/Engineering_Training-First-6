# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from glob import glob
import os
import sys

import argparse
from data_utils import suggest_metadata

output_prefix_2d = 'data_2d_custom_'

def decode(filename):
    #由于Detectron运行在python2.7下，需要使用latin1编码模式
    print('Processing {}'.format(filename))
    data = np.load(filename, encoding='latin1', allow_pickle=True)#加载.npz文件
    bb = data['boxes']
    kp = data['keypoints']
    metadata = data['metadata'].item()
    results_bb = []
    results_kp = []
    for i in range(len(bb)):
        if len(bb[i][1]) == 0 or len(kp[i][1]) == 0:
            #未检测到此帧的bbox或者keypoints时将进行插值
            results_bb.append(np.full(4, np.nan, dtype=np.float32))#插入4个边界框坐标
            results_kp.append(np.full((17, 4), np.nan, dtype=np.float32))#插入17个COCO关节点
            continue
        best_match = np.argmax(bb[i][1][:, 4])
        best_bb = bb[i][1][best_match, :4]
        best_kp = kp[i][1][best_match].T.copy()
        results_bb.append(best_bb)
        results_kp.append(best_kp)
        
    bb = np.array(results_bb, dtype=np.float32)
    kp = np.array(results_kp, dtype=np.float32)
    kp = kp[:, :, :2]#提取出(x,y)坐标
    
    #通过线性插值修复缺失的bboxes/keypoints
    mask = ~np.isnan(bb[:, 0])
    indices = np.arange(len(bb))
    for i in range(4):
        bb[:, i] = np.interp(indices, indices[mask], bb[mask, i])
    for i in range(17):
        for j in range(2):
            kp[:, i, j] = np.interp(indices, indices[mask], kp[mask, i, j])
    
    print('{} total frames processed'.format(len(bb)))
    print('{} frames were interpolated'.format(np.sum(~mask)))
    print('----------')
    
    return [{
        'start_frame': 0,#包含0
        'end_frame': len(kp),#唯一值
        'bounding_boxes': bb,
        'keypoints': kp,
    }], metadata

#仅作为脚本运行
if __name__ == '__main__':
    if os.path.basename(os.getcwd()) != 'data':
        print('This script must be launched from the "data" directory')
        exit(0)
    #根据函数内容创建对象（参数配置说明）
    parser = argparse.ArgumentParser(description='Custom dataset creator')
    parser.add_argument('-i', '--input', type=str, default='', metavar='PATH', help='detections directory')
    parser.add_argument('-o', '--output', type=str, default='', metavar='PATH', help='output suffix for 2D detections')
    args = parser.parse_args()
    
    if not args.input:
        print('Please specify the input directory')
        exit(0)
        
    if not args.output:
        print('Please specify an output suffix (e.g. detectron_pt_coco)')
        exit(0)
    
    print('Parsing 2D detections from', args.input)
    
    metadata = suggest_metadata('coco')
    metadata['video_metadata'] = {}
    #数据格式转换
    output = {}
    file_list = glob(args.input + '/*.npz')
    for f in file_list:
        canonical_name = os.path.splitext(os.path.basename(f))[0]
        data, video_metadata = decode(f)
        output[canonical_name] = {}
        output[canonical_name]['custom'] = [data[0]['keypoints'].astype('float32')]
        metadata['video_metadata'][canonical_name] = video_metadata

    print('Saving...')
    np.savez_compressed(output_prefix_2d + args.output, positions_2d=output, metadata=metadata)
    print('Done.')