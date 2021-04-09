# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import matplotlib
#控制绘图不显示
matplotlib.use('Agg')

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import subprocess as sp
#获取分辨率
def get_resolution(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    #将command指令，以管道PIPE作为标准输出，用默认系统缓冲，开启另一个进程
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            #对于每一行输出删除开头结尾的空白字符，以','分割，并赋给w和h
            w, h = line.decode().strip().split(',')
            #返回分辨率
            return int(w), int(h)
#获取每秒传输帧数
def get_fps(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', filename]
    # 将command指令，以管道PIPE作为标准输出，用默认系统缓冲，开启另一个进程
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            # 对于每一行输出删除开头结尾的空白字符，以'/'分割，并赋给a和b
            a, b = line.decode().strip().split('/')
            #返回每秒传输帧数
            return int(a) / int(b)
#读视频
def read_video(filename, skip=0, limit=-1):
    #获取视频分辨率
    w, h = get_resolution(filename)
    
    command = ['ffmpeg',
            '-i', filename,
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vsync', '0',
            '-vcodec', 'rawvideo', '-']
    
    i = 0
    # 将command指令，以管道PIPE作为标准输出，用默认系统缓冲，开启另一个进程
    with sp.Popen(command, stdout = sp.PIPE, bufsize=-1) as pipe:
        while True:
            #读取视频信息
            data = pipe.stdout.read(w*h*3)
            #如果没有信息则终止
            if not data:
                break
            #否则计算加1
            i += 1
            #只要没超过限制就继续
            if i > limit and limit != -1:
                continue
            if i > skip:
                #生成3d坐标的动态数组
                yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))
            
                
                
#向下取样
def downsample_tensor(X, factor):
    #对X的shape数据向下取整采样
    length = X.shape[0]//factor * factor
    #压缩列，对各行求均值，返回 m *1 矩阵
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)
#渲染动画
def render_animation(keypoints, keypoints_metadata, poses, skeleton, fps, bitrate, azim, output, viewport,
                     limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0):
    """
    待办事项

渲染动画。支持的输出模式有：

--“交互式”：显示交互式图形

（如果与%matplotlib inline关联，也适用于笔记本电脑）

--“html”：将动画渲染为HTML5视频。可以使用HTML（…）显示在笔记本中。

-- '文件名.mp4'：将动画渲染并导出为h264视频（需要ffmpeg）。

-- '文件名.gif'：将动画渲染并导出为gif文件（需要imagemagick）。
    """
    #关掉交互模式
    plt.ioff()
    #创建自定义图像,设置图像大小
    fig = plt.figure(figsize=(size*(1 + len(poses)), size))
    #设置画布位置及大小
    ax_in = fig.add_subplot(1, 1 + len(poses), 1)
    #定义x轴不可见
    ax_in.get_xaxis().set_visible(False)
    #定义y轴不可见
    ax_in.get_yaxis().set_visible(False)
    #设置辅助显示层不显示
    ax_in.set_axis_off()
    #给ax_in这个图设置标题
    ax_in.set_title('Input')

    ax_3d = []
    lines_3d = []
    trajectories = []
    radius = 1.7
    #enumerate将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    for index, (title, data) in enumerate(poses.items()):
        #设置画布位置及大小
        ax = fig.add_subplot(1, 1 + len(poses), index+2, projection='3d')
        #转换视角进行观察
        ax.view_init(elev=15., azim=azim)
        #在mplot3d中缩放x轴
        ax.set_xlim3d([-radius/2, radius/2])
        #在mplot3d中缩放z轴
        ax.set_zlim3d([0, radius])
        #在mplot3d中缩放y轴
        ax.set_ylim3d([-radius/2, radius/2])
#        ax.set_aspect('equal')删除白色边框
        #设置x轴刻度
        ax.set_xticklabels([])
        #设置y轴刻度
        ax.set_yticklabels([])
        #设置z轴刻度
        ax.set_zticklabels([])
        ax.dist = 7.5
        #给ax这个图设置标题
        ax.set_title(title) #, pad=35
        #在列表末尾添加新的对象
        ax_3d.append(ax)
        #在列表末尾添加新的对象
        lines_3d.append([])
        #在列表末尾添加新的对象
        trajectories.append(data[:, 0, [0, 1]])
    #将元组转换为列表
    poses = list(poses.values())

    # 解码视频
    if input_video_path is None:
        # 黑色背景
        all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
    else:
        # 使用ffmpeg加载视频
        all_frames = []
        #获取3D坐标数组
        for f in read_video(input_video_path, skip=input_video_skip, limit=limit):
            # 在列表末尾添加新的对象f
            all_frames.append(f)
        effective_length = min(keypoints.shape[0], len(all_frames))
        all_frames = all_frames[:effective_length]
        
        keypoints = keypoints[input_video_skip:] # todo remove
        for idx in range(len(poses)):
            poses[idx] = poses[idx][input_video_skip:]
        
        if fps is None:
            fps = get_fps(input_video_path)
    
    if downsample > 1:
        keypoints = downsample_tensor(keypoints, downsample)
        all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
        for idx in range(len(poses)):
            poses[idx] = downsample_tensor(poses[idx], downsample)
            trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
        fps /= downsample

    initialized = False
    image = None
    lines = []
    points = None
    
    if limit < 1:
        limit = len(all_frames)
    else:
        limit = min(limit, len(all_frames))

    parents = skeleton.parents()
    def update_video(i):
        nonlocal initialized, image, lines, points

        for n, ax in enumerate(ax_3d):
            ax.set_xlim3d([-radius/2 + trajectories[n][i, 0], radius/2 + trajectories[n][i, 0]])
            ax.set_ylim3d([-radius/2 + trajectories[n][i, 1], radius/2 + trajectories[n][i, 1]])

        # 更新2D姿势
        joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
        colors_2d = np.full(keypoints.shape[1], 'black')
        colors_2d[joints_right_2d] = 'red'
        if not initialized:
            image = ax_in.imshow(all_frames[i], aspect='equal')
            
            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                    
                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    # 仅当关键点匹配时绘制骨架（否则我们没有父定义）
                    lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                            [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))

                col = 'red' if j in skeleton.joints_right() else 'black'
                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
                                               [pos[j, 1], pos[j_parent, 1]],
                                               [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))

            points = ax_in.scatter(*keypoints[i].T, 10, color=colors_2d, edgecolors='white', zorder=10)

            initialized = True
        else:
            image.set_data(all_frames[i])

            for j, j_parent in enumerate(parents):
                if j_parent == -1:
                    continue
                
                if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
                    lines[j-1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
                                           [keypoints[i, j, 1], keypoints[i, j_parent, 1]])

                for n, ax in enumerate(ax_3d):
                    pos = poses[n][i]
                    lines_3d[n][j-1][0].set_xdata([pos[j, 0], pos[j_parent, 0]])
                    lines_3d[n][j-1][0].set_ydata([pos[j, 1], pos[j_parent, 1]])
                    lines_3d[n][j-1][0].set_3d_properties([pos[j, 2], pos[j_parent, 2]], zdir='z')

            points.set_offsets(keypoints[i])
        
        print('{}/{}      '.format(i, limit), end='\r')
        

    fig.tight_layout()
    
    anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000/fps, repeat=False)
    if output.endswith('.mp4'):
        Writer = writers['ffmpeg']
        writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
        anim.save(output, writer=writer)
    elif output.endswith('.gif'):
        anim.save(output, dpi=80, writer='imagemagick')
    else:
        raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
    plt.close()
