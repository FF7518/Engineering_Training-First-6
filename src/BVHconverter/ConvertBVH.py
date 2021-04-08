# Convert 3D pose to BVH
# Version 1.0
# 将3D姿态估计得到的坐标点信息转换为bvh骨骼文件
# input: *.npy (npz是多个npy的集合，也可以加载进来，但需要分开处理)
# input: shape = (frame_nums, joints=17, dimensions=3)
# output: *.bvh
from bvh_skeleton import h36m_skeleton
import numpy as np

def generate_bvh(inputfile, outputfile):
    h36m_skel = h36m_skeleton.H36mSkeleton()
    data = np.load(inputfile)
    _ = h36m_skel.poses2bvh(data, output_file=outputfile)
    
generate_bvh('./output_mpc.npy', './3d.bvh')