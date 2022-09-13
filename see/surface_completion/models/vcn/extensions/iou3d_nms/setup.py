# -*- coding: utf-8 -*-
# @Author: Haozhe Xie
# @Date:   2019-08-07 20:54:24
# @Last Modified by:   Haozhe Xie
# @Last Modified time: 2019-12-10 10:04:25
# @Email:  cshzxie@gmail.com

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(name='iou3d_nms_cuda',
      version='1.0.0',
      ext_modules=[
          CUDAExtension('iou3d_nms_cuda', [
              'src/iou3d_cpu.cpp',
              'src/iou3d_nms_api.cpp',
              'src/iou3d_nms.cpp',
              'src/iou3d_nms_kernel.cu',
          ]),
      ],
      cmdclass={'build_ext': BuildExtension})
