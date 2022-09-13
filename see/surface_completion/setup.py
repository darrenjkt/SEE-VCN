from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# pip install -e . --user
setup(name='seev2',
      version='1.0',
      description='Complete the surface of incomplete lidar point clouds',      
      author='Darren Tsai',
      author_email='d.tsai@acfr.usyd.edu.au',
      license='MIT',
      packages=find_packages(exclude=['scripts','experiments']),
      cmdclass={'build_ext': BuildExtension},
      install_requires=[  
        'argparse',
        'easydict',
        'h5py',
        'matplotlib',
        'numpy',    
        'opencv-python==4.5.3.56',
        'pyyaml',
        'scipy',
        'tensorboardX',
        'timm==0.4.5 ',
        'tqdm==4.51.0',
        'open3d==0.14.1', 
        'transforms3d',
      ],
      ext_modules=[
          CUDAExtension(
            name='chamfer', 
            sources=[
              'models/partialsc/extensions/chamfer_dist/chamfer_cuda.cpp',
              'models/partialsc/extensions/chamfer_dist/chamfer.cu',
          ]),
          CUDAExtension(
            name='iou3d_nms_cuda', 
            sources=[
              'models/partialsc/extensions/iou3d_nms/src/iou3d_cpu.cpp',
              'models/partialsc/extensions/iou3d_nms/src/iou3d_nms_api.cpp',
              'models/partialsc/extensions/iou3d_nms/src/iou3d_nms.cpp',
              'models/partialsc/extensions/iou3d_nms/src/iou3d_nms_kernel.cu',
          ]),
      ])