from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='um_tensor',
    ext_modules=[
        CUDAExtension(
            name='um_tensor',
            sources=['um_tensor.cpp'],
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)