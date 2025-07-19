from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='umalloc',
    ext_modules=[
        CUDAExtension(
            name='umalloc',
            sources=['umalloc.cpp'],
        ),
    ],
    cmdclass={'build_ext': BuildExtension}
)
