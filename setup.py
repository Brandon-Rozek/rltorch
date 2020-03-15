"""
rltorch stands for Reinforcement Learning Torch -- RL library built on top of PyTorch
"""
import setuptools


setuptools.setup(
    name="rltorch",
    author="Brandon Rozek",
    author_email="rozekbrandon@gmail.com",
    license='MIT',
    description="Reinforcement Learning Framework for PyTorch",
    version="0.1",
    packages=setuptools.find_packages(),
    install_requires=[
        "numpy~=1.16.0",
        "opencv-python~=4.2.0.32",
        "gym~=0.10.11",
        "torch~=1.4.0",
        "numba~=0.48.0"
    ]
)
