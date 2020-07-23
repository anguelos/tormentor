#import setuptools
from setuptools import setup, Extension

setup(
    name='tormentor',
    version='0.1.0',
    packages=['tormentor', 'diamond_square'],
    license='MIT',
    author='Anguelos Nicolaou',
    author_email='anguelos.nicolaou@gmail.com',
    url='https://github.com/anguelos/tormentor',
    description="A very easy to use argument parser.",
    long_description_content_type="text/markdown",
    long_description=open('README.md').read(),
    download_url='https://github.com/anguelos/tormentor/archive/0.1.0.tar.gz',
    keywords=["pytorch", "augmentation", "kornia", "image segmentation", "computer vision"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering"],
    install_requires=["torch", "matplotlib", "torchvision", "fargv"],
)
