#import setuptools
from setuptools import setup, Extension
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path('tormentor/version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name='tormentor',
    version=main_ns['__version__'],
    packages=['tormentor', 'diamond_square'],
    license='MIT',
    author='Anguelos Nicolaou',
    author_email='anguelos.nicolaou@gmail.com',
    url='https://github.com/anguelos/tormentor',
    description="Image Data Augmentation with Pytorch and Kornia",
    long_description_content_type="text/markdown",
    long_description=open('README.rst').read(),
    download_url='https://github.com/anguelos/tormentor/archive/0.1.0.tar.gz',
    keywords=["pytorch", "augmentation", "kornia", "image segmentation", "computer vision"],
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering"],
    install_requires=["torch", "matplotlib", "torchvision", "fargv", "kornia>=0.3.2"],
)
