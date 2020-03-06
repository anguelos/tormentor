from setuptools import setup, find_packages

requirements = [
    'torch>=1.0.0',
    'kornia>=0.2.0',
]


if __name__ == '__main__':
    setup(
        # Metadata
        name="tormentor",
        version="0.1.0",
        author='Anguelos Nicolaou',
        author_email='anguelos.nicolaou@gmail.com',
        url='https://github.com/anguelos/tormentor',
        description='Augmentation Framework for PyTorch',
        long_description=open('README.md').read(),
        license='Apache License 2.0',
        python_requires='>=3.6',

        # Test
        setup_requires=['pytest-runner'],
        tests_require=['pytest'],

        # Package info
        packages=find_packages(exclude=('test', 'examples',)),

        zip_safe=True,
        install_requires=requirements,
            classifiers=[
                'Intended Audience :: Developers',
                'Intended Audience :: Education',
                'Intended Audience :: Science/Research',
                'Operating System :: POSIX :: Linux',
                'Programming Language :: Python :: 3 :: Only',
                'License :: OSI Approved :: Apache Software License',
                'Topic :: Scientific/Engineering :: Image Recognition',
                'Topic :: Software Development :: Libraries',
             ],
    )
