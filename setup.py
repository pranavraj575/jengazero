from distutils.core import setup
from setuptools import find_packages

setup(
    name='jengazero',
    version='6.9.0',
    packages=find_packages(),
    install_requires=['numpy',
                      'scipy',
                      'torch',
                      ],
    license='Liscence to Krill',
)
