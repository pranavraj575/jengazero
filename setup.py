from distutils.core import setup
from setuptools import find_packages

setup(
    name='jengazebo',
    version='6.9.0',
    packages=find_packages(),
    install_requires=['numpy',
                      'scipy',
                      ],
    license='Liscence to Krill',
)
