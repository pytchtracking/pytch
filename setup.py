import sys

import os
import time
import shutil

from setuptools import setup


setup(
    name="pytch",
    version="0.1.1",
    description="Vocal Trainer",
    author="Pytch Contributors",
    packages=["pytch"],
    scripts=["apps/pytch"],
    install_requires=[
        "cython>=0.29",
        "numpy>=1.15.4",
        "scipy>=1.1.0",
        "PyQt5",
        "aubio>=0.4.7",
        "pyaudio",
    ],
    python_requires=">=3.6",
)
