import sys


import os
import time
import shutil

from distutils.core import setup


setup(
    name='pytch',
    version='0.1',
    description='Vocal Trainer',
    author='Frank Scherbaum and Marius Kriegerowski',
    package_dir={'pytch': 'src'},
    packages=['pytch'],
    scripts=['apps/pytch'],
    #package_dir={'': 'pytch'},
    #packages=[''],
    #packages=['qtest'],
    #py_modules=['pytch'],
    #py_modules=['src/utils'],
)

