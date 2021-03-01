import sys

import os
import time
import shutil

from distutils.core import setup, Extension
from distutils.command.build_ext import build_ext


class custom_build_app(build_ext):
    def run(self):
        self.make_app()

    def make_app(self):
        import glob
        import os
        import shutil
        from setuptools import setup

        APP = ["apps/pytch"]
        DATA_FILES = []
        OPTIONS = {
            "argv_emulation": True,
            "packages": "pytch",
        }

        setup(
            app=APP,
            data_files=DATA_FILES,
            options={"py2app": OPTIONS},
            setup_requires=["py2app"],
        )


setup(
    name="pytch",
    version="0.1",
    description="Vocal Trainer",
    author="Frank Scherbaum and Marius Kriegerowski",
    package_dir={"pytch": "src"},
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
    cmdclass={
        "py2app": custom_build_app,
    },
    python_requires=">=3.6",
)
