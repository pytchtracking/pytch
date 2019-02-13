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

        APP = ['apps/pytch']
        DATA_FILES = []
        OPTIONS = {
            'argv_emulation': True,
            'iconfile': '/home/marius/src/pyrocko/src/data/snuffler.icns',
            'packages': 'pytch',
            'excludes': [
                # 'PyQt4.QtDesigner',
        ]
        }

        setup(
            app=APP,
            data_files=DATA_FILES,
            options={'py2app': OPTIONS},
            setup_requires=['py2app'],
        )


setup(
    name='pytch',
    version='0.1',
    description='Vocal Trainer',
    author='Frank Scherbaum and Marius Kriegerowski',
    package_dir={'pytch': 'src'},
    packages=['pytch'],
    scripts=['apps/pytch'],
    cmdclass={
        'py2app': custom_build_app,
    },
    # ext_modules=[
    #     Extension('midi_ext',
    #               extra_compile_args=['-Wextra'],
    #               sources=[os.path.join('src', 'midi_ext.c')]),
    # ]

)

