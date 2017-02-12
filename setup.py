import sys


import os
import time
import shutil

from distutils.core import setup
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
            'iconfile': '/Users/marius/src/pyrocko/src/data/snuffler.icns',
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

        # Manually delete files which refuse to be ignored using 'excludes':
        want_delete = glob.glob(
            'dist/snuffler.app/Contents/Frameworks/libvtk*')

        map(os.remove, want_delete)

        want_delete_dir = glob.glob(
            'dist/Snuffler.app/Contents/Resources/lib/python2.7/'
            'matplotlib/test*')
        map(shutil.rmtree, want_delete_dir)




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
    }

    #package_dir={'': 'pytch'},
    #packages=[''],
    #packages=['qtest'],
    #py_modules=['pytch'],
    #py_modules=['src/utils'],
)

