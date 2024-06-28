from setuptools import setup


setup(
    name="pytch",
    version="0.1.2",
    description="Vocal Trainer",
    author="Pytch Contributors",
    packages=["pytch"],
    scripts=["apps/pytch"],
    install_requires=[
        "cython>=3.0.0",
        "numpy>=1.25.2",
        "scipy>=1.11.1",
        "pyqt6>=6.7.0",
        "libf0>=1.0.2",
        "sounddevice>=0.4.7",
        "matplotlib>=3.7.2",
    ],
    python_requires=">=3.12",
)
