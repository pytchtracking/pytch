Pytch Vocal Trainer
===============

## Prerequisites

Pytch requires python3, as well as the following libraries:

- scipy
- numpy
- PyQt5
- aubio
- pyaudio
- portaudio
- matplotlib

Optional but recommended is the installation of PyQt5-OpenGL bindings.
Help can be found here: http://pyqt.sourceforge.net/Docs/PyQt5/installation.html

## Download and Installation

Download the project
```
git clone https://github.com/pytchtracking/pytch
cd pytch
```

### Linux
```
sudo apt-get install portaudio19-dev libportaudio2
python setup.py install
```

### Apple Silicon (M1/M2):
Install portaudio, aubio, and PyQt5 via brew:
```
brew install portaudio aubio pyqt5
```

If you have conda installed, make sure that it is deactivated whenever you run pytch:
```
while [ ! -z $CONDA_PREFIX ]; do conda deactivate; done
```

Install pytch:
```
python3 setup.py install
```

For development purposes, install pytch with:
```
pip3 install -e .
```

# Invocation
Open a terminal, type
```
pytch
```
hit return and sing!

# Contribution

Every contribution is welcome. To ensure consistent style we use [black](https://github.com/psf/black).
You can add automated style checks at commit time using [pre-commit](https://pre-commit.com/)

```bash
pip3 install pre-commit
pre-commit install
```
