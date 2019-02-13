# Vocal Trainer
[![Build Status](https://travis-ci.org/HerrMuellerluedenscheid/pytch.svg?branch=master)](https://travis-ci.org/HerrMuellerluedenscheid/pytch)

# Prerequisites

Pytch requires python3, as well as the following libraries:

- python header
- scipy
- numpy
- PyQt5
- aubio
- pyaudio
- portaudio
- libportaudio2

Optional but recommended is the installation of PyQt5-OpenGL bindings.
Help can be found here: http://pyqt.sourceforge.net/Docs/PyQt5/installation.html

```
sudo apt-get install python3-dev portaudio19-dev libportaudio2
pip install numpy PyQt5 pyaudio git+https://git.aubio.org/aubio/aubio
```

Note: `portaudio` seems to have a bug at least in the current version that is
being shipped with debian stretch. If you find the list of input devices empty,
it might help to install the latest version fresh off their github repo:

```
git clone https://git.assembla.com/portaudio.git
cd portaudio
./configure
make install
```

# Download and Installation
Go to a directory where you keep your source codes and clone the project:
```
git clone https://github.com/HerrMuellerluedenscheid/pytch.git
cd pytch
sudo python setup.py install
```

Installation via Miniconda/Anaconda:
```
conda env create -f environment.yml
source activate pytch
```

# Invocation
Open a terminal, type
```
pytch
```
hit return and sing!

## Todo
- cross spectra
- add interaction to spectra (zoom)
- add midi channels (as guide)
