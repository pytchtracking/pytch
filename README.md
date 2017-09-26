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

Optional but recommended is the installation of PyQt5-OpenGL bindings.
Help can be found here: http://pyqt.sourceforge.net/Docs/PyQt5/installation.html

```
sudo apt-get install python3-dev portaudio19-dev
pip install numpy PyQt5 pyaudio git+https://git.aubio.org/aubio/aubio
```

Note: if your sysmte's default python version is not 2, replace `pip` with
`pip3`.

# Download and Installation
Go to a directory where you keep your source codes and clone the project:
```
git clone https://github.com/HerrMuellerluedenscheid/pytch.git
cd pytch
sudo python setup.py install
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
