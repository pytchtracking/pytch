# Vocal Trainer
[![Build Status](https://travis-ci.org/HerrMuellerluedenscheid/pytch.svg?branch=master)](https://travis-ci.org/HerrMuellerluedenscheid/pytch)

# Prerequisites

Both, python2 and python3 are supported

- numpy
- PyQt5
- aubio
- pyaudio


or
```
qt5-default
```
follow these instructions: http://pyqt.sourceforge.net/Docs/PyQt5/installation.html
download sip, python configure.py, make, make install
download pyqt5 src, python configure.py, make, make install


```
sudo apt-get install portaudio19-dev
```
Install python bindings
```
pip install numpy pyqt5 pyaudio git+https://git.aubio.org/aubio/aubio
```

Note: if your sysmte's default python version is not 2, replace `pip` with
`pip3`.

# Download and Installation
Go to a directory where you keep your source codes and clone the project:
```
git clone https://github.com/HerrMuellerluedenscheid/pytch.git
```
cd into that directory and run
```
sudo python setup.py install
```

# Invocation
Open a terminal, type
```
pytch
```
hit return and sing!

## Todo
- drag drop spektra um der hoehe nach zu sortieren .
- cross spectrum
- add interaction to spectra (zoom)
- add midi channels (as guide)
