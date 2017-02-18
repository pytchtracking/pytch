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
pip3 install pyqt5, pyaudio, git+git://git.aubio.org/git/aubio
```

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
- differential -> scale +- 1500 or abs() < +1500
- differential grid (fix) 100 cent.
- Gauge: keep last value (at least some seconds)
- cross spectrum
- add interaction to spectra (zoom)
- colorize channels
- add midi channels (as guide)
