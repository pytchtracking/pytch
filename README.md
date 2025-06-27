Pytch - A Real-Time Pitch Analysis Tool For Polyphonic Music
===============

[![Tests](https://github.com/pytchtracking/pytch/actions/workflows/python-app.yml/badge.svg)](https://github.com/pytchtracking/pytch/actions/workflows/python-app.yml)

![screenshot](pictures/screenshot.png)

## Demo & Wiki

If you want to see `pytch` in action, watch our [demo video](https://youtu.be/LPt83Wqf2e4).

Please have a look at our [wiki](https://github.com/pytchtracking/pytch/wiki) for an explanation of the GUI.


## Download and Installation

Clone the project
```
git clone https://github.com/pytchtracking/pytch
cd pytch
```

Install pytch:
```
pip install .
```

On Linux, make sure the portaudio library is installed:
```
sudo apt install libportaudio2
```

## Run
Open a terminal, type
```
pytch
```
hit return and sing!

## Contributing

Every contribution is welcome. Please feel free to open and issue or a pull request. To ensure consistent style we use [black](https://github.com/psf/black).
You can add automated style checks at commit time using [pre-commit](https://pre-commit.com/)

```bash
pip install pre-commit
pre-commit install
```
