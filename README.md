Pytch - A Real-Time Pitch Analysis Tool For Polyphonic Music
===============

[![Tests](https://github.com/pytchtracking/pytch/actions/workflows/python-app.yml/badge.svg)](https://github.com/pytchtracking/pytch/actions/workflows/python-app.yml)

![screenshot](pictures/screenshot.png)

## Download and Installation

Clone the project
```
git clone https://github.com/pytchtracking/pytch
cd pytch
```

Install pytch:
```
pip3 install .
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

## Contribution

Every contribution is welcome. To ensure consistent style we use [black](https://github.com/psf/black).
You can add automated style checks at commit time using [pre-commit](https://pre-commit.com/)

```bash
pip3 install pre-commit
pre-commit install
```
