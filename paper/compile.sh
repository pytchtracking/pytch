#!/bin/bash

cd ..
docker run --rm -it -v "$PWD:/data" -u "$(id -u):$(id -g)" openjournals/inara -o pdf paper/paper.md -p
