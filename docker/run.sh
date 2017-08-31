#!/bin/bash

source params.sh

nvidia-docker run --rm -it ${IPC} ${NET} ${VOLUMES} ${CONTNAME} ${IMAGENAME} bash