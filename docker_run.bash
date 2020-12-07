#! /bin/bash
docker run --network=host -ti --gpus all --rm -v /mnt/lebensraum/:/kek tensorflow/tensorflow:latest-gpu
