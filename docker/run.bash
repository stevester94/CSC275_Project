#! /bin/bash
docker run --network=host -ti --gpus all --rm -v /mnt/lebensraum/:/mnt/lebensraum/ csc275-tf-image
