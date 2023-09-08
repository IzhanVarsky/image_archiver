#!/bin/bash

set -e

rest=$@

IMAGE=image_archiver:latest

CONTAINER_ID=$(docker inspect --format="{{.Id}}" ${IMAGE} 2> /dev/null)
if [[ "${CONTAINER_ID}" ]]; then
    docker run --runtime=nvidia --shm-size=8g --gpus all --rm \
    -v `pwd`:/scratch \
    --user $(id -u):$(id -g) --workdir=/scratch -e HOME=/scratch $IMAGE $@
else
    echo "Unknown container image: ${IMAGE}"
    exit 1
fi