#!/bin/bash

mkdir -p ${PWD}/cityscape
chmod 777 ${PWD}/cityscape

nvidia-docker run --rm -it \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v ${PWD}/cityscape:/sssuite/cityscape \
    -v ${PWD}/gtFine:/sssuite/gtFine \
    -v ${PWD}/leftImg8bit:/sssuite/leftImg8bit \
    aaalgo/sssuite:aicampus \
    ./import_cityscape_resize.py \
    --image ./leftImg8bit \
    --label ./gtFine \
    --out ./cityscape

