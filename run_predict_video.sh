#!/bin/bash
mkdir -p ${PWD}/data/cityscape_FRRN-A/Test
chmod 777 ${PWD}/data/cityscape_FRRN-A/Test

nvidia-docker run --rm -it \
    -e CUDA_VISIBLE_DEVICES=0 \
    -v ${PWD}/data/cityscape_FRRN-A/checkpoints:/sssuite/checkpoints \
    -v ${PWD}/data/cityscape_FRRN-A/Test:/sssuite/Test \
    -v ${PWD}/data/cityscape:/sssuite/cityscape \
    -v ${PWD}/data/cityscape/videos-sample.mov:/sssuite/cityscape/videos-sample.mov \
    -v ${PWD}/predict_video.py:/sssuite/main.py \
    aaalgo/sssuite:aicampus \
    ./main.py \
    --mode predict \
    --dataset cityscape \
    --fps 30 \
    --model FRRN-A \
    --video /sssuite/cityscape/videos-sample.mov
