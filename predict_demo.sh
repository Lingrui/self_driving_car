#!/bin/bash
mkdir -p ${PWD}/data/cityscape_FRRN-A/Test
chmod 777 ${PWD}/data/cityscape_FRRN-A/Test

nvidia-docker run --rm -it \
	-e CUDA_VISIBLE_DEVICES=0 \
	-v ${PWD}/data/cityscape:/sssuite/cityscape \
	-v ${PWD}/data/cityscape_FRRN-A/checkpoints:/sssuite/checkpoints \
	-v ${PWD}/data/cityscape_FRRN-A/Val:/sssuite/Val \
	-v ${PWD}/data/cityscape_FRRN-A/Test:/sssuite/Test \
	aaalgo/sssuite:aicampus \
	./main_predict.py \
	--mode predict \
	--dataset cityscape\
	--crop_height 320 \
	--crop_width 640 \
	--model FRRN-A \
	--image /sssuite/cityscape/test/
