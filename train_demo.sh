#!/bin/bash
mkdir -p ${PWD}/data
mkdir -p ${PWD}/data/cityscape_FRRN-A
mkdir -p ${PWD}/data/cityscape_FRRN-A/checkpoints
mkdir -p ${PWD}/data/cityscape_FRRN-A/Val

chmod 777 ${PWD}/data/cityscape_FRRN-A
chmod 777 ${PWD}/data/cityscape_FRRN-A/checkpoints
chmod 777 ${PWD}/data/cityscape_FRRN-A/Val

nvidia-docker run --rm -it \
	-e CUDA_VISIBLE_DEVICES=0 \
	-v ${PWD}/data/cityscape:/sssuite/cityscape \
	-v ${PWD}/data/cityscape_FRRN-A/checkpoints:/sssuite/checkpoints \
	-v ${PWD}/data/cityscape_FRRN-A/Val:/sssuite/Val \
	aaalgo/sssuite:aicampus \
	./main.py \
	--mode train \
	--dataset cityscape\
	--batch_size 1 \
	--crop_height 320 \
	--crop_width 640 \
	--num_val_images 500 \
	--model FRRN-A \
	--num_epochs 60
