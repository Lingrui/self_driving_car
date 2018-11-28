# Self Driving Car

## Overview

This repository is a toolkit to take a try on semantic segmentation
task, we choose Cityscape dataset as example and simplified the traning
categories to show how it works. 

## Build the docker image
```
cd docker
make
```
It might take few minutes to build up the docker image 

To check if the docker image have successful built
```
docker image | grep aicampus
```

## Data preparation
Assume you have already downloaded the following Cityscape datasets and
unzipped them at current directory(path_to/self_driving_car/).
- gtFine_trainvaltest.zip (241MB)
- leftImg8bit_trainvaltest.zip (11GB)

All the data should be arranged in a certain folder in following
structure:
```
 |── self_driving_car
 |   ├── docker
 |   ├── gtFine
 |   ├── leftImg8bit
```
We choose some of the labels in the original dataset and merge them into
3 main categories to run the test 

The original image size of cityscape is 1024 * 2048 , we downsize them to
320 * 640 to accelrate the training process

```
./import_data.sh
```

1. Car
including car, bus, train, truck, trailer, caravan
(no trailer and caravan annation in bdd label annotation set)

2. Person·
including person, rider

3. Road·

Double check if there is a file called "class_dict.csv" in downsized
output dataset

## Model training
```
./train_demo.sh
```

## Make prediction 
```
./predict_demo.sh
```

