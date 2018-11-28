#!/usr/bin/env python3 
import cv2 
import os 
import numpy as np 
from glob import glob 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument ('--image',required = True, help = "Unzipped Cityscape image path" )
parser.add_argument ('--label',required = True, help = "Unzipped Cityscape color label path" )
parser.add_argument ('--out',required = True, help = "path to resized dataset" )
args = parser.parse_args()

data = ['train','val','test']
H = 320
W = 640

#define color change rules 
dic = {
        "128_64_128":[128,64,128,3], #road
        "220_20_60":[220,20,60,2], #person
        "255_0_0":[220,20,60,2],  #rider
        "0_0_142":[0,0,142,1], #car
        "0_0_70":[0,0,142,1], #truck
        "0_60_100":[0,0,142,1], #bus
        "0_0_90":[0,0,142,1], #caravan
        "0_0_110":[0,0,142,1], #trailer
        "0_80_100":[0,0,142,1], #train
    }

#load and resize image
def load_image(path):
    image = cv2.imread(path,1)
    resized_image = cv2.resize(image,(W,H),interpolation=cv2.INTER_NEAREST)
    return resized_image 

#load color label and change the color to specific ones
def load_label(path):
    label = cv2.imread(path,1)
    resized_label = cv2.resize(label,(W,H),interpolation=cv2.INTER_NEAREST)
    for i in range(0,H):
        for j in range(0,W):
            B,G,R = resized_label[i,j]
            color = str(R) + "_" + str(G) + "_" + str(B)
            if color in dic.keys():
                R_,G_,B_,ID = dic[color]
                resized_label[i,j] = [B_,G_,R_] 
            else:
                resized_label[i,j] = [0,0,0]
    return resized_label

def load_all(path,label_path,out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for cate in data:
        #creat output dir
        if not os.path.exists(os.path.join(out_path,cate)):
            os.mkdir(os.path.join(out_path,str(cate)))
            os.mkdir(os.path.join(out_path,str(cate+'_labels'))) #color labels
        #replace the color of each pixel
        for image in glob(os.path.join(path,cate,"*/*.png")):
            image_name = (os.path.basename(image)).replace("leftImg8bit","")
            gt_img = image.replace(path,label_path)
            gt_img = gt_img.replace("leftImg8bit.png","gtFine_color.png")
            if os.path.exists(gt_img):
                resized_image = load_image(image) 
                color_label = load_label(gt_img) 
                #save modified images 
                cv2.imwrite(os.path.join(out_path,str(cate),image_name),resized_image)
                cv2.imwrite(os.path.join(out_path,str(cate+'_labels'),image_name),color_label)
load_all(args.image,args.label,args.out)

f = open(os.path.join(args.out,"class_dict.csv"),"w")
f.write("name,r,g,b\n")
f.write("background,0,0,0\n")
f.write("car,0,0,142\n")
f.write("person,220,20,60\n")
f.write("rood,128,64,128\n")
f.close()
