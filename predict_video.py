#!/usr/bin/env python3
#from __future__ import print_function
import matplotlib
matplotlib.use('Agg') 
import os,time,cv2, sys, math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import time, datetime
import argparse
import random
import os, sys
import subprocess
from glob import glob
import helpers 
import utils 

import matplotlib.pyplot as plt
sys.path.append("models")
from FC_DenseNet_Tiramisu import build_fc_densenet
from Encoder_Decoder import build_encoder_decoder
from RefineNet import build_refinenet
from FRRN import build_frrn
from MobileUNet import build_mobile_unet
from PSPNet import build_pspnet
from GCN import build_gcn
from DeepLabV3 import build_deeplabv3
from DeepLabV3_plus import build_deeplabv3_plus
from AdapNet import build_adaptnet

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--num_epochs', type=int, default=300, help='Number of epochs to train for')
parser.add_argument('--mode', type=str, default="train", help='Select "train", "test", or "predict" mode. \
    Note that for prediction mode you have to specify an image to run the model on.')
parser.add_argument('--checkpoint_step', type=int, default=10, help='How often to save checkpoints (epochs)')
parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
parser.add_argument('--class_balancing', type=str2bool, default=False, help='Whether to use median frequency class weights to balance the classes in the loss')
parser.add_argument('--loss_func', type=str, default="cross_entropy", help='Which loss function to use (cross_entropy or lovasz)')
parser.add_argument('--video', type=str, default=None, help='The video you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--fps', type=int, default=None, help='The framerate of video you want to predict on. Only valid in "predict" mode.')
parser.add_argument('--continue_training', type=str2bool, default=False, help='Whether to continue training from a checkpoint')
parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
parser.add_argument('--num_val_images', type=int, default=10, help='The number of images to used for validations')
parser.add_argument('--h_flip', type=str2bool, default=False, help='Whether to randomly flip the image horizontally for data augmentation')
parser.add_argument('--v_flip', type=str2bool, default=False, help='Whether to randomly flip the image vertically for data augmentation')
parser.add_argument('--brightness', type=float, default=None, help='Whether to randomly change the image brightness for data augmentation. Specifies the max bightness change.')
parser.add_argument('--rotation', type=float, default=None, help='Whether to randomly rotate the image for data augmentation. Specifies the max rotation angle.')
parser.add_argument('--model', type=str, default="FC-DenseNet56", help='The model you are using. Currently supports:\
    FC-DenseNet56, FC-DenseNet67, FC-DenseNet103, Encoder-Decoder, Encoder-Decoder-Skip, RefineNet-Res50, RefineNet-Res101, RefineNet-Res152, \
    FRRN-A, FRRN-B, MobileFCN, MobileUNet, MobileUNet-Add, PSPNet-Res50, PSPNet-Res101, PSPNet-Res152, GCN-Res50, GCN-Res101, GCN-Res152, DeepLabV3-Res50 \
    DeepLabV3-Res101, DeepLabV3-Res152, DeepLabV3_plus-Res50, DeepLabV3_plus-Res101, DeepLabV3_plus-Res152, AdapNet, custom')
args = parser.parse_args()

# Get a list of the training, validation, and testing file paths
def prepare_data(dataset_dir=args.dataset):
    train_input_names=[]
    train_output_names=[]
    val_input_names=[]
    val_output_names=[]
    test_input_names=[]
    test_output_names=[]
    for file in os.listdir(dataset_dir + "/train"):
        cwd = os.getcwd()
        train_input_names.append(cwd + "/" + dataset_dir + "/train/" + file)
    for file in os.listdir(dataset_dir + "/train_labels"):
        cwd = os.getcwd()
        train_output_names.append(cwd + "/" + dataset_dir + "/train_labels/" + file)
    for file in os.listdir(dataset_dir + "/val"):
        cwd = os.getcwd()
        val_input_names.append(cwd + "/" + dataset_dir + "/val/" + file)
    for file in os.listdir(dataset_dir + "/val_labels"):
        cwd = os.getcwd()
        val_output_names.append(cwd + "/" + dataset_dir + "/val_labels/" + file)
    for file in os.listdir(dataset_dir + "/test"):
        cwd = os.getcwd()
        test_input_names.append(cwd + "/" + dataset_dir + "/test/" + file)
    for file in os.listdir(dataset_dir + "/test_labels"):
        cwd = os.getcwd()
        test_output_names.append(cwd + "/" + dataset_dir + "/test_labels/" + file)
    train_input_names.sort(),train_output_names.sort(), val_input_names.sort(), val_output_names.sort(), test_input_names.sort(), test_output_names.sort()
    return train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names


def load_image(path):
    image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return image

def data_augmentation(input_image, output_image):
    # Data augmentation
    input_image, output_image = utils.random_crop(input_image, output_image, args.crop_height, args.crop_width)

    if args.h_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 1)
        output_image = cv2.flip(output_image, 1)
    if args.v_flip and random.randint(0,1):
        input_image = cv2.flip(input_image, 0)
        output_image = cv2.flip(output_image, 0)
    if args.brightness:
        factor = random.uniform(-1*args.brightness, args.brightness)
        table = np.array([((i / 255.0) ** factor) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        input_image = cv2.LUT(input_image, table)
    if args.rotation:
        angle = random.uniform(-1*args.rotation, args.rotation)
    if args.rotation:
        M = cv2.getRotationMatrix2D((input_image.shape[1]//2, input_image.shape[0]//2), angle, 1.0)
        input_image = cv2.warpAffine(input_image, M, (input_image.shape[1], input_image.shape[0]), flags=INTER_NEAREST)
        output_image = cv2.warpAffine(output_image, M, (output_image.shape[1], output_image.shape[0]), flags=INTER_NEAREST)

    return input_image, output_image

def download_checkpoints(model_name):
    subprocess.check_output(["python", "get_pretrained_checkpoints.py", "--model=" + model_name])

# Get the names of the classes so we can record the evaluation results
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

# Get the selected model. 
# Some of them require pre-trained ResNet

if "Res50" in args.model and not os.path.isfile("models/resnet_v2_50.ckpt"):
    download_checkpoints("Res50")
if "Res101" in args.model and not os.path.isfile("models/resnet_v2_101.ckpt"):
    download_checkpoints("Res101")
if "Res152" in args.model and not os.path.isfile("models/resnet_v2_152.ckpt"):
    download_checkpoints("Res152")

# Compute your softmax cross entropy loss
print("Preparing the model ...")
net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 

network = None
init_fn = None
if args.model == "FC-DenseNet56" or args.model == "FC-DenseNet67" or args.model == "FC-DenseNet103":
    network = build_fc_densenet(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "RefineNet-Res50" or args.model == "RefineNet-Res101" or args.model == "RefineNet-Res152":
    # RefineNet requires pre-trained ResNet weights
    network, init_fn = build_refinenet(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "FRRN-A" or args.model == "FRRN-B":
    network = build_frrn(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "Encoder-Decoder" or args.model == "Encoder-Decoder-Skip":
    network = build_encoder_decoder(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "MobileFCN" or args.model == "MobileUNet" or args.model == "MobileUNet-Add":
    network = build_mobile_unet(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "PSPNet-Res50" or args.model == "PSPNet-Res101" or args.model == "PSPNet-Res152":
    # Image size is required for PSPNet
    # PSPNet requires pre-trained ResNet weights
    network, init_fn = build_pspnet(net_input, label_size=[args.crop_height, args.crop_width], preset_model = args.model, num_classes=num_classes)
elif args.model == "GCN-Res50" or args.model == "GCN-Res101" or args.model == "GCN-Res152":
    # GCN requires pre-trained ResNet weights
    network, init_fn = build_gcn(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "DeepLabV3-Res50" or args.model == "DeepLabV3-Res101" or args.model == "DeepLabV3-Res152":
    # DeepLabV requires pre-trained ResNet weights
    network, init_fn = build_deeplabv3(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "DeepLabV3_plus-Res50" or args.model == "DeepLabV3_plus-Res101" or args.model == "DeepLabV3_plus-Res152":
    # DeepLabV3+ requires pre-trained ResNet weights
    network, init_fn = build_deeplabv3_plus(net_input, preset_model = args.model, num_classes=num_classes)
elif args.model == "AdapNet":
    network = build_adaptnet(net_input, num_classes=num_classes)
elif args.model == "custom":
    network = build_custom(net_input, num_classes)
else:
    raise ValueError("Error: the model %d is not available. Try checking which models are available using the command python main.py --help")

losses = None
if args.class_balancing:
    print("Computing class weights for", args.dataset, "...")
    class_weights = utils.compute_class_weights(labels_dir=args.dataset + "/train_labels", label_values=label_values)
    unweighted_loss = None
    if args.loss_func == "cross_entropy":
        unweighted_loss = tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output)
    elif args.loss_func == "lovasz":
        unweighted_loss = utils.lovasz_softmax(probas=network, labels=net_output)
    losses = unweighted_loss * class_weights
else:
    if args.loss_func == "cross_entropy":
        losses = tf.nn.softmax_cross_entropy_with_logits(logits=network, labels=net_output)
    elif args.loss_func == "lovasz":
        losses = utils.lovasz_softmax(probas=network, labels=net_output)
loss = tf.reduce_mean(losses)

opt = tf.train.AdamOptimizer(0.0001).minimize(loss, var_list=[var for var in tf.trainable_variables()])

saver=tf.train.Saver(max_to_keep=1000)
sess.run(tf.global_variables_initializer())

utils.count_params()

# If a pre-trained ResNet is required, load the weights.
# This must be done AFTER the variables are initialized with sess.run(tf.global_variables_initializer())
if init_fn is not None:
    init_fn(sess)

# Load a previous checkpoint if desired
model_checkpoint_name = "checkpoints/latest_model_" + args.model + "_" + args.dataset + ".ckpt"
if args.continue_training or not args.mode == "train":
    print('Loaded latest model checkpoint')
    saver.restore(sess, model_checkpoint_name)

avg_scores_per_epoch = []

# Load the data
print("Loading the data ...")
train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = prepare_data()

if args.mode == "train":

    print("\n***** Begin training *****")
    print("Dataset -->", args.dataset)
    print("Model -->", args.model)
    print("Crop Height -->", args.crop_height)
    print("Crop Width -->", args.crop_width)
    print("Num Epochs -->", args.num_epochs)
    print("Batch Size -->", args.batch_size)
    print("Num Classes -->", num_classes)

    print("Data Augmentation:")
    print("\tVertical Flip -->", args.v_flip)
    print("\tHorizontal Flip -->", args.h_flip)
    print("\tBrightness Alteration -->", args.brightness)
    print("\tRotation -->", args.rotation)
    print("")

    avg_loss_per_epoch = []

    # Which validation images do we want
    val_indices = []
    num_vals = min(args.num_val_images, len(val_input_names))

    # Set random seed to make sure models are validated on the same validation images.
    # So you can compare the results of different models more intuitively.
    random.seed(16)
    val_indices=random.sample(range(0,len(val_input_names)),num_vals)
    
    # record training and validation logloss 
    train_log=open("checkpoints/train_log.csv",'w')
    val_log=open("checkpoints/val_log.csv",'w')

    # Do the training here
    for epoch in range(0, args.num_epochs):

        # Create directories if needed
        if not os.path.isdir("%s/%04d"%("checkpoints",epoch)):
            os.makedirs("%s/%04d"%("checkpoints",epoch))
        train_l=open("%s/%04d/train_scores.csv"%("checkpoints",epoch),'w')
        train_l.write("Epoch, Count, Current_Loss, Time\n")

        current_losses = []

        cnt=0
        # Equivalent to shuffling
        id_list = np.random.permutation(len(train_input_names))

        num_iters = int(np.floor(len(id_list) / args.batch_size))
        st = time.time()
        epoch_st=time.time()
        for i in range(num_iters):
            # st=time.time()
            
            input_image_batch = []
            output_image_batch = [] 

            # Collect a batch of images
            for j in range(args.batch_size):
                index = i*args.batch_size + j
                id = id_list[index]
                input_image = load_image(train_input_names[id])
                output_image = load_image(train_output_names[id])

                with tf.device('/cpu:0'):
                    input_image, output_image = data_augmentation(input_image, output_image)


                    # Prep the data. Make sure the labels are in one-hot format
                    input_image = np.float32(input_image) / 255.0
                    output_image = np.float32(helpers.one_hot_it(label=output_image, label_values=label_values))
                    
                    input_image_batch.append(np.expand_dims(input_image, axis=0))
                    output_image_batch.append(np.expand_dims(output_image, axis=0))

            if args.batch_size == 1:
                input_image_batch = input_image_batch[0]
                output_image_batch = output_image_batch[0]
            else:
                input_image_batch = np.squeeze(np.stack(input_image_batch, axis=1))
                output_image_batch = np.squeeze(np.stack(output_image_batch, axis=1))

            # Do the training
            _,current=sess.run([opt,loss],feed_dict={net_input:input_image_batch,net_output:output_image_batch})
            current_losses.append(current)
            cnt = cnt + args.batch_size
            if cnt % 20 == 0:
                ######sanity check for nan loss########
                if not (current >= 0):
                    error_log=open("checkpoints/ERROR",'w')
                    error_log.write("NaN training error from Epoch %d " % epoch)
                    #break
                    exit()
                #####sanity check ends#################
                string_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f"%(epoch,cnt,current,time.time()-st)
                train_l.write("%d, %d, %.4f, %.2f \n"%(epoch, cnt, current,time.time()-st ))
                utils.LOG(string_print)
                train_log.write("%s \n" % string_print)  # save training log
                #st = time.time()

        mean_loss = np.mean(current_losses)
        avg_loss_per_epoch.append(mean_loss)
        train_l.close()
        
        # Create directories if needed
        if not os.path.isdir("%s/%04d"%("checkpoints",epoch)):
            os.makedirs("%s/%04d"%("checkpoints",epoch))

        # Save latest checkpoint to same file name
        print("Saving latest checkpoint")
        #print (model_checkpoint_name)
        saver.save(sess,model_checkpoint_name)

        if val_indices != 0 and (epoch % args.checkpoint_step == 0 or epoch >= 40):
            print("Saving checkpoint for this epoch")
            saver.save(sess,"%s/%04d/model.ckpt"%("checkpoints",epoch))

        if epoch % args.validation_step == 0:
            val_st = time.time()
            print("Performing validation")
            target=open("%s/%04d/val_scores.csv"%("checkpoints",epoch),'w')
            target.write("val_name,loss, time, avg_accuracy, precision, recall, f1 score, mean iou, %s\n" % (class_names_string))

            scores_list = []
            class_scores_list = []
            precision_list = []
            recall_list = []
            f1_list = []
            iou_list = []
            val_loss_list = [] # save validation loss 

            # Do the validation on a small set of validation images
            for ind in val_indices:
                input_image = np.expand_dims(np.float32(load_image(val_input_names[ind])[:args.crop_height, :args.crop_width]),axis=0)/255.0
                gt = load_image(val_output_names[ind])[:args.crop_height, :args.crop_width]
                gt_tmp = helpers.one_hot_it(label=gt, label_values=label_values)
                gt_image = np.expand_dims(np.float32(gt_tmp),axis=0)
                gt = helpers.reverse_one_hot(gt_tmp)

                # st = time.time()
                #output_image = sess.run(network,feed_dict={net_input:input_image})
                output_image ,current=sess.run([network,loss],feed_dict={net_input:input_image,net_output:gt_image})
                output_image = np.array(output_image[0,:,:,:])
                output_image = helpers.reverse_one_hot(output_image)
                out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

                accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)
            
                file_name = utils.filepath_to_name(val_input_names[ind])
                target.write("%s, %f, %.2f, %f, %f, %f, %f, %f"%(file_name, current,time.time()-val_st, accuracy, prec, rec, f1, iou))
                for item in class_accuracies:
                    target.write(", %f"%(item))
                target.write("\n")

                scores_list.append(accuracy)
                class_scores_list.append(class_accuracies)
                precision_list.append(prec)
                recall_list.append(rec)
                f1_list.append(f1)
                iou_list.append(iou)
                val_loss_list.append(current)
                #print ("CURRENT LOSS: ",current) ## add val loss
                
                gt = helpers.colour_code_segmentation(gt, label_values)
     
                file_name = os.path.basename(val_input_names[ind])
                file_name = os.path.splitext(file_name)[0]
                #cv2.imwrite("%s/%04d/%s_pred.png"%("checkpoints",epoch, file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
                #cv2.imwrite("%s/%04d/%s_gt.png"%("checkpoints",epoch, file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))
            target.close()

            avg_score = np.mean(scores_list)
            class_avg_scores = np.mean(class_scores_list, axis=0)
            avg_scores_per_epoch.append(avg_score)
            avg_precision = np.mean(precision_list)
            avg_recall = np.mean(recall_list)
            avg_f1 = np.mean(f1_list)
            avg_iou = np.mean(iou_list)
            avg_val_loss = np.mean(val_loss_list)

            print("\nAverage validation accuracy for epoch # %04d = %f"% (epoch, avg_score))
            print("Average per class validation accuracies for epoch # %04d:"% (epoch))
            for index, item in enumerate(class_avg_scores):
                print("%s = %f" % (class_names_list[index], item))
            print("Validation precision = ", avg_precision)
            print("Validation recall = ", avg_recall)
            print("Validation F1 score = ", avg_f1)
            print("Validation IoU score = ", avg_iou)
            print ("Validation logloss: ",avg_val_loss) ## val loss

        epoch_time=time.time()-epoch_st
        remain_time=epoch_time*(args.num_epochs-1-epoch)
        m, s = divmod(remain_time, 60)
        h, m = divmod(m, 60)
        if s!=0:
            train_time="Remaining training time = %d hours %d minutes %d seconds\n"%(h,m,s)
        else:
            train_time="Remaining training time : Training completed.\n"
        utils.LOG(train_time)
        scores_list = []
        ######
        val_print = "Epoch = %d Count = %d Current_Loss = %.4f Time = %.2f"%(epoch,len(val_loss_list),avg_val_loss,epoch_time)
        val_log.write("%s \n" % val_print)  # save training log
        st = time.time()

    ##
    train_log.close()
    val_log.close()
    ##

    fig = plt.figure(figsize=(11,8))
    ax1 = fig.add_subplot(111)

    ax1.plot(range(args.num_epochs), avg_scores_per_epoch)
    ax1.set_title("Average validation accuracy vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Avg. val. accuracy")

    plt.savefig('accuracy_vs_epochs.png')

    plt.clf()

    ax1 = fig.add_subplot(111)

    ax1.plot(range(args.num_epochs), avg_loss_per_epoch)
    ax1.set_title("Average loss vs epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Current loss")

    plt.savefig('loss_vs_epochs.png')

elif args.mode == "test":
    print("\n***** Begin testing *****")
    print("Dataset -->", args.dataset)
    print("Model -->", args.model)
    print("Crop Height -->", args.crop_height)
    print("Crop Width -->", args.crop_width)
    print("Num Classes -->", num_classes)
    print("")

    # Create directories if needed
    if not os.path.isdir("%s"%("Val")):
            os.makedirs("%s"%("Val"))

    target=open("%s/val_scores.csv"%("Val"),'w')
    target.write("val_name, avg_accuracy, precision, recall, f1 score, mean iou %s\n" % (class_names_string))
    scores_list = []
    class_scores_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    iou_list = []
    run_times_list = []

    # Run testing on ALL test images
    for ind in range(len(val_input_names)):
        sys.stdout.write("\rRunning test image %d / %d"%(ind+1, len(val_input_names)))
        sys.stdout.flush()

        input_image = np.expand_dims(np.float32(load_image(val_input_names[ind])[:args.crop_height, :args.crop_width]),axis=0)/255.0
        gt = load_image(val_output_names[ind])[:args.crop_height, :args.crop_width]
        gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

        st = time.time()
        output_image = sess.run(network,feed_dict={net_input:input_image})

        run_times_list.append(time.time()-st)

        output_image = np.array(output_image[0,:,:,:])
        output_image = helpers.reverse_one_hot(output_image)
        out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

        accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)
    
        file_name = utils.filepath_to_name(val_input_names[ind])
        target.write("%s, %f, %f, %f, %f, %f"%(file_name, accuracy, prec, rec, f1, iou))
        for item in class_accuracies:
            target.write(", %f"%(item))
        target.write("\n")

        scores_list.append(accuracy)
        class_scores_list.append(class_accuracies)
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)
        iou_list.append(iou)
        
        gt = helpers.colour_code_segmentation(gt, label_values)

        cv2.imwrite("%s/%s_pred.png"%("Val", file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
        cv2.imwrite("%s/%s_gt.png"%("Val", file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))


    target.close()

    avg_score = np.mean(scores_list)
    class_avg_scores = np.mean(class_scores_list, axis=0)
    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    avg_f1 = np.mean(f1_list)
    avg_iou = np.mean(iou_list)
    avg_time = np.mean(run_times_list)
    print("Average test accuracy = ", avg_score)
    print("Average per class test accuracies = \n")
    for index, item in enumerate(class_avg_scores):
        print("%s = %f" % (class_names_list[index], item))
    print("Average precision = ", avg_precision)
    print("Average recall = ", avg_recall)
    print("Average F1 score = ", avg_f1)
    print("Average mean IoU score = ", avg_iou)
    print("Average run time = ", avg_time)

elif args.mode == "predict":

    if args.video is None:
        ValueError("You must pass an video path when using prediction mode.")
    video_output = None
    print("\n***** Begin prediction *****")
    print("Dataset -->", args.dataset)
    print("Model -->", args.model)
    #print("Crop Height -->", args.crop_height)
    #print("Crop Width -->", args.crop_width)
    print("Num Classes -->", num_classes)
    print("Video -->", args.video)
    print("")
    
    sys.stdout.write("Testing video " + args.video)
    sys.stdout.flush()
     
    video_input = cv2.VideoCapture(args.video)
    video_name = os.path.basename(args.video)
    video_name = video_name.replace(".mov","")
    i = 0
    while video_input.grab():
        flag, image = video_input.retrieve()
        if not flag:
            break
        H, W, _ = image.shape
        image = cv2.resize(image,(W//2,H//2),interpolation=cv2.INTER_NEAREST)
        #cv2.imwrite(os.path.join("Test","%s.png" % i),image)
        H, W, _ = image.shape
        input_image = np.expand_dims(np.float32(image),axis=0)/255.0
        output_image = sess.run(network,feed_dict={net_input:input_image})
        output_image = np.array(output_image[0,:,:,:])
        output_image = helpers.reverse_one_hot(output_image)
        class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
        out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
        if video_output is None:
            video_output = cv2.VideoWriter(os.path.join("Test","%s.avi" % video_name),cv2.VideoWriter_fourcc(*"MJPG"),args.fps,(W*2,H))
            
        combine_image = np.concatenate((image,cv2.cvtColor(out_vis_image.astype(np.uint8),cv2.COLOR_RGB2BGR)),axis=1)
        #convert_color = cv2.cvtColor(combine_image.astype(np.uint8),cv2.COLOR_RGB2BGR)
        #video_output.write(convert_color.astype(np.uint8))
        video_output.write(combine_image.astype(np.uint8))
        i += 1 
        pass
    video_output.release()

    '''       
    # to get the right aspect ratio of the output
    for imgs in glob(os.path.join(args.image,"*")):
    #loaded_image = load_image(args.image)
        loaded_image = load_image(imgs)
        height, width, channels = loaded_image.shape
        resize_height = int(height / (width / args.crop_width))

        resized_image =cv2.resize(loaded_image, (args.crop_width, resize_height))
        input_image = np.expand_dims(np.float32(resized_image[:args.crop_height, :args.crop_width]),axis=0)/255.0

        st = time.time()
        output_image = sess.run(network,feed_dict={net_input:input_image})

        run_time = time.time()-st

        output_image = np.array(output_image[0,:,:,:])
        output_image = helpers.reverse_one_hot(output_image)

        # this needs to get generalized
        #class_names_list, label_values = helpers.get_label_info(os.path.join("CamVid", "class_dict.csv"))
        class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))

        out_vis_image = helpers.colour_code_segmentation(output_image, label_values)
        #file_name = utils.filepath_to_name(args.image)
        file_name = utils.filepath_to_name(imgs)
        cv2.imwrite("%s/%s_pred.png"%("Test", file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
    '''
        #print("")
        #print("Finished!")
        #print("Wrote image " + "%s/%s_pred.png"%("Test", file_name))

else:
    ValueError("Invalid mode selected.")
