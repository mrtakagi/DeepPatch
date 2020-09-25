from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

Image.LOAD_TRUNCATED_IMAGES = True


def logit(x):
    return math.log(x/(1-x))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--try_fixing", type=bool, default=False, help="try fixing weight")
    parser.add_argument("--fixed_parameter_file", type=str, default="", help="fixed weight")
    parser.add_argument("--fixed_conf_thres", type=float, default=0.9, help="fixed conf thres")
    parser.add_argument("--fixed_class", type=int, default=None, help="fixed class")
    parser.add_argument("--output", type=str, default="output", help="output directory")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))
    #model.module_list[93][0].weight[85*(6%3)+4, 24, 0, 0] = 0.6
    if opt.fixed_parameter_file != "":
        with open(f"{opt.fixed_parameter_file}", 'r') as f:
            code=f.read()
            print(code)
            exec(code)
    model.eval()  # Set in evaluation mode
    #import pdb;pdb.set_trace()

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index


    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    print("\nPerforming object detection:")
    prev_time = time.time()
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres, True)

        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        #imgs.extend(img_paths)
        #img_detections.extend(detections)
        detections=detections[0]

        print("\nSaving images:")
        # Iterate through images and save plot of detections
        #for img_i, (path, detections) in enumerate(zip(imgs, img_detections)):
        path=img_paths[0]
        filename = ".".join(path.split("/")[-1].split(".")[0:-1])
    
        print("(%d) Image: '%s'" % (batch_i, path))
    
        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)
    
        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            # 0: x1
            # 1: y1
            # 2: x2
            # 3: y2
            # 4: conf
            # 5: cls_conf
            # 6: cls_pre
            # 7: yoloid
            # 8: grid
            # 9: x
            # 10: y
            # 11: layer
            # 12: weight
            unique_labels = detections[:, 6].cpu().unique()
            n_cls_preds = 16 # len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            with open(f"{opt.output}/{filename}.csv", 'w') as f:
                print("Label,Conf,ObjConf,YoloId,GridSize,X,Y,Layer",file=f)
                detection_id = 0
                for x1, y1, x2, y2, conf, cls_conf, cls_pred, yoloid, grid_size, grid_x, grid_y, layer, weightpos in detections:
                    print("\t+ Id: %d, Label: %s, Conf: %.5f , ObjConf: %.5f , YoloId: %d, GridSize: %d, X: %d, Y: %d, Layer: %d" %
                          (detection_id,
                           classes[int(cls_pred)],
                           cls_conf.item(),
                           conf,
                           yoloid,
                           grid_size,
                           grid_x,
                           grid_y,
                           layer
                          ))
                    print("%d,%s,%.5f,%.5f,%d,%d,%d,%d,%d" %
                          (detection_id,
                           classes[int(cls_pred)],
                           cls_conf.item(),
                           conf,
                           yoloid,
                           grid_size,
                           grid_x,
                           grid_y,
                           layer
                          ),
                          file=f)
                    #if # classes[int(cls_pred)] == "person":
                    if opt.try_fixing:
                        if opt.fixed_class is not None:
                            fixed_classid = opt.fixed_class
                            org_classid = int(cls_pred)
                            b = model.module_list[int(layer)][0].bias[85*(int(yoloid)%3)+5+fixed_classid]
                            w = model.module_list[int(layer)][0].weight[85*(int(yoloid)%3)+5+fixed_classid, :, 0, 0]
                            ob = model.module_list[int(layer)][0].bias[85*(int(yoloid)%3)+5+org_classid]
                            ow = model.module_list[int(layer)][0].weight[85*(int(yoloid)%3)+5+org_classid, :, 0, 0]
                            i = model(input_imgs,None,layer-1)[0,:,int(grid_y),int(grid_x)]
                            iw = i*w
                            oiw = i*ow
                            sorted,idxes = iw.sort(descending=True)
                            #print("--Current class--")
                            #print(torch.abs(torch.sigmoid(torch.dot(i,w)+b)-conf))
                            if fixed_classid != org_classid and torch.dot(i,w)+b <= torch.dot(i,ow)+ob:
                                #print("--Current parameter--")
                                #print(w[idxes[0]])
                                while torch.dot(i,w)+b <= torch.dot(i,ow)+ob:
                                    w[idxes[0]] *= 1.1
                                #print("--Recommended parameter--")
                                print("\t\tmodel.module_list[%d][0].weight[85*(%d%%3)+5+%d, %d, 0, 0] = %.5f" %
                                      (int(layer),
                                       int(yoloid),
                                       int(fixed_classid),
                                       int(idxes[0]),
                                       w[idxes[0]]
                                      ))
                                #print(w[idxes[0]])
                                #print("----")
                        else:
                            b = model.module_list[int(layer)][0].bias[85*(int(yoloid)%3)+4]
                            w = model.module_list[int(layer)][0].weight[85*(int(yoloid)%3)+4, :, 0, 0]
                            i = model(input_imgs,None,layer-1)[0,:,int(grid_y),int(grid_x)]
                            iw = i*w
                            sorted,idxes = iw.sort(descending=True)
                            # torch.sigmoid(torch.dot(i,w)+b) should be the same as conf.
                            #print("--Current conf--")
                            #print(torch.abs(torch.sigmoid(torch.dot(i,w)+b)-conf))
                            if torch.dot(i,w)+b <= logit(opt.fixed_conf_thres):
                                #print("--Current parameter--")
                                #print(w[idxes[0]])
                                while torch.dot(i,w)+b <= logit(opt.fixed_conf_thres):
                                    w[idxes[0]] *= 1.1
                                #print("--Recommended parameter--")
                                print("\t\tmodel.module_list[%d][0].weight[85*(%d%%3)+4, %d, 0, 0] = %.5f" %
                                      (int(layer),
                                       int(yoloid),
                                       int(idxes[0]),
                                       w[idxes[0]]
                                      ))
                                #print(w[idxes[0]])
                                #print("----")


        
                    box_w = x2 - x1
                    box_h = y2 - y1
        
                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])%16]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(
                        x1,
                        y1,
                        s=str(detection_id) + ":" + classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )
                    detection_id = detection_id + 1
    
        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.savefig(f"{opt.output}/{filename}.png", bbox_inches="tight", pad_inches=0.0)
        plt.close()
