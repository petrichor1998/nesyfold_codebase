"""
This is the pipeline for labelling the filters using the ADE20k dataset

"""
import copy
import shutil
import argparse
import pandas as pd
import torch.nn as nn
from torchvision.models._utils import IntermediateLayerGetter
from ERIC_datasets import MNIST_dataloaders, GTSRB_dataloaders, PASCALanimals_dataloaders, PASCALall_dataloaders, \
    PLACES_dataloaders
from custom_dataset import create_train_test, Places365_train_test, Places365_val
import cv2
import re
import torch
from collections import Counter
from tqdm import tqdm
from torchvision.models import vgg16
import numpy as np
from torchvision.transforms import ToPILImage
from torch.utils.data import DataLoader
import torchvision.transforms as T
import PIL
import json
import os
import time
from torchvision.models import vgg16
from torchvision.models._utils import IntermediateLayerGetter
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
from algo import create_norm_tensor
from custom_dataset import Places365_train_test, Places365_val,create_train_test
import torch
from collections import Counter
from tqdm import tqdm
import numpy as np
# from train_models import VGG16
from ERIC_datasets import MNIST_dataloaders, GTSRB_dataloaders, PASCALanimals_dataloaders, PASCALall_dataloaders, \
    PLACES_dataloaders, ADE20k
# from torchvision.datasets import SUN397
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from dataloaders import get_class_list
from torch.utils.data import random_split
import torchvision.transforms as T
from torch.optim import SGD, Adam, RMSprop
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorboard as tb

from filter_visualize2 import create_labelled_rules
from label_filters import find_images_for_filters
def save_filter_label_json(margin, filter_obj_dict_name, top_object_dict, filter_obj_dict_dir):

    filter_obj_dict_path = os.path.join(filter_obj_dict_dir, filter_obj_dict_name)
    top_object_dict_pruned = {}

    for filter, v_list in top_object_dict.items():
        top_val = v_list[0][1][0]
        for tup in v_list:
            if tup[1][0] < top_val - margin:
                top_object_dict_pruned[filter] = v_list[:v_list.index(tup)]
                break

            if v_list.index(tup) == len(v_list) - 1:
                top_object_dict_pruned[filter] = v_list
    final_filter_object_dict = {}
    used_object_dict = {}
    for filter, val_list in top_object_dict_pruned.items():
        obj_string = ""
        for tup in val_list:
            if tup[0] in used_object_dict.keys():
                if len(tup[0].split()) == 2:
                    obj_string += tup[0].replace(" ", "_")
                else:
                    obj_string += tup[0]
                obj_string += str(used_object_dict[tup[0]]) + "_"
                used_object_dict[tup[0]] += 1
            else:
                if len(tup[0].split()) == 2:
                    obj_string += tup[0].replace(" ", "_")
                else:
                    obj_string += tup[0]
                obj_string += "1_"
                used_object_dict[tup[0]] = 2
        obj_string = obj_string[:-1]
        final_filter_object_dict[filter] = obj_string
    # save the final_filter_object_dict to a json file
    with open(filter_obj_dict_path, "w") as f:
        json.dump(final_filter_object_dict, f)
    return final_filter_object_dict
def main():
    parser = argparse.ArgumentParser(prog="exp2.py", description="Run the semantic labelling algo")
    parser.add_argument("--dataset_name", type=str, help="name of the dataset")
    # parser.add_argument("--norm_tensor_dir", type=str, help="path to store the norm tensor generated")
    parser.add_argument("--model_check_path", type=str, help="model checkpoint path")
    parser.add_argument("--rules_path", type=str, help="path of the rules file")
    # parser.add_argument("--images_dir", type=str, help="path of the images directory to store the top-10 images")
    # parser.add_argument("--filter_masks_dir", type=str, help="path of the images directory to store the top-10 images filter masks")
    # parser.add_argument("--classes", type=str, help="the names of classes to be ussed from the ADE20k dataset seperated by a '-' ")
    parser.add_argument("--ebp", default=False, action='store_true', help="whether the rule-set is generated with ebp or not")

    parser.add_argument("--margin", default=0.05,type=float, help="margin for approximating the predicate labels")
    args = parser.parse_args()
    dataset_name = args.dataset_name
    EBP = args.ebp
    class_list = get_class_list(dataset_name)
    margin = args.margin
    # margin = 0.05 #0.1, 0.15, 0.2, 1.5, 2.0
    percentile = 50
    #ignore
    algo_type = "single"
    # algo_type = "group"
    # Get the ADE20k dataset dataloaders and use the algo.py to ge the norm tensors.
    # norm_tensor_dir = args.norm_tensor_dir + dataset_name + f"/{algo_type}/"
    if EBP:
        norm_tensor_dir = "norm_tensors/ADE20k/EBP/" + dataset_name + f"/{algo_type}/"
        checkpoint_path = args.model_check_path
        rules_path = args.rules_path
        images_dir = "filter_images/" + dataset_name + f"/EBP/{algo_type}"
        filter_masks_dir = "filter_masks/" + dataset_name + f"/EBP/{algo_type}/"
        labelled_rules_dir = "labelled_rules/EBP/" + dataset_name + f"/{algo_type}/"
        filter_obj_dict_dir = "filter_obj_dict/EBP/" + dataset_name + f"/{algo_type}/"
    else:
        norm_tensor_dir = "norm_tensors/ADE20k/vanilla/" + dataset_name + f"/{algo_type}/"
        checkpoint_path = args.model_check_path
        rules_path = args.rules_path
        images_dir = "filter_images/" + dataset_name + f"/vanilla/{algo_type}"
        filter_masks_dir = "filter_masks/" + dataset_name + f"/vanilla/{algo_type}/"
        labelled_rules_dir = "labelled_rules/vanilla/" + dataset_name + f"/{algo_type}/"
        filter_obj_dict_dir = "filter_obj_dict/vanilla/" + dataset_name + f"/{algo_type}/"

    # class_list = ["bathroom", "bedroom", "kitchen", "dining_room", "living_room"]
    # class_list = ["bathroom", "bedroom", "kitchen", "dining_room", "living_room", "home_office", "office", "waiting_room", "conference_room", "hotel_room"]
    # class_list = ["desert_road", "forest_road", "street"]
    # create the labelled rules dir
    if not os.path.exists(labelled_rules_dir):
        os.makedirs(labelled_rules_dir)
    if not os.path.exists(norm_tensor_dir):
        os.makedirs(norm_tensor_dir)

    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    elif os.path.exists(images_dir):
        shutil.rmtree(images_dir)
        os.makedirs(images_dir)

    if not os.path.exists(filter_obj_dict_dir):
        os.makedirs(filter_obj_dict_dir)

    if not os.path.exists(filter_masks_dir):
        os.makedirs(filter_masks_dir)
    elif os.path.exists(filter_masks_dir):
        # delete the dir and create a new one
        shutil.rmtree(filter_masks_dir)
        os.makedirs(filter_masks_dir)


    #extract the file name from the rules path
    rules_name = os.path.basename(rules_path)
    labelled_rules_name = "labelled_" + rules_name
    norm_tensor_name = "norm_tensor_" + rules_name[:-4] + ".pt"



    norm_tensor_path = os.path.join(norm_tensor_dir, norm_tensor_name)
    labelled_rules_path = os.path.join(labelled_rules_dir, labelled_rules_name)
    dataset = ADE20k("/nesyfold_codebase", class_list = class_list, transform = T.Compose([T.Resize((224,224)), T.ToTensor()]))
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = vgg16(num_classes=len(class_list))
    device = "cuda:0"
    chkpoint = torch.load(checkpoint_path)
    model.load_state_dict(chkpoint["model_state_dict"])
    model.to(device)
    # create the norm tensor
    create_norm_tensor(data_loader, model, norm_tensor_path,device)
    # Store images that most activate the filters in the CNN
    find_images_for_filters(dataset, rules_path, norm_tensor_path, images_dir)
    # create the labelled rules
    # create_labelled_rules(model, images_dir, class_list, rules_path, labelled_rules_path, filter_obj_dict_path)
    # def create_labelled_rules(model, filter_images_dir, class_list, rules_path, labelled_rules_path, filter_obj_dict_path, margin = 0.05, device="cpu"):

    device = "cpu"
    print("Creating labelled rules")
    model.to(device)
    required_layers = {"30": "conv13"}

    #storing the activations for all images from layer 13
    model_inter = IntermediateLayerGetter(model.features, return_layers=required_layers)

    #
    # find the feature maps corresponding to each filter for the first image in the train_loader
    trans1 = T.Resize((224,224))
    trans2 = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    # image_in = trans2(trans1(image))
    # images = images.to(device)
    # labels = labels.to(device)
    # filter_images_dir = "/home/pxp180054/projects/XAI_images/filter_images"
    # the keys of this dict will behe filter numbers and the values will be the dictionary with keys as objects and values as percentages sorted descending by values
    filter_obj_dict = {}
    # iterate over each filter dir that has filter images that most activate it
    for filter in os.listdir(images_dir):

        obj_percent_dict = {}
        filter_path = os.path.join(images_dir, filter)
        # create the filter mask dir
        # if the dir exists then replace it
        if os.path.exists(os.path.join(filter_masks_dir, filter)):
            shutil.rmtree(os.path.join(filter_masks_dir, filter))
            os.makedirs(os.path.join(filter_masks_dir, filter))
        else:
            os.makedirs(os.path.join(filter_masks_dir, filter))

        # iterate over each image in the filter dir
        for image in os.listdir(filter_path):
            if image.endswith(".jpg"):
                image_path = os.path.join(filter_path, image)
                im = PIL.Image.open(image_path)
                im_in = trans2(trans1(im))
                features = model_inter(im_in)
                features = features["conv13"]
                features = features.detach().numpy()
                fm1_normal = features[int(filter)]
                fm1_normal_re = cv2.resize(fm1_normal, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                # fm1_inter = np.interp(fm1_normal_re, (fm1_normal_re.min(), fm1_normal_re.max()), (0, 100))
                flattened_arr = fm1_normal_re.flatten()
                # non_zero_values = flattened_arr[flattened_arr > 0]
                perc = np.percentile(flattened_arr, percentile)
                mask_combined = (fm1_normal_re < perc)
                fm1_normal_re[mask_combined] = 0
                mask_3d = np.random.rand(224, 224, 3)
                # mask_3d[:, :, :] = fm1_normal_re[:, :, np.newaxis] <= 0.03
                mask_3d[:, :, :] = fm1_normal_re[:, :, np.newaxis] == 0
                im1 = trans1(im)
                im1_np = np.asarray(im1)
                im1_copy = im1_np.copy()
                mask_3d = mask_3d.astype(bool)
                im1_copy[mask_3d] = 0
                im1_copy = ToPILImage()(im1_copy)
                im1_copy.save(os.path.join(filter_masks_dir,filter,image[:-4] + "_mask.jpg"))
                # break

                with open(os.path.join(filter_path, image[:-4] + ".txt")) as f:
                    json_add = f.read().split()[1]
                with open(json_add, encoding='latin') as f:
                    info_dict = json.load(f)

                obj_mask_dict = {}
                for obj in info_dict["annotation"]["object"]:
                    # if obj["raw_name"] not in ["ceiling", "floor", "wall", "person", "side", "front", "sky"]:
                    obj_mask_dict[(obj["id"], obj["raw_name"])] = obj["instance_mask"]
                t = json_add.split("/")[:-1]
                img_dir = "/".join(t)
                obj_non_zero_dict = {}
                for obj in obj_mask_dict:
                    mask = PIL.Image.open(img_dir + "/" + obj_mask_dict[obj])
                    # mask = trans1(mask)
                    mask_rgb_og = mask.convert("RGB")
                    mask_np_og = np.asarray(mask)
                    tmask = mask_np_og != 255
                    mask_np_og_copy = mask_np_og.copy()
                    mask_np_og_copy[tmask] = 0

                    mask_img = ToPILImage()(mask_np_og_copy)
                    mask_re = trans1(mask_img)
                    mask_rgb_re = mask_re.convert("RGB")
                    mask_np_re = np.asarray(mask_rgb_re)
                    mask_np_bool = mask_np_re[:, :, :] == 0
                    mask_np_bool = mask_np_bool.astype(bool)
                    im1_copy_np = np.asarray(im1_copy).copy()
                    # og_non_zero = im1_copy_np.any(axis=2).sum()
                    og_non_zero = fm1_normal_re.sum()
                    im1_copy_np[mask_np_bool] = 0

                    # check  the numer of non zero pixels in the mask_np_bool

                    non_zero_arr = im1_copy_np.any(axis=-1)
                    new_mask = non_zero_arr == False
                    fm1_inter_copy = fm1_normal_re.copy()
                    fm1_inter_copy[new_mask] = 0
                    non_zero = fm1_inter_copy.sum()
                    # non_zero = im1_copy_np.sum()

                    if og_non_zero == 0:
                        obj_non_zero_dict[obj] = 0
                    else:
                        obj_non_zero_dict[obj] = non_zero / og_non_zero
        #             break

        # break
                # combine percentages of the same objcets and store in a
                temp_obj_percent_dict = {}
                for obj in obj_non_zero_dict:
                    if obj[1] not in temp_obj_percent_dict:
                        temp_obj_percent_dict[obj[1]] = obj_non_zero_dict[obj]

                    else:
                        temp_obj_percent_dict[obj[1]] += obj_non_zero_dict[obj]
                # get the frequency of each object
                for obj in temp_obj_percent_dict:
                    if temp_obj_percent_dict[obj] > 0.1:
                        temp_obj_percent_dict[obj] = (temp_obj_percent_dict[obj], 1)
                    else:
                        temp_obj_percent_dict[obj] = (temp_obj_percent_dict[obj], 0)
                # sort the dictionary by values
                for obj in temp_obj_percent_dict:
                    if obj not in obj_percent_dict:
                        obj_percent_dict[obj] = temp_obj_percent_dict[obj]
                    else:
                        obj_percent_dict[obj] = (obj_percent_dict[obj][0] + temp_obj_percent_dict[obj][0], obj_percent_dict[obj][1] + temp_obj_percent_dict[obj][1])
                obj_percent_dict = {k: v for k, v in sorted(obj_percent_dict.items(), key=lambda item: item[1][0], reverse=True)}

        filter_obj_dict[filter] = obj_percent_dict



    #createa a top_object_dict 10 objects for each filter of the form {filter : [(obj1, percent1), (obj2, percent2), ...]}

    top_object_dict = {}
    for filter in filter_obj_dict:
        top_object_dict[filter] = list(filter_obj_dict[filter].items())[:10]
    #create a object_filter_dict of the form {obj1: [(filter1, percent1), (filter2, percent2), ...], obj2: [(filter1, percent1), (filter2, percent2), ...], ...}
    object_filter_dict = {}
    for filter in top_object_dict:
        for obj in top_object_dict[filter]:
            if obj[0] not in object_filter_dict:
                object_filter_dict[obj[0]] = [(filter, obj[1])]
            else:
                object_filter_dict[obj[0]].append((filter, obj[1]))
    # sor the object_filter_dict by the percentage of the object in the filter
    for obj in object_filter_dict:
        object_filter_dict[obj] = sorted(object_filter_dict[obj], key=lambda x: x[1][0], reverse=True)

    # normalizing the percentages of the objects in the filters
    for filter, v_list in top_object_dict.items():
        total = 0
        for tup in v_list:
            total += tup[1][0]
        for i in range(len(v_list)):
            v_list[i] = (v_list[i][0], (v_list[i][1][0]/total, v_list[i][1][1]))
        # for tup in object_filter_dict[obj]:
        #     object_filter_dict[obj][object_filter_dict[obj].index(tup)] = (tup[0], (tup[1][0]/total, tup[1][1]))

    # keep only the top few objects for each filter according to whether their percentages fallwithin -1.0
    # for margin in [0.2]:
    # for margin in [0.05]:
    print("Saving for margin: ", margin)
    filter_obj_dict_name = "filter_obj_dict_" + rules_name[:-4] + "_" + str(margin) + ".json"
    final_filter_object_dict = save_filter_label_json(margin, filter_obj_dict_name, top_object_dict, filter_obj_dict_dir)
    with open(rules_path, "r") as f:
        rules = f.read()
    for f, o in final_filter_object_dict.items():
        rules = re.sub("(?<!\d)(?<!\w)" + f + "(?=\()", o, rules)

    for i in range(len(class_list)):
        sub_pattern = "512\(X,'" + str(i) + "'\)"
        rep_pattern = "target(X,'" + class_list[i] + "')"
        rules = re.sub(sub_pattern, rep_pattern, rules)
    with open(labelled_rules_path, "w") as f:
         f.write(rules)
    print(rules)
    print("Stored labelled rules")

if __name__ == '__main__':
    main()