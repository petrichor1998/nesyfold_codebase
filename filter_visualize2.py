"""
This file is to be used after the filter images are generated and present in the filter_images folder.
This file will finally generate labelled rules.
This file just labels each filter as a combination of top few objects according to the margin hyperparameter
"""
import copy
import shutil

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
import inflect
#get the data from the PLACES_dataloaders

# train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(class_list=["bathroom","bedroom", "kitchen"], train_shuffle=False)
# dataset = Places365_train_test(root= "./", class_list=["bathroom","bedroom", "kitchen"], transform=T.Resize((224,224)))
# dataset2 = Places365_train_test(root= "./", class_list=["bathroom","bedroom", "kitchen"], transform=T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), T.RandomHorizontalFlip(), T.RandomVerticalFlip()]))
# create a dataloader with dataset2
# dataloader = DataLoader(dataset2, batch_size=1, shuffle=False)

#loading the image form ADE20k dataset from bathroom class
# image = PIL.Image.open("/home/pxp180054/projects/XAI_images/ADE20K_2021_17_01/images/ADE/training/home_or_hotel/bathroom/ADE_train_00000006.jpg")
#load the pretrained model
# model = vgg16(pr.....etrained=True)
# model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 3)
# model.load_state_dict(torch.load("model_checkpoints/model_checkpoints_bathroom_bedroom_kitchen/chkpoint3.pt")["model_state_dict"])
# device = "cpu"
# model.to(device)

def create_labelled_rules(model, filter_images_dir, class_list, rules_path, labelled_rules_path, filter_obj_dict_path, filter_masks_dir, margin = 0.05, device="cpu"):
    print("Creating labelled rules")
    model.to(device)
    required_layers = {"30": "conv13"}

    #storing the activations for all images from layer 13
    model_inter = IntermediateLayerGetter(model.features, return_layers=required_layers)

    #
    # find the feature maps corresponding to each filter for the first image in the train_loader
    trans1 = T.Resize((224,224))
    trans2 = T.Compose([T.ToTensor()])
    # image_in = trans2(trans1(image))
    # images = images.to(device)
    # labels = labels.to(device)
    # filter_images_dir = "/home/pxp180054/projects/XAI_images/filter_images"
    # the keys of this dict will behe filter numbers and the values will be the dictionary with keys as objects and values as percentages sorted descending by values
    filter_obj_dict = {}
    # iterate over each filter dir that has filter images that most activate it
    for filter in os.listdir(filter_images_dir):
        obj_percent_dict = {}
        filter_path = os.path.join(filter_images_dir, filter)
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
                mask_3d = np.random.rand(224, 224, 3)
                mask_3d[:, :, :] = fm1_normal_re[:, :, np.newaxis] <= 0.03
                im1 = trans1(im)
                im1_np = np.asarray(im1)
                im1_copy = im1_np.copy()
                mask_3d = mask_3d.astype(bool)
                im1_copy[mask_3d] = 0
                im1_copy = ToPILImage()(im1_copy)
                im1_copy.save(os.path.join(filter_masks_dir, filter, image[:-4] + "_mask.jpg"))


                # image_path = os.path.join(filter_path, image)
                # im = PIL.Image.open(image_path)
                # im_in = trans2(trans1(im))
                # features = model_inter(im_in)
                # features = features["conv13"]
                # features = features.detach().numpy()
                # fm1_normal = features[int(filter)]
                # fm1_normal_re = cv2.resize(fm1_normal, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
                # fm1_inter = np.interp(fm1_normal_re, (fm1_normal_re.min(), fm1_normal_re.max()), (0, 100))
                # flattened_arr = fm1_inter.flatten()
                # non_zero_values = flattened_arr[flattened_arr != 0]
                # perc = np.percentile(non_zero_values, percentile)
                # mask = fm1_inter == 0
                # fm1_inter[mask] = 0
                # mask_3d = np.random.rand(224, 224, 3)
                # # mask_3d[:, :, :] = fm1_normal_re[:, :, np.newaxis] <= 0.03
                # mask_3d[:, :, :] = fm1_inter[:, :, np.newaxis] == 0
                # im1 = trans1(im)
                # im1_np = np.asarray(im1)
                # im1_copy = im1_np.copy()
                # mask_3d = mask_3d.astype(bool)
                # im1_copy[mask_3d] = 0
                # im1_copy = ToPILImage()(im1_copy)
                # im1_copy.save(os.path.join(filter_masks_dir, filter, image[:-4] + "_mask.jpg"))

                with open(os.path.join(filter_path, image[:-4] + ".txt")) as f:
                    json_add = f.read().split()[1]
                with open(json_add) as f:
                    info_dict = json.load(f)

                obj_mask_dict = {}
                for obj in info_dict["annotation"]["object"]:
                    # if obj["raw_name"] not in ["ceiling", "floor", "wall", "person", "side", "front", "sky"]:
                    obj_mask_dict[(obj["id"], obj["raw_name"])] = obj["instance_mask"]
                t = json_add.split("/")[:-1]
                img_dir = "/".join(t)
                obj_non_zero_dict = {}
                for obj in obj_mask_dict:
                    # mask = PIL.Image.open(img_dir + "/" + obj_mask_dict[obj])
                    # mask = trans1(mask)
                    # mask = mask.convert("RGB")
                    # mask_np = np.asarray(mask)
                    # mask_np = mask_np.astype(bool)
                    # mask_np_bool = mask_np[:, :, :] == 0
                    # mask_np_bool = mask_np_bool.astype(bool)
                    # im1_copy_np = np.asarray(im1_copy).copy()
                    # og_non_zero = im1_copy_np.any(axis=2).sum()
                    # im1_copy_np[mask_np_bool] = 0
                    #
                    # # check  the numer of non zero pixels in the mask_np_bool
                    # non_zero = im1_copy_np.any(axis=-1).sum()
                    # obj_non_zero_dict[obj] = non_zero / og_non_zero

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
                    # Now here instead of summing the count of non zero values
                    # I need to sum the actual non zero values
                    non_zero_arr = im1_copy_np.any(axis=-1)
                    new_mask = non_zero_arr == False
                    fm1_inter_copy = fm1_normal_re.copy()
                    fm1_inter_copy[new_mask] = 0
                    non_zero = fm1_inter_copy.sum()
                    # non_zero = im1_copy_np.sum()

                    obj_non_zero_dict[obj] = non_zero / og_non_zero

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


    # convert the plural nouns to singular and add their percentages
    # p = inflect.engine()
    # filter_obj_dict_copy = copy.deepcopy(filter_obj_dict)
    # for filter in filter_obj_dict_copy:
    #     for obj in filter_obj_dict_copy[filter]:
    #             if p.singular_noun(obj) == False:
    #                 pass
    #             else:
    #                 new_obj = p.singular_noun(obj)
    #                 if new_obj not in filter_obj_dict[filter]:
    #                     filter_obj_dict[filter][new_obj] = filter_obj_dict[filter][obj]
    #                     del filter_obj_dict[filter][obj]
    #                 else:
    #                     filter_obj_dict[filter][new_obj] += filter_obj_dict[filter][obj]
    #                     del filter_obj_dict[filter][obj]
    #     filter_obj_dict[filter] = {k: v for k, v in sorted(filter_obj_dict[filter].items(), key=lambda item: item[1], reverse=True)}

    #createa a top_object_dict 10 objects for each filter of the form {filter : [(obj1, percent1), (obj2, percent2), ...]}

    top_object_dict = {}
    for filter in filter_obj_dict:
        top_object_dict[filter] = list(filter_obj_dict[filter].items())[:15]
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

    # keep only the top few objects for each filter according to whether their percentages fallwithin -1.0
    for filter, v_list in top_object_dict.items():
        top_val  = v_list[0][1][0]
        for tup in v_list:
            if tup[1][0] <= top_val - margin:
                top_object_dict[filter] = v_list[:v_list.index(tup)]
                break
            else:
                continue
    # filter_object_dict = {}
    # used_object_list = []
    # used_filter_list = []
    # unused_filter_list = []
    # for filter, val_list in top_object_dict.items():
    #     for tup1 in val_list:
    #         if tup1[0] in used_object_list:
    #             continue
    #         else:
    #             for tup2 in object_filter_dict[tup1[0]]:
    #                 if tup2[0] in used_filter_list:
    #                     continue
    #                 elif tup2[0] == filter:
    #                     filter_object_dict[filter] = tup1
    #                     used_filter_list.append(filter)
    #                     used_object_list.append(tup1[0])
    #                     break
    #                 else:
    #                     break
    #         if filter in filter_object_dict.keys():
    #             break
    #
    #     if filter not in filter_object_dict.keys():
    #         filter_object_dict[filter] = ""
    #         unused_filter_list.append(filter)
    # final_filter_object_dict = {filter: object, filter: object, ...}
    final_filter_object_dict = {}
    used_object_dict = {}
    for filter, val_list in top_object_dict.items():
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
    # with open(filter_obj_dict_path, "w") as f:
    #     json.dump(filter_obj_dict, f)
    # final_filter_object_dict = {}
    # for filter in filter_object_dict:
    #     if filter_object_dict[filter] == "":
    #         final_filter_object_dict[filter] = ""
    #     else:
    #         #check if the value is two words. If it is then add a "_" between the words
    #         if len(filter_object_dict[filter][0].split()) == 2:
    #
    #             final_filter_object_dict[filter] = filter_object_dict[filter][0].replace(" ", "_")
    #
    #         else:
    #             final_filter_object_dict[filter] = filter_object_dict[filter][0]
    # class_list = ["bathroom", "bedroom", "kitchen"]
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
    print("Stored labelled rules")

# get the path of the directory of the images

# img_dir = "/home/pxp180054/projects/XAI_images/ADE20K_2021_17_01/images/ADE/training/home_or_hotel/bathroom/"
#
# # read the json file as a dictionary
#
#
# # iterate over info_dict["annotation"]["object"] which is a list of dicts representing each object. Get their "raw_name"
# # get their "instance_mask" file name
# obj_mask_dict = {}
# for obj in info_dict["annotation"]["object"]:
#     if obj["raw_name"] not in ["ceiling", "floor", "wall"]:
#         obj_mask_dict[(obj["id"], obj["raw_name"])] = obj["instance_mask"]
#
# #iterate over the obj_mask_dict and get the mask for each object and take intersection with the im1_copy.
# obj_non_zero_dict = {}
# for obj in obj_mask_dict:
#     mask = PIL.Image.open(img_dir + "/" + obj_mask_dict[obj])
#     mask = trans1(mask)
#     mask = mask.convert("RGB")
#     mask_np = np.asarray(mask)
#     mask_np = mask_np.astype(bool)
#     mask_np_bool = mask_np[:, :, :] == 0
#     mask_np_bool = mask_np_bool.astype(bool)
#     im1_copy_np = np.asarray(im1_copy).copy()
#     og_non_zero = im1_copy_np.any(axis=2).sum()
#     im1_copy_np[mask_np_bool] = 0
#
#     #check  the numer of non zero pixels in the mask_np_bool
#     non_zero = im1_copy_np.any(axis=-1).sum()
#     obj_non_zero_dict[obj] = non_zero/og_non_zero
# print(obj_non_zero_dict)

