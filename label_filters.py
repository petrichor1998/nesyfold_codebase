"""
Before using this file you have to generate a norm tensor that has been created on the ADE20k dataset using the algo.py file
This file uses the ADE20k dataset to generate the images that most activate your filters trained on the Places365 dataset.
"""
# find the indices of the images
import matplotlib.pyplot as plt
import os
import torch
import shutil
from ERIC_datasets import ADE20k
from custom_dataset import Places365_train_test, Places365_val, create_train_test
import torchvision.transforms as T
from relevant_filters import get_relevant_filters
from PIL import Image
# class_list = ["desert_road", "driveway", "forest_road"]
# # class_list = ["bathroom", "bedroom", "kitchen"]
# # fullset = Places365(root= "./", class_list=class_list, transform=T.Compose([T.Resize((224,224))]))
# train_test_set = Places365_train_test(root= "./", class_list=class_list, transform=T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), T.RandomHorizontalFlip(), T.RandomVerticalFlip()]))
# # val_set = Places365_val(root= "./", class_list=class_list, transform=T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), T.RandomHorizontalFlip(), T.RandomVerticalFlip()]))
# train_set, test_set = create_train_test(train_test_set)
# dataset = ADE20k("/home/pxp180054/projects/XAI_images", class_list = ["bathroom", "bedroom", "kitchen"], transform = T.Compose([T.Resize((224,224)), T.ToTensor()]))

def find_images_for_filters(dataset, rules_path, norm_tensor_path, images_dir):
# rel_filter_list = sorted([106, 188,484,206,3,34,261,94,236,181,456,2,398,421,505,240, 389, 437,217,463,372,36, 153])
    print("Finding top images for each filter...")
    rel_filter_list = get_relevant_filters(rules_path)
    norm_tensor = torch.load(norm_tensor_path)["norm_tensor"]
    vals, indices = torch.topk(norm_tensor, k =10, dim = 0)
    required_indices = indices[:, rel_filter_list]
    required_indices = required_indices.T

    #collect the images for each filter
    filter_image_dict = {}
    for i in range(required_indices.shape[0]):
        temp_list = []
        temp_json_list = []
        for j in range(required_indices.shape[1]):
            temp_list.append(dataset.__getitem__(required_indices[i][j], need_PIL=True)[0][0])
            temp_json_list.append(dataset.__getitem__(required_indices[i][j], need_PIL=True)[0][1])
        filter_image_dict[rel_filter_list[i]] = (temp_list, temp_json_list)

    # save these images to the images folder in FasterRCNN
    # create a seperate directory by the filter name to save the images
    # images_dir = "/home/pxp180054/projects/XAI_images/filter_images"


    if os.path.exists(images_dir):
        shutil.rmtree(images_dir)
    os.mkdir(images_dir)

    for filter, (im_list, json_list) in filter_image_dict.items():
        path = os.path.join(images_dir, str(filter))
        # check if dir exists and if it does then delete it and create a new one

        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
        # for im_list, json_list in duo_list:
        for i, im in enumerate(im_list):
            im.save(os.path.join(path, str(i)+".jpg"))
            with open(os.path.join(path, str(i)+".txt"), "w") as f:
                f.write(str(i) + " " + json_list[i])
    print("Saved top images...")


