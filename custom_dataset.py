from torchvision.datasets import VisionDataset
import os
import PIL
import torch
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import random
import torchvision.transforms as T
random.seed(1)

class Places365_train_test(VisionDataset):
    """
    Manages loading and transforming the data for the Places365 dataset.
    Mimics the other VisionDataset classes available in torchvision.datasets.
    Classes used by Townsend et. al.: bathroom, bedroom, kitchen
    """

    base_folder = "places365"

    def __init__(self,
                 root: str,
                 class_list = [],
                 transform = None,
                 target_transform = None,
                 ) -> None:

        # Call superconstructor
        super(Places365_train_test, self).__init__(root, transform=transform, target_transform=target_transform)

        self.test_indices = []
        self.class_list = sorted(class_list)
        self.class_to_idx_dict = {}
        for i in range(len(self.class_list)):
            self.class_to_idx_dict[self.class_list[i]] = i


        # Load data with the given classes
        self.load_data()

    def __getitem__(self, index, need_PIL=False):

        target = self.targets[index]

        # Conforming to other datasets
        image = PIL.Image.fromarray(self.data[index])

        # If a transform was provided, transform the image
        if need_PIL == False:
            image = self.transform(image)
        else:
            img_transform = T.Compose([T.Resize((224,224))])
            image = img_transform(image)
        # If a target transform was provided, transform the target
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:

        # Return the length of targets list
        return len(self.targets)

    def load_data(self):
        print("loading data")
        # Get path of extracted dataset
        image_dirs_path = os.path.join(self.root, self.base_folder, "train_256/data_256_standard")
        #Get all the images
        images_dict = {}
        for c in self.class_list:
            images_dict[c] = []

        for c in self.class_list:
            first_letter = c[0]
            for alphabet_dir in os.listdir(image_dirs_path):
                if alphabet_dir == first_letter:
                    alphabet_dir_path = os.path.join(image_dirs_path, alphabet_dir)
                    for class_dir in os.listdir(alphabet_dir_path):
                        if class_dir == c:
                            class_dir_path = os.path.join(alphabet_dir_path, class_dir)
                            images_dict[c] = images_dict[c] + os.listdir(class_dir_path)
                            break
                    break
        # Convert the images to numpy arrays
        targets = []
        image_np_list = []
        for c, im_list in images_dict.items():
            for im_idx in range(len(im_list)):
                image_path = os.path.join(image_dirs_path, c[0], c, im_list[im_idx])
                image = PIL.Image.open(image_path)
                image_numpy = np.array(image)
                image_np_list.append(image_numpy)
                targets.append(self.class_to_idx_dict[c])
            self.test_indices = self.test_indices + list(range(len(image_np_list) - 1000, len(image_np_list)))
            # print(list(range(len(image_np_list) - 1000, len(image_np_list) - 1)))
        self.targets = targets
        self.data = image_np_list
        return image_np_list, targets
class Places365_val(VisionDataset):
    """
    Manages loading and transforming the data for the Places365 dataset.
    Mimics the other VisionDataset classes available in torchvision.datasets.
    Classes used by Townsend et. al.: Desert Road, Driveway, Forest Road, Highway, Street
    Corresponding class numbers are : {"desert_road" : 118, "driveway" : 127, "forest_road" : 152, "highway": 175, "street": 319}
    """

    base_folder = "places365"

    def __init__(self,
                 root: str,
                 class_list = [],
                 transform = None,
                 target_transform = None,
                 ) -> None:

        # Call superconstructor
        super(Places365_val, self).__init__(root, transform=transform, target_transform=target_transform)

        self.class_num_dict = {"desert_road" : 118, "driveway" : 127, "forest_road" : 152, "highway": 175, "street": 319,
                               "bathroom" : 45,
                               "bedroom" : 52,
                               "kitchen" : 203}
        # self.class_num_dict = {"bathroom" : 45,
        #                        "bedroom" : 52,
        #                        "kitchen" : 203}
        self.class_num_list = []
        self.class_list = sorted(class_list)
        for c in self.class_list:
            self.class_num_list.append(self.class_num_dict[c])
        self.class_num_to_idx_dict = {}
        for i in range(len(self.class_num_list)):
            self.class_num_to_idx_dict[self.class_num_list[i]] = i


        # Load data with the given classes
        self.load_data()

    def __getitem__(self, index, need_PIL=False):

        target = self.targets[index]

        # Conforming to other datasets
        image = PIL.Image.fromarray(self.data[index])

        # If a transform was provided, transform the image
        if need_PIL == False:
            image = self.transform(image)
        else:
            img_transform = T.Compose([T.Resize((224,224))])
            image = img_transform(image)

        # If a target transform was provided, transform the target
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:

        # Return the length of targets list
        return len(self.targets)

    def load_data(self):
        print("loading val data")
        # Get path of extracted dataset
        image_dirs_path = os.path.join(self.root, self.base_folder, "val_256")
        with open("places365/filelist_places365/places365_val.txt", "r") as f:
            val_imgs_list = f.readlines()
        val_dict = {}
        for s in val_imgs_list:
            t_list = s.split(" ")
            val_dict[t_list[0]] = int(t_list[1])
        #Get all the images
        images_dict = {}
        for c in self.class_num_list:
            images_dict[c] = []

        for img in os.listdir(image_dirs_path):
            if val_dict[img] in self.class_num_list:
                images_dict[val_dict[img]].append(img)
        # for c in self.class_list:
        #     first_letter = c[0]
        #     for alphabet_dir in os.listdir(image_dirs_path):
        #         if alphabet_dir == first_letter:
        #             alphabet_dir_path = os.path.join(image_dirs_path, alphabet_dir)
        #             for class_dir in os.listdir(alphabet_dir_path):
        #                 if class_dir == c:
        #                     class_dir_path = os.path.join(alphabet_dir_path, class_dir)
        #                     images_dict[c] = images_dict[c] + os.listdir(class_dir_path)
        #                     break
        #             break
        # Convert the images to numpy arrays
        targets = []
        image_np_list = []
        for c, im_list in images_dict.items():
            for im_idx in range(len(im_list)):
                image_path = os.path.join(image_dirs_path, im_list[im_idx])
                image = PIL.Image.open(image_path)
                image_numpy = np.array(image)
                image_np_list.append(image_numpy)
                targets.append(self.class_num_to_idx_dict[c])
        self.targets = targets
        self.data = image_np_list
        return image_np_list, targets

class custom_subset(Dataset):
    r"""
    Subset of a dataset at specified indices.
    Arguments:
        dataset (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
        labels(sequence) : targets as required for the indices. will be the same length as indices
    """



    def __init__(self, dataset, indices, labels):
        self.dataset = torch.utils.data.Subset(dataset, indices)
        self.targets = labels.type(torch.long)
        self.indices = np.array(indices)

    def __getitem__(self, idx):
        image = self.dataset[idx][0]
        target = self.targets[idx]
        return image, target

    def __len__(self):
        return len(self.targets)


# def create_train_val_test(fullset, set_sizes):
#     np.random.seed(42)
#     length_of_dataset = len(fullset)
#
#     full_index_list = list(range(length_of_dataset))
#
#     # Pick random indices for the train set
#     train_idx = list(np.random.choice(np.array(full_index_list), size=set_sizes["train_size"], replace=False))
#     remaining_idx = list(set(full_index_list) - set(train_idx))
#
#     # Pick random indices for the validation set
#     val_idx = list(np.random.choice(np.array(remaining_idx), size=set_sizes["val_size"], replace=False))
#     remaining_idx = list(set(remaining_idx) - set(val_idx))
#
#     # Pick random indices for the test set
#     # test_idx = list(np.random.choice(np.array(remaining_idx), size=set_sizes["test_size"], replace=False))
#     # remaining_idx = list(set(remaining_idx) - set(test_idx))
#     test_idx = remaining_idx
#     # Create custom subsets that use the fullset (self, dataset, age_attributes, race_attributes, gender_attributes, indices, labels)
#     train_set = custom_subset(fullset, train_idx,torch.Tensor(fullset.targets)[train_idx])
#     val_set = custom_subset(fullset, val_idx,torch.Tensor(fullset.targets)[val_idx])
#     test_set = custom_subset(fullset, test_idx,torch.Tensor(fullset.targets)[test_idx])
#
#     # Finally, return
#     return train_set, val_set, test_set

def create_train_test(fullset):
    np.random.seed(42)
    length_of_dataset = len(fullset)

    full_index_list = list(range(length_of_dataset))

    # Pick random indices for the val set
    # print(fullset.test_indices)
    train_idx = list(set(full_index_list) - set(fullset.test_indices))
    # train_idx = list(np.random.choice(np.array(full_index_list), size=set_sizes["train_size"], replace=False))
    # remaining_idx = list(set(full_index_list) - set(train_idx))
    #
    # # Pick random indices for the test set
    # test_idx = list(np.random.choice(np.array(remaining_idx), size=set_sizes["test_size"], replace=False))
    # remaining_idx = list(set(remaining_idx) - set(test_idx))
    test_idx = fullset.test_indices
    # Create custom subsets that use the fullset (self, dataset, age_attributes, race_attributes, gender_attributes, indices, labels)
    # train_set = custom_subset(fullset, train_idx,torch.Tensor(fullset.targets)[train_idx])
    train_set = custom_subset(fullset, train_idx,torch.Tensor(fullset.targets)[train_idx])
    test_set = custom_subset(fullset, test_idx,torch.Tensor(fullset.targets)[test_idx])

    # Finally, return
    return train_set, test_set

# # #loading the full set from the train folder
# class_list = ["bathroom", "bedroom", "kitchen"]
# # class_list = ["desert_road", "driveway", "forest_road"]
# train_test_set = Places365_train_test(root= "./", class_list=class_list, transform=T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), T.RandomHorizontalFlip(), T.RandomVerticalFlip()]))
# val_set = Places365_val(root="./", class_list= class_list, transform=T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]))
# # creating train, val and test set
# # set_sizes = {
# #     "train_size": int(len(train_test_set) * 0.9),
# #     "test_size": int(len(train_test_set) * 0.1)
# #     }
#
#
# train_set, test_set = create_train_test(train_test_set)
#
# print("TRAIN IDXS", train_set.indices)
# print("TEST IDXS", test_set.indices)
#
# # To load data:
# # batch_size = 128 is the max that works.
# train_loader = DataLoader(train_set, batch_size=64,
#                         shuffle=False, num_workers=0)
# #
# # val_loader = DataLoader(val_set, batch_size=64,
# #                         shuffle=True, num_workers=0)
#
# test_loader = DataLoader(test_set, batch_size=64,
#                         shuffle=False, num_workers=0)
