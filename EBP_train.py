import argparse
import os
import time

import pandas as pd
from torchvision.models import vgg16
from torchvision.models._utils import IntermediateLayerGetter
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from custom_dataset import Places365_train_test, Places365_val,create_train_test
import torch
from collections import Counter
from tqdm import tqdm
import numpy as np
# from train_models import VGG16
from ERIC_datasets import MNIST_dataloaders, GTSRB_dataloaders, PASCALanimals_dataloaders, PASCALall_dataloaders, \
    PLACES_dataloaders
from dataloaders import get_dataloader
from torch.optim import SGD, Adam, RMSprop
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorboard as tb
from algo import create_filter_data, create_group_data
from FOLDSEM.FOLDSEM import foldsem


def compute_probs(train_loader, model, params, num_classes):
    train_set = train_loader.dataset
    required_layers = {"30": "conv13"}

    # storing the activations for all images from layer 13
    model_inter = IntermediateLayerGetter(model.features, return_layers=required_layers)
    # act_maps = torch.empty((params["batch_size"],512,14,14))
    norm_tensor = torch.empty(len(train_set), 512)
    # Now constructing a 2D tensor to accumulate the activations for each class. Shape: num_class * num_filters
    D = torch.zeros((num_classes, 512))
    for batch_idx, (inputs, targets) in tqdm(enumerate(train_loader)):
        model.eval()
        # print(inputs, targets, index)
        with torch.no_grad():
            inputs = inputs.float().to(device)
            # inputs = inputs.float().to(device)
            # target_tensor = torch.cat((target_tensor, targets), dim=0)
            out = model_inter(inputs)
            inter_out = out["conv13"].cpu()
            temp = torch.abs(inter_out)
            temp_sum = torch.sum(temp, dim=(2, 3))
            # now divide the total by product of the dimensions
            batch_mat_norms = temp_sum / (inter_out.shape[2] * inter_out.shape[3])
            # act_maps = torch.cat((act_maps, out), dim = 0)
            # batch_mat_norms = torch.linalg.norm(inter_out, ord = 2, dim =(2,3))
            norm_tensor[params["batch_size"] * batch_idx:params["batch_size"] * batch_idx + batch_mat_norms.shape[
                0]] = batch_mat_norms
            # Now for all the images in the batch if the image belongs to a class then add the norm to the corresponding class
            for i in range(targets.shape[0]):
                D[targets[i]] += batch_mat_norms[i]
    # Now finding the K highest activated filters for each class. Starting with K = 100
    # Check in D for the K highest activated filters for each class and store the index in E
    K = params["K"]
    E = torch.empty((num_classes, K))
    for i in range(num_classes):
        E[i] = torch.topk(D[i], K)[1]
    # convert E to int
    E = E.int()
    # calculating probability of each filter
    P = torch.empty((num_classes, D.shape[1]))
    for i in range(D.shape[0]):
        for j in range(D.shape[1]):
            if j in E[i]:
                P[i][j] = 1
            else:
                # find the index of filter which is the Kth highest activated filter
                Kth_index = E[i][K - 1].item()
                P[i][j] = 1 - (D[i][j] / D[i][Kth_index])
    return P
def train(train_loader, val_loader, model, params, epoch_start, criterion, optimizer, writer, check_num,scheduler, device, P, checkpoints_path = None):

    required_layers = {"30": "conv13"}
    model_inter = IntermediateLayerGetter(model.features, return_layers=required_layers)

    for epoch in range(epoch_start, params["epochs"]):
        model.train()
        train_loss = 0

        for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
            # print(inputs.float().shape)
            inputs, targets = inputs.float().to(device), targets.to(device)
            out = model(inputs)
            celoss = criterion(out, targets)
            # calculating the Regularization loss

            R_loss = 0
            out_inter = model_inter(inputs)
            inter_out = out_inter["conv13"]
            temp = torch.abs(inter_out)
            temp_sum = torch.sum(temp, dim=(2, 3))
            # now divide the total by product of the dimensions
            batch_mat_norms = temp_sum / (inter_out.shape[2] * inter_out.shape[3])
            # calculating the 1 - P

            for i in range(targets.shape[0]):
                R_loss = R_loss + ((1 - P[targets[i]]) * (batch_mat_norms[i])).sum()
            loss = celoss + params["lambda"] * R_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss = train_loss + loss.item()

        if epoch % 1 == 0:
            if check_num > 5:
                check_num = 1

            # calculate val loss
            model.eval()
            with torch.no_grad():
                total_correct = 0
                num_instances = 0
                val_loss = 0
                for batch_idx, (inputs, targets) in enumerate(tqdm(val_loader)):
                    inputs, targets = inputs.float().to(device), targets.to(device)
                    ypred = model(inputs)
                    celoss = criterion(ypred, targets)
                    R_loss = 0
                    out_inter = model_inter(inputs)
                    inter_out = out_inter["conv13"]
                    temp = torch.abs(inter_out)
                    temp_sum = torch.sum(temp, dim=(2, 3))
                    # now divide the total by product of the dimensions
                    batch_mat_norms = temp_sum / (inter_out.shape[2] * inter_out.shape[3])

                    for i in range(targets.shape[0]):
                        R_loss = R_loss + ((1 - P[targets[i]]) * (batch_mat_norms[i])).sum()
                    values_val, indices_val = torch.max(ypred, 1)
                    correct_val = torch.eq(indices_val, targets).sum()
                    val_loss = val_loss + (celoss.item() + params["lambda"] * R_loss.item())
                    total_correct += correct_val
                    num_instances += len(indices_val)
            full_val_accuracy = total_correct / num_instances
            # ...log the validation loss and total train loss
            writer.add_scalar('validation accuracy',
                              full_val_accuracy,
                              epoch)

            writer.add_scalar('total train loss',
                              train_loss,
                              epoch)
            writer.add_scalar('validation loss',
                              val_loss,
                              epoch)
            print(f"val_accuracy: {full_val_accuracy}")
            if not os.path.exists(checkpoints_path):
                os.makedirs(checkpoints_path)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_accuracy': full_val_accuracy,
                "scheduler_state_dict": scheduler.state_dict()
            }, checkpoints_path + f"chkpoint{check_num}.pt")
            print("saving checkpoint!")
            print(f"chkpoint{check_num}.pt")
            check_num = check_num + 1
        scheduler.step(val_loss)
        print(f"epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}")
        # evaluate(test_loader, model)
    end_time = time.time()

device = "cuda:0"
def main():
    parser = argparse.ArgumentParser(prog="EBP_train.py", description="Run the training for CNN with EBP")
    parser.add_argument("--dataset_name", type=str, help="name of the dataset")
    # parser.add_argument("--model_check_dir", type=str, help="model checkpoint path")

    args = parser.parse_args()
    dataset_name = args.dataset_name
    params = {"batch_size": 32, "epochs": 50, "lr": 5e-7, "l2": 5e-3, "decay_factor": 0.5, "patience": 10, "lambda": 1e-3, "K" : 5}
    # acc_train_final = []
    # acc_val_final = []
    # acc_test_final = []
    # f_train_final = []
    # f_val_final = []
    # f_test_final = []
    # n_preds_final = []
    # n_rules_final = []
    # size_final = []
    #ignore
    # algo_type = "single"

    # for a given dataset, iterate through the checkpoints and pick the one that has epochs == 100
    # then load the model from that checkpoint
    for run in range(1,6):
    # run = 1

        print(f"run: {run}")
        checkpoints_path = f"model_checkpoints/EBP/model_checkpoints_{dataset_name}/{run}/"
        comment = f"""run: {run}"""
        writer = SummaryWriter(f'train_logs/EBP/{dataset_name}/{comment}')
        train_loader, val_loader, test_loader, num_classes, class_weights = get_dataloader(dataset_name, params, train_shuffle=True)
        # if run in [1]:
        #     continue
        # load the data from dataloaders
        # train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(batch_size = params["batch_size"], class_list=["desert_road", "driveway", "lake"], train_shuffle=False)
        # train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(class_list=["forest_road", "highway", "street"], train_shuffle=False)
        # train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(class_list=["bathroom", "bedroom", "kitchen", "dining_room", "living_room"], train_shuffle=False)

        # train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(class_list=["bathroom", "bedroom", "kitchen", "dining_room", "living_room", "home_office", "office", "waiting_room", "conference_room", "hotel_room"], train_shuffle=False)
        # train_loader, val_loader, test_loader, num_classes, class_weights = GTSRB_dataloaders(params["batch_size"], train_shuffle=False)
        # train_loader, val_loader, test_loader, num_classes, class_weights = MNIST_dataloaders(params["batch_size"], train_shuffle=False)
        model_checkpoint_dir_path = "model_checkpoints/vanilla/model_checkpoints_" + dataset_name
        model = vgg16(pretrained=True)
        model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)

        # chkpoint = torch.load("/home/pxp180054/projects/XAI_images/model_checkpoints/model_checkpoints_dedrf/1/chkpoint5.pt")

        # model.load_state_dict(chkpoint["model_state_dict"])
        device = "cuda:0"
        model.to(device)
        loaded_chk = False
        for run_dir in os.listdir(model_checkpoint_dir_path):
            if run_dir == str(run):
                run_dir_path = os.path.join(model_checkpoint_dir_path, run_dir)
                for chkpt in os.listdir(run_dir_path):
                    if torch.load(os.path.join(run_dir_path, chkpt))['epoch'] == 100:
                        model.load_state_dict(torch.load(os.path.join(run_dir_path, chkpt))['model_state_dict'])

                        loaded_chk = True
                        break
                break
        # model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, 5)
        # model.to(device)
        if loaded_chk == False:
            print("Could not load checkpoint so exiting")
            exit()
        print("Calculating the probability of activation for each elite filter...")
        P = compute_probs(train_loader, model, params, num_classes)
        P = P.to(device)
        # now implementing the retraining loop
        # loading the train data , val data and test data again
        # train_loader, val_loader, test_loader, num_classes, class_weights = PLACES_dataloaders(batch_size = params["batch_size"], class_list=["bathroom", "bedroom", "kitchen"])
        class_weights = class_weights.to(device)
        optimizer = Adam(model.parameters(), lr= params["lr"], weight_decay = params["l2"])
        # optimizer.load_state_dict(chkpoint["optimizer_state_dict"])
        criterion = CrossEntropyLoss(weight=class_weights)
        # criterion = CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(optimizer, factor = params["decay_factor"], patience= params["patience"])
        check_num = 1
        epoch_start = 0
        running_loss = 0
        print("Starting the retraining loop with EBP...")
        train(train_loader, val_loader, model, params, epoch_start, criterion, optimizer, writer, check_num,scheduler, device, P, checkpoints_path=checkpoints_path)


main()