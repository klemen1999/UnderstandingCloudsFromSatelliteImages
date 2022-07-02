import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm as tq

from data.data_loader import CloudDatasetClassification
from utils.utils import create_log_folder, preprocessing_to_tensor


MODEL_NAME = "ResNet34-BCE-20E"
#MODEL_NAME = "Efficient-BCE-20E"
#MODEL_NAME = "DenseNet-BCE-20E"

DATA_PATH = "./dataset"
EPOCHS = 20
BATCH_SIZE = 16
NUM_WORKERS = 0


BACKBONE = "resnet34"
#BACKBONE = "nvidia_efficientnet_b0"
#BACKBONE = "densenet121"

LR = 1e-4


if __name__ == "__main__":

    logs_path = create_log_folder(MODEL_NAME)

    losses = {
        "bce": nn.BCEWithLogitsLoss()
    }
    LOSS = "bce"

    train_on_gpu = torch.cuda.is_available()
    print(f"Train on GPU: {train_on_gpu}")

    train = pd.read_csv(f"{DATA_PATH}/train.csv")
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])
    train["one_hot"] = (train["EncodedPixels"].isnull() == False).astype(int)

    id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().\
        reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})
    id_mask_count = id_mask_count.sort_values('img_id')
    train_ids, valid_ids = train_test_split(id_mask_count['img_id'].values, random_state=42, stratify=id_mask_count['count'], test_size=0.2)

    train_dataset = CloudDatasetClassification(df=train, path=DATA_PATH, datatype='train', img_ids=train_ids,
        preprocessing=preprocessing_to_tensor())
    valid_dataset = CloudDatasetClassification(df=train, path=DATA_PATH, datatype='valid', img_ids=valid_ids,
        preprocessing=preprocessing_to_tensor())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)


    if "resnet" in BACKBONE: 
        model = torch.hub.load('pytorch/vision:v0.10.0', BACKBONE, pretrained=True)
        model.fc = nn.Linear(512, 4, bias=True)
    elif "efficientnet" in BACKBONE:
        model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', BACKBONE, pretrained=True)
        model.classifier.fc = nn.Linear(1280, 4, bias=True) 
    elif "densenet" in BACKBONE:
        model = torch.hub.load('pytorch/vision:v0.10.0', BACKBONE, pretrained=True)
        model.classifier = nn.Linear(1024, 4, bias=True)
    
    if train_on_gpu:
        model.cuda()

    # model, criterion, optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=5)
    criterion = losses[LOSS]

    # number of epochs to train the model
    train_loss_list = []
    valid_loss_list = []
    lr_rate_list = []
    valid_loss_min = np.Inf # track change in validation loss

    sigmoid = nn.Sigmoid()

    for epoch in range(EPOCHS):
        # keep track of training and validation loss
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        bar = tq(train_loader, postfix={"train_loss":0.0})
        for data, target in bar:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
            bar.set_postfix(ordered_dict={"train_loss":loss.item()})

        model.eval()
        del data, target
        with torch.no_grad():
            bar = tq(valid_loader, postfix={"valid_loss":0.0})
            for data, target in bar:
                # move tensors to GPU if CUDA is available
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                # forward pass: compute predicted outputs by passing inputs to the model
                output = model(data)
                # calculate the batch loss
                loss = criterion(output, target)
                # update average validation loss
                valid_loss += loss.item()*data.size(0)
                bar.set_postfix(ordered_dict={"valid_loss":loss.item()})

        # calculate average losses
        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)
        train_loss_list.append(train_loss)
        valid_loss_list.append(valid_loss)
        lr_rate_list.append([param_group['lr'] for param_group in optimizer.param_groups])

        # print training/validation statistics
        print('Epoch: {}  Training Loss: {:.6f}  Validation Loss: {:.6f}'.format(
            epoch, train_loss, valid_loss))

        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            torch.save(model.state_dict(), os.path.join(logs_path, f'{MODEL_NAME}-checkpoint.pt'))
            valid_loss_min = valid_loss

        scheduler.step(valid_loss)

    torch.save(model.state_dict(), os.path.join(logs_path, f'{MODEL_NAME}-final.pt'))


    metadata = {
        "model": MODEL_NAME,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "backbone": BACKBONE,
        "lr": LR,
        "loss": LOSS,
        "train_loss_list": train_loss_list,
        "valid_loss_list": valid_loss_list,
        "lr_rate_list": lr_rate_list,
    }

    with open(os.path.join(logs_path, "metadata.json"), "w+") as f:
        json.dump(metadata, f)

    plt.figure(figsize=(10,10))
    plt.plot(train_loss_list,  marker='o', label="Training Loss")
    plt.plot(valid_loss_list,  marker='o', label="Validation Loss")
    plt.ylabel('loss', fontsize=22)
    plt.legend()
    plt.savefig(os.path.join(logs_path, "loss.png"))

