import numpy as np
import pandas as pd
import cv2
import os
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm as tq
from sklearn.metrics import f1_score

from data.data_loader import CloudDatasetClassification
from utils.utils import create_log_folder, preprocessing_to_tensor


#MODEL_NAME = "ResNet34-BCE-20E"
#MODEL_NAME = "Efficient-BCE-20E"
MODEL_NAME = "DenseNet-BCE-20E"

DATA_PATH = "./dataset"
EPOCHS = 20
BATCH_SIZE = 16
NUM_WORKERS = 0

#BACKBONE = "resnet34"
#BACKBONE = "nvidia_efficientnet_b0"
BACKBONE = "densenet121"


if __name__ == "__main__":

    logs_path = create_log_folder(MODEL_NAME)

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

    valid_dataset = CloudDatasetClassification(df=train, path=DATA_PATH, datatype='valid', img_ids=valid_ids,
        preprocessing=preprocessing_to_tensor())

    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    if "resnet" in BACKBONE:
        model = torch.hub.load('pytorch/vision:v0.10.0', BACKBONE, pretrained=True)
        model.fc = nn.Linear(512, 4)
    elif "efficientnet" in BACKBONE:
        model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', BACKBONE, pretrained=True)
        model.classifier.fc = nn.Linear(1280, 4, bias=True) 
    elif "densenet" in BACKBONE:
        model = torch.hub.load('pytorch/vision:v0.10.0', BACKBONE, pretrained=True)
        model.classifier = nn.Linear(1024, 4, bias=True)


    if train_on_gpu:
        model.cuda()

    model.load_state_dict(torch.load(os.path.join(logs_path, MODEL_NAME+"-final.pt")))
    # model.load_state_dict(torch.load(os.path.join(logs_path, MODEL_NAME+"-checkpoint.pt")))
    model.eval();

    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    counter = 0
    tr = min(len(valid_ids), 2000)
    predictions = {0:np.zeros((2, tr), dtype=np.float32),
        1:np.zeros((2, tr), dtype=np.float32),
        2:np.zeros((2, tr), dtype=np.float32),
        3:np.zeros((2, tr), dtype=np.float32)}

    for data, target in tq(valid_loader):
        if train_on_gpu:
            data = data.cuda()
        target = target.cpu().detach().numpy()
        output = model(data).cpu().detach().numpy()

        for i, batch in enumerate(output):
            batch = sigmoid(batch)
            predictions[0][0, counter] = batch[0]
            predictions[0][1, counter] = target[i][0]

            predictions[1][0, counter] = batch[1]
            predictions[1][1, counter] = target[i][1]

            predictions[2][0, counter] = batch[2]
            predictions[2][1, counter] = target[i][2]

            predictions[3][0, counter] = batch[3]
            predictions[3][1, counter] = target[i][3]

            counter += 1

        if counter >= tr:
            break


    class_params = {0:0, 1:0, 2:0, 3:0}
    for class_id in range(4):
        best_f1 = 0
        best_tr = 0
        print(f"Searching for class {class_id}")
        for t in range(0, 100, 5):
            t /= 100
            y_pred = (predictions[class_id][0,:] >= t).astype(int)
            y_true = predictions[class_id][1,:]
            curr_f1 = f1_score(y_true, y_pred)
            if curr_f1 > best_f1:
                best_f1 = curr_f1
                best_tr = t
        print("Best f1:", best_f1, "at tr:", best_tr)
        class_params[class_id] = (best_tr, best_f1)


    print("Class params", class_params)
    save_params = {str(key):str(value) for key, value in class_params.items()}
    
    with open(os.path.join(logs_path, "cls_class_params.json"), "w+") as f:
        json.dump(save_params, f)

    #with open(os.path.join(logs_path, "cls_class_params_checkpoint.json"), "w+") as f:
    #    json.dump(save_params, f)

