import numpy as np
import os
import cv2
import json
import torch
import pandas as pd
from tqdm.auto import tqdm as tq
from sklearn.model_selection import train_test_split
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from data.data_loader import CloudDataset
from utils.utils import create_log_folder, get_preprocessing, mask2rle, post_process, resize_it, dice


MODEL_NAME_SEG = "Unet-ResNet50-BCEDice-20E"
#MODEL_NAME_CLS = "ResNet34-BCE-20E"
#MODEL_NAME_CLS = "Efficient-BCE-20E"
MODEL_NAME_CLS = "DenseNet-BCE-20E"

#MODEL_NAME = "Seg-ResNet-CLS-ResNet"
#MODEL_NAME = "Seg-ResNet-CLS-EfficientNet"
MODEL_NAME = "Seg-ResNet-CLS-DenseNet"

#BACKBONE = "resnet34"
#BACKBONE = "nvidia_efficientnet_b0"
BACKBONE = "densenet121"

DATA_PATH = "./dataset"


def read_class_params(model_name):
    with open(f"./logs/{model_name}/class_params.json") as f:
        raw_data = json.load(f)
        data = {int(key):tuple(map(float, raw_data[key].strip()[1:-1].split(','))) for key in raw_data}
    return data

def read_class_params_cls(model_name):
    with open(f"./logs/{model_name}/cls_class_params.json") as f:
        raw_data = json.load(f)
        data = {int(key):tuple(map(float, raw_data[key].strip()[1:-1].split(','))) for key in raw_data}
    return data


if __name__ == "__main__":
    logs_path = create_log_folder(MODEL_NAME)

    train_on_gpu = torch.cuda.is_available()
    print(f"Use GPU: {train_on_gpu}")

    ENCODER = 'resnet50'
    ENCODER_WEIGHTS = 'imagenet'
    DEVICE = 'cuda'

    ACTIVATION = None
    model_seg = smp.Unet(
       encoder_name=ENCODER, 
       encoder_weights=ENCODER_WEIGHTS, 
       classes=4, 
       activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    
    if train_on_gpu:
        model_seg.cuda()
    
    # load best model
    model_seg.load_state_dict(torch.load(f"./logs/{MODEL_NAME_SEG}/{MODEL_NAME_SEG}-final.pt"))
    model_seg.eval()

    class_params_seg = read_class_params(MODEL_NAME_SEG)

    if "resnet" in BACKBONE:
        model_cls = torch.hub.load('pytorch/vision:v0.10.0', BACKBONE, pretrained=True)
        model_cls.fc = nn.Linear(512, 4)
    elif "efficientnet" in BACKBONE:
        model_cls = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', BACKBONE, pretrained=True)
        model_cls.classifier.fc = nn.Linear(1280, 4, bias=True) 
    elif "densenet" in BACKBONE:
        model_cls = torch.hub.load('pytorch/vision:v0.10.0', BACKBONE, pretrained=True)
        model_cls.classifier = nn.Linear(1024, 4, bias=True)

    if train_on_gpu:
        model_cls.cuda()

    model_cls.load_state_dict(torch.load(f"./logs/{MODEL_NAME_CLS}/{MODEL_NAME_CLS}-final.pt"))
    model_cls.eval()

    class_params_cls = read_class_params_cls(MODEL_NAME_CLS)


    sub = pd.read_csv(f"{DATA_PATH}/sample_submission.csv")
    sub["label"] = sub["Image_Label"].apply(lambda x: x.split("_")[1])
    sub["im_id"] = sub["Image_Label"].apply(lambda x: x.split("_")[0])
    test_ids = sub["Image_Label"].apply(lambda x: x.split("_")[0]).drop_duplicates().values

    test_dataset = CloudDataset(df=sub,
                            datatype='test', 
                            img_ids=test_ids,
                            preprocessing=get_preprocessing(preprocessing_fn))
    test_loader = DataLoader(test_dataset, batch_size=4,
                            shuffle=False, num_workers=2)

    encoded_pixels = []
    image_id = 0
    cou = 0
    np_saved = 0

    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    for data, target in tq(test_loader):
        if train_on_gpu:
            data = data.cuda()
        output_seg = model_seg(data)
        output_cls = model_cls(data)

        for i, batch in enumerate(output_seg):
            cls_vector = sigmoid(output_cls[i].cpu().detach().numpy())

            for j, probability in enumerate(batch):
                probability = probability.cpu().detach().numpy()

                if probability.shape != (350, 525):
                    probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                predict, num_predict = post_process(sigmoid(probability), class_params_seg[image_id % 4][0], class_params_seg[image_id % 4][1])
                # compare current prediction vs classifier threshold
                cls_prediction = cls_vector[j] >= class_params_cls[j][0]

                if not cls_prediction:
                    encoded_pixels.append('')
                else:
                    r = mask2rle(predict)
                    encoded_pixels.append(r)
                    np_saved += np.sum(predict > 0)
                cou += 1
                image_id += 1

    print(f"number of pixel saved {np_saved}")

    sub['EncodedPixels'] = encoded_pixels
    sub.to_csv(os.path.join(logs_path, 'submission.csv'), columns=['Image_Label', 'EncodedPixels'], index=False)
