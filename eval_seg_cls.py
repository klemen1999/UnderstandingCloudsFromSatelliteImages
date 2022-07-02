import numpy as np
import os
import cv2
import json
import torch
import pandas as pd
from tqdm.auto import tqdm as tq
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader

from data.data_loader import CloudDataset
from utils.utils import create_log_folder, get_preprocessing, mask2rle, post_process, resize_it, dice


MODEL_NAME = "Ensamble-SEG-CLS"
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

def get_models_cls():
    models_cls_info = [
        {"name": "ResNet34-BCE-20E", "backbone":"resnet34"},
        {"name": "Efficient-BCE-20E", "backbone":"nvidia_efficientnet_b0"},
        {"name": "DenseNet-BCE-20E", "backbone":"densenet121"}
    ]

    models_cls = []
    for model_info in models_cls_info:
        BACKBONE = model_info["backbone"]
        if "resnet" in BACKBONE:
            model_cls = torch.hub.load('pytorch/vision:v0.10.0', BACKBONE, pretrained=True)
            model_cls.fc = nn.Linear(512, 4)
        elif "efficientnet" in BACKBONE:
            model_cls = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', BACKBONE, pretrained=True)
            model_cls.classifier.fc = nn.Linear(1280, 4, bias=True) 
        elif "densenet" in BACKBONE:
            model_cls = torch.hub.load('pytorch/vision:v0.10.0', BACKBONE, pretrained=True)
            model_cls.classifier = nn.Linear(1024, 4, bias=True)

        if torch.cuda.is_available():
            model_cls.cuda()

        model_cls.load_state_dict(torch.load(f"./logs/{model_info['name']}/{model_info['name']}-final.pt"))
        model_cls.eval()

        class_params_cls = read_class_params_cls(model_info["name"])
        class_params_list = class_params_cls.values()
        class_params_list = np.array([i[0] for i in class_params_list])
        models_cls.append({"model":model_cls, "params":class_params_list})

    return models_cls


if __name__ == "__main__":
    logs_path = create_log_folder(MODEL_NAME)

    train_on_gpu = torch.cuda.is_available()
    print(f"Use GPU: {train_on_gpu}")

    sub = pd.read_csv(f"{DATA_PATH}/sample_submission.csv")
    sub["label"] = sub["Image_Label"].apply(lambda x: x.split("_")[1])
    sub["im_id"] = sub["Image_Label"].apply(lambda x: x.split("_")[0])
    test_ids = sub["Image_Label"].apply(lambda x: x.split("_")[0]).drop_duplicates().values

    models = [
        {"name":"Unet-ResNet50-BCEDice-20E", "encoder": "resnet50"},
        {"name":"FPN-ResNet50-BCEDice-20E", "encoder": "resnet50"},
        {"name":"DeepLabV3Plus-ResNet50-BCEDice-20E", "encoder": "resnet50"},
    ]

    models_cls = get_models_cls()
    cls_params_mat = np.vstack([model["params"]for model in models_cls])

    predictions = {}
    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    for model_info in models:
        model_name = model_info["name"]
        print(f"Predicting with model {model_name}")
        if model_name.startswith("Unet"):
            model = smp.Unet(
                encoder_name=model_info["encoder"], 
                encoder_weights="imagenet", 
                classes=4, 
                activation=None,
            )
        elif model_name.startswith("FPN"):
            model = smp.FPN(
                encoder_name=model_info["encoder"], 
                encoder_weights="imagenet", 
                classes=4, 
                activation=None,
            )
        elif model_name.startswith("DeepLab"):
            model = smp.DeepLabV3Plus(
                encoder_name=model_info["encoder"],
                encoder_weights="imagenet",
                classes=4,
                activation=None,
            )
        else:
            raise Exception("Invalid model type")

        if train_on_gpu:
            model.cuda()
        model.load_state_dict(torch.load(f"./logs/{model_name}/{model_name}-final.pt"))
        model.eval()

        preprocessing_fn = smp.encoders.get_preprocessing_fn(model_info["encoder"], "imagenet")

        class_params = read_class_params(model_name)

        test_dataset = CloudDataset(df=sub,
                        datatype='test', 
                        img_ids=test_ids,
                        preprocessing=get_preprocessing(preprocessing_fn))
        test_loader = DataLoader(test_dataset, batch_size=4,
                                shuffle=False, num_workers=2)

        image_id = 0

        for data, target in tq(test_loader):
            if train_on_gpu:
                data = data.cuda()
            output_seg = model(data)

            outputs_cls = []
            for model_cls in models_cls:
                output_cls = model_cls["model"](data)
                outputs_cls.append(output_cls)

            for i, batch in enumerate(output_seg):
                cls_vectors = np.vstack([sigmoid(output_cls[i].cpu().detach().numpy()) for output_cls in outputs_cls])
                cls_predictions = (cls_vectors >= cls_params_mat).astype(int)

                for j, probability in enumerate(batch):
                    probability = probability.cpu().detach().numpy()
                    if probability.shape != (350, 525):
                        probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                    predict, num_predict = post_process(sigmoid(probability), class_params[image_id % 4][0], class_params[image_id % 4][1])
                    cls_prediction = np.sum(cls_predictions[:,j]) >= 2
                    # cls_prediction = np.sum(cls_predictions[:,j]) >= 1

                    if image_id not in predictions:
                        predictions[image_id] = []

                    if not cls_prediction:
                        predictions[image_id].append(np.zeros((350, 525)))
                    else:
                        predictions[image_id].append(predict)
                    image_id += 1

    encoded_pixels = []
    np_saved = 0
    for i, image_id in enumerate(predictions):
        images = predictions[image_id]
        images_sum = np.sum(images, axis=0)
        final_img = (images_sum >= 2).astype(int)
        r = mask2rle(final_img)
        encoded_pixels.append(r)
        np_saved += np.sum(final_img > 0)

    print("Saved", np_saved)
    sub['EncodedPixels'] = encoded_pixels
    sub.to_csv(os.path.join(logs_path, 'submission.csv'), columns=['Image_Label', 'EncodedPixels'], index=False)

