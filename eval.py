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
from utils.utils import get_preprocessing, mask2rle, post_process, resize_it, dice, get_prec_rec_f1

#MODEL_NAME = "Unet-ResNet50-BCEDice-20E"
#MODEL_NAME = "Unet-EfficientB1-BCEDice-20E"
#MODEL_NAME = "FPN-ResNet50-BCEDice-20E"
MODEL_NAME = "DeepLabV3Plus-ResNet50-BCEDice-20E"
DATA_PATH = "./dataset"


LOAD_PARAMS = False

def read_class_params(model_name):
    with open(f"./logs/{model_name}/class_params-final.json") as f:
        raw_data = json.load(f)
        data = {int(key):tuple(map(float, raw_data[key].strip()[1:-1].split(','))) for key in raw_data}
    return data


def find_best_threshold_params(model, preprocessing_fn):
    train = pd.read_csv(f"{DATA_PATH}/train.csv")
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])

    id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().\
        reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})
    id_mask_count = id_mask_count.sort_values('img_id')
    
    _, valid_ids = train_test_split(id_mask_count['img_id'].values, random_state=42, stratify=id_mask_count['count'], test_size=0.1)
    valid_dataset = CloudDataset(df=train, path=DATA_PATH, datatype='valid', img_ids=valid_ids,
        preprocessing=get_preprocessing(preprocessing_fn))
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=0)

    valid_masks = []
    count = 0
    tr = min(len(valid_ids)*4, 2000)
    probabilities = np.zeros((tr, 350, 525), dtype = np.float32)
    for data, target in tq(valid_loader):
        if torch.cuda.is_available():
            data = data.cuda()
        target = target.cpu().detach().numpy()
        outpu = model(data).cpu().detach().numpy()
        for p in range(data.shape[0]):
            output, mask = outpu[p], target[p]
            for m in mask:
                valid_masks.append(resize_it(m))
            for probability in output:
                probabilities[count, :, :] = resize_it(probability)
                count += 1
            if count >= tr - 1:
                break
        if count >= tr - 1:
            break

    sigmoid = lambda x: 1 / (1 + np.exp(-x))

    class_params = {}
    for class_id in range(4):
        attempts = []
        print(f"Searching for class {class_id}")
        for t in range(0, 100, 5):
            t /= 100
            for ms in [0, 100, 1200, 5000, 10000, 30000]:
                masks, d = [], []
                for i in range(class_id, len(probabilities), 4):
                    probability = probabilities[i]
                    predict, num_predict = post_process(sigmoid(probability), t, ms)
                    masks.append(predict)
                for i, j in zip(masks, valid_masks[class_id::4]):
                    if (i.sum() == 0) & (j.sum() == 0):
                        d.append(1)
                    else:
                        d.append(dice(i, j))
                attempts.append((t, ms, np.mean(d)))

        attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])
        attempts_df = attempts_df.sort_values('dice', ascending=False)
        best_threshold = attempts_df['threshold'].values[0]
        best_size = attempts_df['size'].values[0]
        class_params[class_id] = (best_threshold, best_size)
    
    return class_params

def eval_validation(model, params, preprocessing_fn):
    train = pd.read_csv(f"{DATA_PATH}/train.csv")
    train['label'] = train['Image_Label'].apply(lambda x: x.split('_')[1])
    train['im_id'] = train['Image_Label'].apply(lambda x: x.split('_')[0])

    id_mask_count = train.loc[train['EncodedPixels'].isnull() == False, 'Image_Label'].apply(lambda x: x.split('_')[0]).value_counts().\
        reset_index().rename(columns={'index': 'img_id', 'Image_Label': 'count'})
    _, valid_ids = train_test_split(id_mask_count['img_id'].values, random_state=42, stratify=id_mask_count['count'], test_size=0.1)
    valid_dataset = CloudDataset(df=train, path=DATA_PATH, datatype='valid', img_ids=valid_ids,
        preprocessing=get_preprocessing(preprocessing_fn))
    valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False, num_workers=0)

    precision = [0,0,0,0]
    recall = [0,0,0,0]
    f1 = [0,0,0,0]

    image_id = 0
    counter = 0
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    for data, target in tq(valid_loader):
        if torch.cuda.is_available():
            data = data.cuda()
        output = model(data)
        target = target.cpu().detach().numpy()
        for i, batch in enumerate(output):
            masks = target[i]
            for j, probability in enumerate(batch):
                probability = probability.cpu().detach().numpy()
                ground_truth = cv2.resize(masks[j], dsize=(525, 350), interpolation=cv2.INTER_NEAREST)
                if probability.shape != (350, 525):
                    probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                predict, num_predict = post_process(sigmoid(probability), params[image_id % 4][0], params[image_id % 4][1])
                curr_prec, curr_rec, curr_f1 = get_prec_rec_f1(ground_truth=ground_truth, prediction=predict)
                precision[image_id % 4] += curr_prec
                recall[image_id % 4] += curr_rec
                f1[image_id % 4] += curr_f1
                counter += 1
                image_id += 1


    precision = [i/counter for i in precision]
    recall = [i/counter for i in recall]
    f1 = [i/counter for i in f1]
    print("Precision", precision)
    print("Recall", recall)
    print("F1", f1)



if __name__ == "__main__":

    train_on_gpu = torch.cuda.is_available()
    print(f"Use GPU: {train_on_gpu}")

    logs_path = os.path.join("./logs", MODEL_NAME)

    ENCODER = 'resnet50'
    #ENCODER = 'efficientnet-b1'
    ENCODER_WEIGHTS = 'imagenet'
    DEVICE = 'cuda'

    ACTIVATION = None
    
    #model = smp.Unet(
    #    encoder_name=ENCODER, 
    #    encoder_weights=ENCODER_WEIGHTS, 
    #    classes=4, 
    #    activation=ACTIVATION,
    #)

    #model = smp.FPN(
    #    encoder_name=ENCODER, 
    #    encoder_weights=ENCODER_WEIGHTS, 
    #    classes=4, 
    #    activation=ACTIVATION,
    #)

    model = smp.DeepLabV3Plus(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=4,
        activation=ACTIVATION,
    )

    preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    
    if train_on_gpu:
        model.cuda()
    
    # load best model
    model.load_state_dict(torch.load(os.path.join(logs_path, MODEL_NAME+"-final.pt")))
    # model.load_state_dict(torch.load(os.path.join(logs_path, MODEL_NAME+"-checkpoint.pt")))
    model.eval();

    if LOAD_PARAMS:
        class_params = read_class_params(MODEL_NAME)
    else:
        class_params = find_best_threshold_params(model, preprocessing_fn)
        print("Class params", class_params)
        save_params = {str(key):str(value) for key, value in class_params.items()}

        with open(os.path.join(logs_path, "class_params-final.json"), "w+") as f:
            json.dump(save_params, f)


    # eval_validation(model, class_params, preprocessing_fn)

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

    subm = pd.read_csv(f"{DATA_PATH}/sample_submission.csv")
    pathlist = [f"{DATA_PATH}/test_images/" + i.split("_")[0] for i in subm['Image_Label']]

    encoded_pixels = []
    image_id = 0
    cou = 0
    np_saved = 0

    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    counter = 0
    for data, target in tq(test_loader):
        if train_on_gpu:
            data = data.cuda()
        output = model(data)
        for i, batch in enumerate(output):
            for probability in batch:
                probability = probability.cpu().detach().numpy()
                counter += 1
                if probability.shape != (350, 525):
                    probability = cv2.resize(probability, dsize=(525, 350), interpolation=cv2.INTER_LINEAR)
                predict, num_predict = post_process(sigmoid(probability), class_params[image_id % 4][0], class_params[image_id % 4][1])
                # cv2.imwrite(f"temp/{counter}.png",predict*255)
                if num_predict == 0:
                    encoded_pixels.append('')
                else:
                    r = mask2rle(predict)
                    encoded_pixels.append(r)
                    np_saved += np.sum(predict > 0)
                cou += 1
                image_id += 1

    print(f"number of pixel saved {np_saved}")

    sub['EncodedPixels'] = encoded_pixels
    sub.to_csv(os.path.join(logs_path, 'submission-final.csv'), columns=['Image_Label', 'EncodedPixels'], index=False)
    # sub.to_csv(os.path.join(logs_path, 'submission-checkpoint.csv'), columns=['Image_Label', 'EncodedPixels'], index=False)
