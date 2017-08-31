from .utils import CLASSES, mkdir
import pandas as pd
from scipy.stats.mstats import gmean
import random
import os
from os.path import join
from PIL import Image

import torch

from .transforms import test_transform
from .models import get_pretrained_model
from .kfold_data import get_train, get_test


def random_flip(img):
    mirror = random.choice([
        None,
        Image.FLIP_LEFT_RIGHT
    ])
    if mirror is not None:
        img = img.transpose(mirror)
    return img


def test_augmentations(img, crop=(224, 224)):
    img_list = []
    crops = [
        ((img.size[0] - crop[0]) // 2, (img.size[1] - crop[1]) // 2, \
         img.size[0] - (img.size[0] - crop[0]) // 2, img.size[1] - (img.size[1] - crop[1]) // 2),
        (0, 0, crop[0], crop[1]),
        (img.size[0] - crop[0], 0, img.size[0], crop[1]),
        (img.size[0] - crop[0], img.size[1] - crop[1], img.size[0], img.size[1]),
        (0, img.size[1] - crop[1], crop[0], img.size[1])
    ]
    for _ in range(2):
        for box in crops:
            crop_img = img.crop(box)

            if img_list:
                crop_img = random_flip(img)

            img_list.append(crop_img)
        img = img.rotate(90)
    return img_list


def predict_df(model, df):
    pred_df_center = pd.DataFrame(0, columns=CLASSES, index=df.image_name)
    pred_df_random = pd.DataFrame(0, columns=CLASSES, index=df.image_name)
    transform = test_transform(crop=False)
    for i, row in df.iterrows():
        img = Image.open(row.Path).convert('RGB')
        img_list = test_augmentations(img)
        trans_img_list = [transform(im).unsqueeze(0) for im in img_list]
        input = torch.cat(trans_img_list, dim=0)
        prob_pred = model.predict(input)
        prob_pred = prob_pred.numpy()
        prob_pred_center = prob_pred[0]
        prob_pred_random = gmean(prob_pred, axis=0)
        pred_df_center.loc[row.image_name, CLASSES] = prob_pred_center
        pred_df_random.loc[row.image_name, CLASSES] = prob_pred_random
    return pred_df_center, pred_df_random


def pred_fold(kfold_df, test_df, params, fold):
    fold_save_path = join(params['save_dir'], 'fold_%d' % fold)
    model_save_path = join(fold_save_path, 'model.pth.tar')
    mkdir(fold_save_path)

    val_df = kfold_df[kfold_df.Fold == fold]

    model = get_pretrained_model(params['arch'], params['lr'])
    model.set_savestate_path(model_save_path)
    model.load_model(model.state_path)

    val_pred_df_center, val_pred_df_random = predict_df(model, val_df)
    val_pred_df_center.to_hdf(join(fold_save_path, 'val_center.h5'), 'prob')
    val_pred_df_random.to_hdf(join(fold_save_path, 'val_random.h5'), 'prob')

    test_pred_df_center, test_pred_df_random = predict_df(model, test_df)
    test_pred_df_center.to_hdf(join(fold_save_path, 'test_center.h5'), 'prob')
    test_pred_df_random.to_hdf(join(fold_save_path, 'test_random.h5'), 'prob')

    del model


def pred_kfolds(params):
    mkdir(params['save_dir'])
    kfold_df, test_df = get_train(), get_test()
    folds = sorted(kfold_df.Fold.unique())

    with open(join(params['save_dir'], 'params.txt'), 'w') as f:
        f.write(str(params))

    for fold in folds:
        print("Start predict fold %d" % fold)
        pred_fold(kfold_df, test_df, params, fold)
