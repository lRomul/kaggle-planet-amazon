from .utils import mkdir
import numpy as np
import os
from os.path import join

import torch

from .pandas_dataset import PandasDataset
from .transforms import train_transform, test_transform
from .models import get_pretrained_model
from .kfold_data import get_train


def save_history(history, save_dir):
    save_info = dict()
    save_info['history'] = history
    save_info['best_val_loss'] = np.min(history['val'])
    save_info['best_epoch'] = np.argmin(history['val'])
    save_path = join(save_dir, 'history.txt')
    with open(save_path, 'w') as f:
        f.write(str(save_info))


def train_fold(kfold_df, params, fold):
    fold_save_path = join(params['save_dir'], 'fold_%d'%fold)
    model_save_path = join(fold_save_path, 'model.pth.tar')
    mkdir(fold_save_path)
    train_df = kfold_df[kfold_df.Fold != fold]
    val_df = kfold_df[kfold_df.Fold == fold]

    train_loader = torch.utils.data.DataLoader(
        PandasDataset(train_df, train_transform()),
        batch_size=params['batch_size'], shuffle=True,
        num_workers=2, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        PandasDataset(val_df, test_transform()),
        batch_size=params['batch_size'], shuffle=False,
        num_workers=2, pin_memory=True)

    model = get_pretrained_model(params['arch'], params['lr'])
    model.set_savestate_path(model_save_path) 
    history = model.fit(train_loader, val_loader, n_epoch=params['n_epoch'], lr=params['lr'])
    model.load_model(model.state_path)
    model.validate(val_loader)
    save_history(history, fold_save_path)
    del model, train_loader, val_loader


def train_kfolds(params):
    mkdir(params['save_dir'])
    kfold_df = get_train()
    folds = sorted(kfold_df.Fold.unique())

    with open(join(params['save_dir'], 'params.txt'), 'w') as f:
        f.write(str(params))
    
    for fold in folds:
        print("Start train fold %d" % fold)
        train_fold(kfold_df, params, fold)
