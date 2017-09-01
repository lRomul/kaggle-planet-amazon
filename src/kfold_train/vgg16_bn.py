import sys
sys.path.append('..')

import os
from amazon.utils import DATA_DIR
from amazon.train_kfolds import train_kfolds
from amazon.pred_kfolds import pred_kfolds

save_dir = os.path.join(DATA_DIR, 'kfold_train/vgg16_bn')

params = {
    'save_dir': save_dir,
    'arch': "vgg16_bn",
    'batch_size': 32,
    'lr': 0.01,
    'n_epoch': 50
}

if __name__ == '__main__':
    train_kfolds(params)
    pred_kfolds(params)
