import sys
sys.path.append('..')

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from amazon.utils import DATA_DIR
from amazon.train_kfolds import train_kfolds
from amazon.pred_kfolds import pred_kfolds

save_dir = os.path.join(DATA_DIR, 'kfold_train/vgg19_001')

params = {
    'save_dir': save_dir,
    'arch': "vgg19",
    'batch_size': 32,
    'lr': 0.01,
    'n_epoch': 40
}

if __name__ == '__main__':
    train_kfolds(params)
    pred_kfolds(params)