from os.path import join
import numpy as np
import pandas as pd

from .utils import TRAIN_CSV, TRAIN_DATA_DIR, IMG_EXTN, SAMPLE_PATH, TEST_DATA_DIR, FOLDS_PATH


def get_kfold_idx():
    return np.load(FOLDS_PATH)
        

def get_kfold_df(data_df):
    data_df['Fold'] = -1
    kfold_idx = get_kfold_idx()
    for i, idx in enumerate(kfold_idx):
        data_df.loc[idx, 'Fold'] = i
    return data_df


def get_train():
    train_df = pd.read_csv(TRAIN_CSV)
    train_df['Path'] = train_df.image_name.map(lambda x: join(TRAIN_DATA_DIR, x+IMG_EXTN))
    train_df = get_kfold_df(train_df)
    return train_df

def get_test():
    test_df = pd.read_csv(SAMPLE_PATH)
    test_df['Path'] = test_df.image_name.map(lambda x: join(TEST_DATA_DIR, x+IMG_EXTN))
    return test_df