
import os
from os.path import join
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import fbeta_score

import cv2
from PIL import Image

DATA_DIR = '/workdir/data/'
TRAIN_DATA_DIR = join(DATA_DIR, 'train-jpg')
TEST_DATA_DIR = join(DATA_DIR, 'test')
SAMPLE_PATH = join(DATA_DIR, 'sample_submission_v2.csv')
TRAIN_CSV = join(DATA_DIR, 'train_v2.csv')
FOLDS_PATH = join(DATA_DIR, '10_folds.npy')
IMG_EXTN = '.jpg'
CLASSES = [
    'agriculture',
    'artisinal_mine',
    'bare_ground',
    'blooming',
    'blow_down',
    'clear',
    'cloudy',
    'conventional_mine',
    'cultivation',
    'habitation',
    'haze',
    'partly_cloudy',
    'primary',
    'road',
    'selective_logging',
    'slash_burn',
    'water'
]

ind2cls = {i:cls for i, cls in enumerate(CLASSES)}
cls2ind = {cls:i for i, cls in ind2cls.items()} 


def opencv2PIL(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    return image


def PIL2opencv(image):
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def str2arr(string):
    arr = np.zeros(len(CLASSES), np.float32)
    for cls in string.split(' '):
        ind = cls2ind[cls]
        arr[ind] = 1
    return arr


def arr2str(arr):
    cls_lst = []
    for cls, val in zip(CLASSES, arr):
        if val:
            cls_lst.append(cls)
    return " ".join(cls_lst)


def mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

        
def show_bgr(img):
    plt.figure(figsize=(7, 7))
    plt.imshow(img[:, :, (2, 1, 0)])


def f2_score(y_true, y_pred):
    return fbeta_score(y_true, y_pred, beta=2, average='samples')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count