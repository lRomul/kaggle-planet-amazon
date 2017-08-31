import random
import math
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from skimage import img_as_ubyte, img_as_float
from skimage import transform
from .utils import PIL2opencv, opencv2PIL

import warnings
warnings.filterwarnings("ignore")

class RandomVerticalFlip(object):

    def __call__(self, img):
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class RandomRotation(object):

    def __call__(self, img):
        cv_img = PIL2opencv(img)
        mode =  np.random.choice(['symmetric', 'reflect'])
        angle = random.randint(0, 360)
        rot_cv_img = img_as_ubyte(transform.rotate(cv_img, angle, mode=mode))
        img = opencv2PIL(rot_cv_img)
        return img


class RandomSizedCrop(object):
    """Random crop the given PIL.Image to a random size of (0.5 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.5, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = transforms.Scale(self.size, interpolation=self.interpolation)
        crop = transforms.CenterCrop(self.size)
        return crop(scale(img))

    
def train_transform():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose([
        RandomRotation(),
        transforms.Scale(256),
        RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        RandomVerticalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform


def test_transform(crop=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    if crop:
        test_transform = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        test_transform = transforms.Compose([
            transforms.Scale(224),
            transforms.ToTensor(),
            normalize,
        ])
    return test_transform