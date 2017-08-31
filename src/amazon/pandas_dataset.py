import torch.utils.data as data

import pandas as pd
from PIL import Image
import os
import os.path
from .utils import str2arr


def loader(path):
    return Image.open(path).convert('RGB')


class PandasDataset(data.Dataset):

    def __init__(self, df, transform=None):
        self.transform = transform
        self.df = df[["tags", "Path"]]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        target = str2arr(row.tags)
        img = loader(row.Path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.df)