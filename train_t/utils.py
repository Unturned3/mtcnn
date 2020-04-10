"""

"""


# Built-in
import os
from glob import glob

# Libs
import numpy as np
from tqdm import tqdm
from skimage import io, color
import skimage.transform as sktrsfm
from sklearn.metrics import precision_score, recall_score

# Pytorch
import torch
import torch.nn.functional as F
from torch.utils import data

import torchvision
from torchvision import transforms

# Own modules


class UTKDataLoader(data.Dataset):
    def __init__(self, img, lbl, tsfm=None):
        self.img = img
        self.lbl = lbl
        self.transforms = tsfm

    def __len__(self):
        return len(self.img)

    def __getitem__(self, index):
        img = self.img[index]
        lbl = self.lbl[index]
        if self.transforms:
            img = self.transforms(img)
        return img, torch.tensor(lbl)


def get_images(parent_path, age_thresh=(6, 18, 25, 35, 60), valid_percent=0.2, resize_shape=(32, 32)):
    img_files = sorted(glob(os.path.join(parent_path, '*.jpg')))[:500]
    imgs, ages = [], []
    age_thresh = [-1, *age_thresh, 200]
    for img_file in tqdm(img_files):
        img = io.imread(img_file)
        if img.shape[-1] == 1:
            img = color.grey2rgb(img)
        age = int(os.path.splitext(os.path.basename(img_file))[0].split('_')[0])
        img = sktrsfm.resize(img, resize_shape, anti_aliasing=True, preserve_range=True)[..., :3]
        
        # convert to PIL image, so code can use MTCNN to do processing stuff
        imgs.append(torchvision.transforms.ToPILImage()(img.astype(np.uint8)))

        for cnt, (lb, ub) in enumerate(zip(age_thresh[:-1], age_thresh[1:])):
            if lb < age <= ub:
                ages.append(cnt)
                break
    rand_idx = np.random.permutation(np.arange(len(img_files)))
    imgs = [imgs[a] for a in rand_idx]
    ages = [ages[a] for a in rand_idx]
    valid_num = int(np.floor(len(imgs) * valid_percent))
    return imgs[valid_num:], ages[valid_num:], imgs[:valid_num], ages[:valid_num]


def f1_score(truth, pred, eval_class):
    def binarize(l, ref):
        l_new = np.zeros_like(l)
        for cnt in range(len(l)):
            if l[cnt] == ref:
                l_new[cnt] = 1
            else:
                l_new[cnt] = 0
        return l_new

    truth = binarize(truth, eval_class)
    pred = binarize(pred, eval_class)
    precision = precision_score(truth, pred)
    recall = recall_score(truth, pred)
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

