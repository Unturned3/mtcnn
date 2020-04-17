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


class UTK_dataset(data.Dataset):
    def __init__(self, img, age, bbox, prob, landmarks, trsfm=None):
        self.img, self.age = img, age
        self.bbox, self.prob, self.landmarks = bbox, prob, landmarks
        self.transforms = trsfm
    
    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, idx):
        img, age = self.img[idx], self.age[idx]
        bbox, prob, landmarks = self.bbox[idx], self.prob[idx], self.landmarks[idx]
        if self.transforms:
            img = self.transforms(img)
        #return img, torch.tensor(age), torch.tensor(bbox), torch.tensor(prob), torch.tensor(landmarks)
        return img, torch.tensor(age), torch.tensor(bbox), torch.tensor(landmarks)


def get_images(parent_path, age_thresh=(6, 18, 25, 35, 60), valid_percent=0.1, resize_shape=(32, 32)):
    
    # only *.chip.jpg is supported; No facial bbox, landmark, and probs provided for other images.
    img_files = sorted(glob(os.path.join(parent_path, '*.chip.jpg')))[:1000]
    
    imgs, ages, fn, bbox, prob, landmarks = [], [], [], [], [], [] # fn: file names
    
    ign_list = [] # list of files that were ignored
    
    age_thresh = [-1, *age_thresh, 200]
    
    for img_file in tqdm(img_files):
        
        # read face bbox, prob, and landmarks for each image
        t_bbox = np.load(img_file+".bbox.npy")
        t_prob = np.load(img_file+".prob.npy")
        t_landmarks = np.load(img_file+".landmarks.npy")
        
        if np.array_equal(t_bbox, [0,0,0,0]):
            ign_list.append(os.path.basename(img_file))
            continue # ignore this image; do not load
        
        bbox.append(t_bbox[0])
        prob.append(t_prob)
        if t_prob.shape == np.shape([1]):
            print("##### ERROR #####")
        
        landmarks.append(t_landmarks)
        
        img = io.imread(img_file)
        if img.shape[-1] == 1:
            img = color.grey2rgb(img)
        
        age = int(os.path.splitext(os.path.basename(img_file))[0].split('_')[0])
        img = sktrsfm.resize(img, resize_shape, anti_aliasing=True, preserve_range=True)[..., :3]
        
        imgs.append(torchvision.transforms.ToPILImage()(img.astype(np.uint8)))
        fn.append(img_file)

        for cnt, (lb, ub) in enumerate(zip(age_thresh[:-1], age_thresh[1:])):
            if lb < age <= ub:
                ages.append(cnt)
                break
    
    # temporarily disable randomization for dataset
    """
    rand_idx = np.random.permutation(np.arange(len(img_files)))
    imgs = [imgs[a] for a in rand_idx]
    ages = [ages[a] for a in rand_idx]
    fn = [fn[a] for a in rand_idx]
    bbox = [bbox[a] for a in rand_idx]
    prob = [prob[a] for a in rand_idx]
    landmarks = [landmarks[a] for a in rand_idx]
    """
    
    # debug printout
    print("Ignored images: ")
    for s in ign_list:
        print(s)
    
    vn = int(np.floor(len(imgs) * valid_percent))
    return imgs[vn:], ages[vn:], fn[vn:], bbox[vn:], prob[vn:], landmarks[vn:], imgs[:vn], ages[:vn], fn[:vn], bbox[:vn], prob[:vn], landmarks[:vn]
    # return imgs, ages, imgs, ages, fn


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

