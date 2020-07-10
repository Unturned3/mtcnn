import os
from glob import glob
import numpy as np
from tqdm import tqdm

from PIL import Image

# skimage
from skimage import io, color
import skimage.transform as sktrsfm
from sklearn.metrics import precision_score, recall_score

# Pytorch
import torch
import torch.nn.functional as F
from torch.utils import data
import torchvision
from torchvision import transforms

# for parsing XML image annotations
import xml.etree.ElementTree as ET

def xml_to_numpy(anno_path):
    
    anno_names = sorted(glob(os.path.join(anno_path, '*.xml')))
    
    for ca in tqdm(anno_names):
        
        root = ET.parse(ca).getroot().find('object').find('bndbox')
        
        bb = np.zeros(4)
        bb[0] = int(root.find('xmin').text)
        bb[1] = int(root.find('ymin').text)
        bb[2] = int(root.find('xmax').text)
        bb[3] = int(root.find('ymax').text)
        bb = bb.astype('int64')
        np.save(os.path.splitext(ca)[0]+'.npy', bb)
    
    print('done!')

class CIVDS(data.Dataset):  # CIVDS: Children in Vehicles Dataset
    
    # img: children image
    # prob: whether the image contains a child or not
    # bbox: bounding box of the child in the image
    def __init__(self, img, prob, bbox, trsfm=None):
        self.img = img
        self.prob = prob
        self.bbox = bbox
        self.transforms = trsfm
    
    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, idx):
        img, prob, bbox = self.img[idx], self.prob[idx], self.bbox[idx]
        if prob == 0:
            prob = np.array([0, 1], dtype='float32')
        else:
            prob = np.array([1, 0], dtype='float32')
        
        if self.transforms is not None:
            img = self.transforms(img)
        return img, torch.tensor(prob), torch.tensor(bbox)


def get_images(img_path, anno_path, valid_percent=0.1, resize_shape=(12, 12)):
    
    scale = resize_shape[0] / 400 # assuming that original image size is 400x400
    
    file_names = sorted(glob(os.path.join(img_path, '*.jpg')))
    anno_names = sorted(glob(os.path.join(anno_path, '*.npy')))
    anno_names.extend([''] * (len(file_names) - len(anno_names)))
    
    imgs, probs, bboxes = [], [], []
    
    for cur_file, cur_anno in tqdm(zip(file_names, anno_names), total=len(file_names)):
        
        img = io.imread(cur_file)
        
        if img.shape[-1] == 1:
            img = color.grey2rgb(img)
        
        age = os.path.splitext(os.path.basename(cur_file))[0].split('_')[0]
        
        if age == 'n':
            probs.append(0)
            bboxes.append(np.array([0,0,0,0], dtype='int64'))
        else:
            probs.append(1)
            a_bb = np.load(cur_anno)
            b_bb = np.array([int(round(k*scale)) for k in a_bb], dtype='int64')
            bboxes.append(b_bb)
        
        #img = sktrsfm.resize(img, resize_shape, anti_aliasing=True, preserve_range=True)[..., :3]
        img = torchvision.transforms.ToPILImage()(img.astype(np.uint8))
        img = img.resize(resize_shape, resample=Image.LANCZOS)
        imgs.append(img)
    
    rand_idx = np.random.permutation(np.arange(len(file_names)))
    imgs = [imgs[a] for a in rand_idx]
    probs = [probs[a] for a in rand_idx]
    bboxes = [bboxes[a] for a in rand_idx]
    
    vn = int(np.floor(len(imgs) * valid_percent))
    return imgs[vn:], probs[vn:], bboxes[vn:], imgs[:vn], probs[:vn], bboxes[:vn]


"""
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
"""
