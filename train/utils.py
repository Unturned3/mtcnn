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
        bb = bb.astype('int32')
        np.save(os.path.splitext(ca)[0]+'.npy', bb)
    
    print('done!')

class CIVDS(data.Dataset):  # CIVDS: Children in Vehicles Dataset
    
    # img: children image
    # prob: whether the image contains a child or not
    # bbox: bounding box of the child in the image
    def __init__(self, img, b_prob, b_box, f_prob, f_box, trsfm=None):
        self.img = img
        self.b_prob = b_prob
        self.b_box = b_box
        self.f_prob = f_prob
        self.f_box = f_box
        self.transforms = trsfm
    
    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, idx):
        img, b_prob, b_box, f_prob, f_box = self.img[idx], self.b_prob[idx], self.b_box[idx], self.f_prob[idx], self.f_box[idx]
        """
        if prob == 0:
            prob = np.array([0, 1], dtype='float32')
        else:
            prob = np.array([1, 0], dtype='float32')
        """
        if self.transforms is not None:
            img = self.transforms(img)
        return img, torch.tensor(b_prob), torch.tensor(b_box), torch.tensor(f_prob), torch.tensor(f_box)


def get_images(img_path, anno_path, valid_percent=0.1, resize_shape=(12, 12)):
    
    scale = resize_shape[0] / 400 # assuming that original image size is 400x400
    
    file_names = sorted(glob(os.path.join(img_path, '*.jpg')))
    anno_names = sorted(glob(os.path.join(anno_path, '*.npy')))
    anno_names.extend([''] * (len(file_names) - len(anno_names)))
    
    imgs, b_probs, b_boxes, f_probs, f_boxes = [], [], [], [], []
    
    for cur_file, cur_anno in tqdm(zip(file_names, anno_names), total=len(file_names)):
        
        img = io.imread(cur_file)
        
        if img.shape[-1] == 1:
            img = color.grey2rgb(img)
        
        age = os.path.splitext(os.path.basename(cur_file))[0].split('_')[0]
        
        if age == 'n': # no child in image
            b_probs.append(np.array([0, 1], dtype='float32'))
            b_boxes.append(np.array([0,0,0,0], dtype='int32')) 
            f_probs.append(np.array([0, 1], dtype='float32'))
            f_boxes.append(np.array([0,0,0,0], dtype='int32'))
        else:
            b_probs.append(np.array([1, 0], dtype='float32'))
            o = np.load(cur_anno)
            bb, fb = o[0], o[1]
            bb = np.array([int(round(k*scale)) for k in bb], dtype='int32')
            fb = np.array([int(round(k*scale)) for k in fb], dtype='int32')
            b_boxes.append(bb)
            
            if (fb == np.array([0,0,0,0])).all():  # no face
                f_probs.append(np.array([0, 1], dtype='float32'))
            else:
                f_probs.append(np.array([1, 0], dtype='float32'))
            f_boxes.append(fb)
        
        #img = sktrsfm.resize(img, resize_shape, anti_aliasing=True, preserve_range=True)[..., :3]
        img = torchvision.transforms.ToPILImage()(img.astype(np.uint8))
        img = img.resize(resize_shape, resample=Image.LANCZOS)
        imgs.append(img)
    
    rand_idx = np.random.permutation(np.arange(len(file_names)))
    imgs = [imgs[a] for a in rand_idx]
    b_probs = [b_probs[a] for a in rand_idx]
    b_boxes = [b_boxes[a] for a in rand_idx]
    f_probs = [f_probs[a] for a in rand_idx]
    f_boxes = [f_boxes[a] for a in rand_idx]
    
    vn = int(np.floor(len(imgs) * valid_percent))
    return imgs[vn:], b_probs[vn:], b_boxes[vn:], f_probs[vn:], f_boxes[vn:], imgs[:vn], b_probs[:vn], b_boxes[:vn], f_probs[:vn], f_boxes[:vn]


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
