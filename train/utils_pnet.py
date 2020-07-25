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

from random import randrange

# for parsing XML image annotations
import xml.etree.ElementTree as ET

# debug
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
        if self.transforms is not None:
            img = self.transforms(img)
        return img, torch.tensor(b_prob), torch.tensor(b_box), torch.tensor(f_prob), torch.tensor(f_box)

class CIVDS_bnet(data.Dataset):  # CIVDS: Children in Vehicles Dataset
    
    # img: children image
    # prob: whether the image contains a child or not
    # bbox: bounding box of the child in the image
    def __init__(self, img, b_prob, b_box, trsfm=None):
        self.img = img
        self.b_prob = b_prob
        self.b_box = b_box
        self.transforms = trsfm
    
    def __len__(self):
        return len(self.img)
    
    def __getitem__(self, idx):
        img, b_prob, b_box = self.img[idx], self.b_prob[idx], self.b_box[idx]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, torch.tensor(b_prob), torch.tensor(b_box)
    
def bbox_intersect(a, b):
    # get the intersection of bounding boxes a, b
    return np.array([max(a[0],b[0]), max(a[1],b[1]), min(a[2],b[2]), min(a[3],b[3])], dtype='int32')

def valid_bbox(a):
    if a[0] > a[2] or a[1] > a[3]:
        return False
    return True

def get_iou(a, b, eps=1e-5):
    x1, y1, x2, y2 = bbox_intersect(a, b)
    width = (x2 - x1)
    height = (y2 - y1)

    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    iou = area_overlap / (area_combined+eps)
    return iou

def upd_bbox(oboxa, niba, t_size):
    # updates the coordinates of a bounding box when the original image is cropped
    # assumes that nib is a square bbox
    obox, nib = oboxa.copy(), niba.copy()
    scl = t_size / (nib[2] - nib[0])
    obox = bbox_intersect(obox, nib)
    assert valid_bbox(obox)
    obox[0] -= nib[0]
    obox[1] -= nib[1]
    obox[2] -= nib[0]
    obox[3] -= nib[1]
    return np.array([int(round(k*scl)) for k in obox], dtype='int32')

def get_images(img_path, anno_path, valid_percent=0.1, resize_shape=(12, 12),
               add_orig=True, gen_xtra_neg=True, gen_xtra_body=True, gen_xtra_face=True):
    
    scale = resize_shape[0] / 400 # assuming that original image size is 400x400
    
    # get image file & annotation names
    
    file_names = sorted(glob(os.path.join(img_path, '*.jpg')))
    anno_names = sorted(glob(os.path.join(anno_path, '*.npy')))
    anno_names.extend([''] * (len(file_names) - len(anno_names))) # match length
    
    imgs, b_probs, b_boxes, f_probs, f_boxes = [], [], [], [], []
    
    dbg_i = -1  # debugging index
    
    for cur_file, cur_anno in tqdm(zip(file_names, anno_names), total=len(file_names)):
        
        dbg_i += 1
        
        img = io.imread(cur_file)

        if img.shape[-1] == 1:
            img = color.grey2rgb(img)

        img = torchvision.transforms.ToPILImage()(img.astype(np.uint8))
        age = os.path.splitext(os.path.basename(cur_file))[0].split('_')[0]
        
        # add original (raw) image

        if age == 'n': # no child in image
            if add_orig:
                b_probs.append(np.array([0, 1], dtype='float32'))
                b_boxes.append(np.array([0,0,0,0], dtype='int32')) 
                f_probs.append(np.array([0, 1], dtype='float32'))
                f_boxes.append(np.array([0,0,0,0], dtype='int32'))

        else:
            o = np.load(cur_anno)
            o_bb, o_fb = o[0], o[1]
            
            if add_orig:
                b_probs.append(np.array([1, 0], dtype='float32'))
                b_boxes.append(np.array([int(round(k*scale)) for k in o_bb], dtype='int32'))

                if (o_fb == np.array([0,0,0,0])).all():  # no face
                    f_probs.append(np.array([0, 1], dtype='float32'))
                else:
                    f_probs.append(np.array([1, 0], dtype='float32'))
                f_boxes.append(np.array([int(round(k*scale)) for k in o_fb], dtype='int32'))
        
        if add_orig:
            imgs.append(img.resize(resize_shape, resample=Image.LANCZOS))
        
        # generate additional negative samples
        
        if gen_xtra_neg:
            
            ct_size = 100

            while ct_size >= 12:
                mmax = 400 - ct_size

                for i in range(0, 1):
                    cx, cy = 0, 0

                    while True:
                        cx, cy = randrange(mmax), randrange(mmax)
                        nbox = [cx, cy, cx+ct_size, cy+ct_size]

                        if (get_iou(nbox, o_bb) < 0.10) and (get_iou(nbox, o_fb) < 0.05):
                            break

                    imgs.append(img.crop(nbox).resize(resize_shape, resample=Image.LANCZOS))
                    b_probs.append(np.array([0, 1], dtype='float32'))
                    b_boxes.append(np.array([0,0,0,0], dtype='int32')) 
                    f_probs.append(np.array([0, 1], dtype='float32'))
                    f_boxes.append(np.array([0,0,0,0], dtype='int32'))

                ct_size = int(ct_size * 0.709)

        if age == 'n':
            continue

        # generate additional positive body samples
        
        if gen_xtra_body:
            
            ct_size = 300

            while ct_size >= 200:
                mmax = 400 - ct_size

                for i in range(0, 3):
                    cx, cy, nbox = 0, 0, []

                    trycnt, success = 0, False
                    
                    while trycnt < 20:
                        cx, cy = randrange(mmax), randrange(mmax)
                        nbox = [cx, cy, cx+ct_size, cy+ct_size]
                        c_iou = get_iou(nbox, o_bb)

                        if 0.1 <= c_iou < 0.2:
                            b_probs.append(np.array([0.2, 0], dtype='float32'))
                            success = True
                            break
                            
                        elif 0.2 <= c_iou < 0.4:
                            b_probs.append(np.array([0.5, 0], dtype='float32'))
                            success = True
                            break
                            
                        elif 0.4 <= c_iou < 0.7:
                            b_probs.append(np.array([0.8, 0], dtype='float32'))
                            success = True
                            break

                        elif 0.7 <= c_iou:
                            b_probs.append(np.array([1, 0], dtype='float32'))
                            success = True
                            break
                        
                        trycnt += 1

                    if success:

                        imgs.append(img.crop(nbox).resize(resize_shape, resample=Image.LANCZOS))
                        b_boxes.append(upd_bbox(o_bb, nbox, resize_shape[0]))

                        # add face annotation if face is present as well

                        if (o_fb == np.array([0,0,0,0])).all():  # no face
                            f_probs.append(np.array([0, 1], dtype='float32'))
                            f_boxes.append(np.array([0,0,0,0], dtype='int32'))

                        else:
                            nfb = bbox_intersect(o_fb, nbox)

                            if valid_bbox(nfb):
                                nfb_iou = get_iou(nfb, o_fb)

                                if nfb_iou < 0.35: # low confidence partial face
                                    f_probs.append(np.array([0.5, 0], dtype='float32'))

                                elif 0.35 <= nfb_iou < 0.5: # mid confidence partial face
                                    f_probs.append(np.array([0.8, 0], dtype='float32'))

                                else: # pretty much full-face
                                    f_probs.append(np.array([0.99, 0], dtype='float32'))

                                f_boxes.append(upd_bbox(nfb, nbox, resize_shape[0]))
                            else:
                                f_probs.append(np.array([0, 1], dtype='float32'))
                                f_boxes.append(np.array([0,0,0,0], dtype='int32'))
                    else:
                        print("failed to select a body with appropriate IOU!")
                        assert False

                    """
                    # debug code
                    fig,ax = plt.subplots(1)

                    tnb = upd_bbox(o_bb, nbox, nbox[2]-nbox[0])
                    rect = patches.Rectangle(
                        (tnb[0], tnb[1]),
                        tnb[2]-tnb[0], tnb[3]-tnb[1],
                        linewidth=2, edgecolor='r', facecolor='none'
                    )
                    ax.add_patch(rect)

                    if not (o_fb == np.array([0,0,0,0])).all():
                        tnb = upd_bbox(nfb, nbox, nbox[2]-nbox[0])
                        rect = patches.Rectangle(
                            (tnb[0], tnb[1]),
                            tnb[2]-tnb[0], tnb[3]-tnb[1],
                            linewidth=2, edgecolor='g', facecolor='none'
                        )
                        ax.add_patch(rect)

                    ax.imshow(img.crop(nbox))
                    plt.show()
                    """

                ct_size = int(ct_size * 0.95)

                
        # generate positive face samples
        
        if gen_xtra_face:
        
            if (o_fb == np.array([0,0,0,0])).all():  # no face
                continue

            ct_size = 300

            while ct_size >= 250:

                lrx = o_fb[2] - ct_size
                lry = o_fb[3] - ct_size

                for i in range(0, 2):
                    if lrx >= o_fb[0] or lry >= o_fb[1]:
                        print("### ERROR ###")
                        print("id: {}".format(dbg_i))
                        print(os.path.basename(cur_file))
                        print("o_fb: {}".format(o_fb))
                        plt.imshow(img)
                        plt.show()
                        break

                    cx, cy = randrange(lrx, o_fb[0]), randrange(lry, o_fb[1])
                    nbox = [cx, cy, cx+ct_size, cy+ct_size]
                    f_probs.append(np.array([1, 0], dtype='float32'))

                    imgs.append(img.crop(nbox).resize(resize_shape, resample=Image.LANCZOS))
                    f_boxes.append(upd_bbox(o_fb, nbox, resize_shape[0]))

                    # add partial body annotation if applicable

                    nbb = bbox_intersect(o_bb, nbox)

                    if valid_bbox(nbb):
                        nbb_iou = get_iou(nbb, o_bb)

                        if nbb_iou < 0.35: # low confidence partial body
                            b_probs.append(np.array([0.5, 0], dtype='float32'))

                        elif 0.35 <= nbb_iou < 0.5: # mid confidence partial body
                            b_probs.append(np.array([0.8, 0], dtype='float32'))

                        else: # pretty much full-body
                            b_probs.append(np.array([0.99, 0], dtype='float32'))

                        b_boxes.append(upd_bbox(nbb, nbox, resize_shape[0]))
                    else:
                        b_probs.append(np.array([0, 1], dtype='float32'))
                        b_boxes.append(np.array([0,0,0,0], dtype='int32'))

                ct_size = int(ct_size * 0.9)
    
    rand_idx = np.random.permutation(np.arange(len(imgs)))
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
