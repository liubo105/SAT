import torch
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor import Meteor
from pycocoevalcap.rouge import Rouge
import logging
import os
import collections
from sklearn.metrics import f1_score, confusion_matrix
import numpy as np
from PIL import Image, ImageDraw
import random
import time 
from time import strftime,gmtime

class Logger(object):
    def __init__(self, filename, level=logging.INFO,
                 format='%(asctime)s %(levelname)s %(message)s',
                 datefmt='%a, %d %b %Y %H:%M:%S', filemode='w'):
        self.level = level
        self.format = format
        self.datefmt = datefmt
        self.filename = filename
        self.filemode = filemode
        logging.basicConfig(level=self.level,
                            format=self.format,
                            datefmt=self.datefmt,
                            filename=self.filename,
                            filemode=self.filemode)
        self._set_streaming_handler()

    def _set_streaming_handler(self, level=logging.INFO, formatter='%(asctime)s %(levelname)-8s %(message)s'):
        console = logging.StreamHandler()
        console.setLevel(level)
        curr_formatter = logging.Formatter(formatter)
        console.setFormatter(curr_formatter)
        logging.getLogger(self.filename).addHandler(console)

    def get_logger(self):
        return logging.getLogger(self.filename)


def task_dir(task):

    # if not os.path.isdir(output):
    #     os.makedirs(output, exist_ok=True)

    if not os.path.isdir(task):
        os.makedirs(task, exist_ok=True)

def i2t(images, captions):

    npts = images.shape[0]
    index_list = []
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    for index in range(npts):

        # Get query image
        im = images[index].reshape(1, images.shape[1])

        # Compute scores
        d = np.dot(im, captions.T).flatten()

        inds = np.argsort(d)[::-1]

        index_list.append(inds[0])

        # Score
        rank = np.where(inds == index)[0][0]

        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1,r5,r10,r50,r100,medr,meanr)

def t2i(images, captions):
    npts = captions.shape[0]
    index_list = []
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)

    for index in range(npts):

        # Get query caption
        cap = captions[index].reshape(1, captions.shape[1])

        # Compute scores
        d = np.dot(cap, images.T).flatten()

        inds = np.argsort(d)[::-1]

        index_list.append(inds[0])

        # Score
        rank = np.where(inds == index)[0][0]

        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    r50 = 100.0 * len(np.where(ranks < 50)[0]) / len(ranks)
    r100 = 100.0 * len(np.where(ranks < 100)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1

    return (r1,r5,r10,r50,r100,medr,meanr)







def compute_report_scores(gts, res):
    """
    Performs the MS COCO evaluation using the Python 3 implementation (https://github.com/salaniz/pycocoevalcap)
    :param gts: Dictionary with the image ids and their gold captions,
    :param res: Dictionary with the image ids ant their generated captions
    :print: Evaluation score (the mean of the scores of all the instances) for each measure
    """

    # Set up scorers
    scorers = [
        (Bleu(4), ["BLEU_1", "BLEU_2", "BLEU_3", "BLEU_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L")
    ]
    eval_res = {}
    # Compute score for each metric
    for scorer, method in scorers:
        try:
            score, scores = scorer.compute_score(gts, res, verbose=0)
        except TypeError:
            score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, m in zip(score, method):
                eval_res[m] = sc
        else:
            eval_res[method] = score
    return eval_res


def compute_image_scores(gts, res):
    '''
    accuracy and F1-score according to class
    :param gts: ground-truth labels
    :param res: predicted labels
    '''

    # multi-gpu 
    gts = np.array([i for i in gts if len(i)>0])
    res = np.array([i for i in res if len(i)>0])
    res_filter = np.array(res>0.5,dtype=float)
    acc = np.array(gts == res_filter, dtype=float).sum(0) / len(gts)
    f1 = f1_score(gts, res_filter,average=None)

    result = {'Acc':acc, 'F1': f1}
    return result


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x



def split_tensors(n, x):
    if torch.is_tensor(x):
        assert x.shape[0] % n == 0
        x = x.reshape(x.shape[0] // n, n, *x.shape[1:]).unbind(1)
    elif type(x) is list or type(x) is tuple:
        x = [split_tensors(n, _) for _ in x]
    elif x is None:
        x = [None] * n
    return x


def repeat_tensors(n, x):
    """
    For a tensor of size Bx..., we repeat it n times, and make it Bnx...
    For collections, do nested repeat
    """
    if torch.is_tensor(x):
        x = x.unsqueeze(1)  # Bx1x...
        x = x.expand(-1, n, *([-1] * len(x.shape[2:])))  # Bxnx...
        x = x.reshape(x.shape[0] * n, *x.shape[2:])  # Bnx...
    elif type(x) is list or type(x) is tuple:
        x = [repeat_tensors(n, _) for _ in x]
    return x


class LossMeter(object):
    def __init__(self, maxlen=100):
        """Computes and stores the running average"""
        self.vals = collections.deque([], maxlen=maxlen)

    def __len__(self):
        return len(self.vals)

    def update(self, new_val):
        self.vals.append(new_val)

    @property
    def val(self):
        return sum(self.vals) / (len(self.vals) + 1e-9)

    def __repr__(self):
        return str(self.val)

class TimeMeter(object):
    def __init__(self, batch_number):
        """Computes the used time and remaining time"""
        self.vals = []
        self.all = 0
        self.used = 0
        self.bn = batch_number

    def __len__(self):
        return len(self.vals)

    def update(self, new_val):
        self.vals.append(new_val)
        self.used = sum(self.vals)
        avg = self.used / len(self.vals) 
        self.all = avg * self.bn
        

    def __repr__(self):
        return '['+strftime('%M:%S',gmtime(self.used))+'/'+strftime('%M:%S',gmtime(self.all))+']'