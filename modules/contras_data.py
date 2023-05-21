from torch.utils.data import Dataset, DataLoader
import json
import re
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
import os
from torch.utils.data.distributed import DistributedSampler



def get_transforms(mode):
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
        #   transforms.RandomApply([color_jitter], p=0.8),
        #   GaussianBlur(kernel_size=int(0.1 * self.input_shape[0])),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    else:
        return transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    


class ContrasData(Dataset):
    def __init__(self,args,split,transform):
        super().__init__()
        self.img_dir = args.img_dir
        self.transforms = transform 
        if split == 'eval':
            self.data = json.load(open(args.ann_path,'r'))['train']
        else:
            self.data = json.load(open(args.ann_path,'r'))[split] 
            

        self.args = args
        self.split = split

    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):

        raw_data = self.data[index]
        img_path = raw_data['image_path']
        img = Image.open(os.path.join(self.img_dir, img_path[0])).convert('RGB')
        if self.split == 'train':
            img1 = self.transforms(img) 
            img2 = self.transforms(img) 
            img = (img1,img2)
        else:
            img = self.transforms(img) 
            
        report_tokens = self.clean_report_mimic_cxr(raw_data['report'])
        report = ' . '.join(report_tokens) + ' .'

        return img, report, index

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]

        return tokens



def get_data_loader(args, mode='train',batch_size=32,num_workers=4):

    
    transform = get_transforms(mode)
    if mode == 'train':
        dataset = ContrasData(args, mode, transform)
        sampler = DistributedSampler(dataset)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=(sampler is None),
            num_workers=num_workers, pin_memory=True, sampler=sampler,
            drop_last=True) 
        
    # used for computing mometum features
    elif mode == 'eval':
        dataset = ContrasData(args, mode, transform)
        sampler = DistributedSampler(dataset, shuffle=False)
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, sampler=sampler) 
    

    elif mode =='test':
        dataset = ContrasData(args, mode, transform)
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers, pin_memory=True,
            shuffle=False)


    return loader