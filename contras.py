import torch.backends.cudnn as cudnn
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import os
import random

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from modules.gloria_data import get_data_loader
from models.contras_model import ContrasMed
from modules.contras_loss import HierarchicalLoss, NTXentLoss
from transformers import BertTokenizer
import torch.nn.functional as F
from modules import utils
import argparse
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
import time

def parse_args(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=105, help='random seed')
    parser.add_argument('--gpu', type=int, default=0)

    # Data Splits
    parser.add_argument("--dataset", type=str, default='mimic_cxr')
    parser.add_argument("--img_dir", type=str, default='data/mimic_cxr/images/')
    parser.add_argument("--ann_path", type=str, default='data/mimic_cxr/annotation.json')
    parser.add_argument("--train", default='train')
    parser.add_argument("--valid", default='valid')
    parser.add_argument("--test", default=None)

    # Checkpointconda install tsnecuda -c cannylab -c pytorch
    parser.add_argument('--output', type=str, default='./output')

    # CPU/GPU
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)

    # Model Config
    parser.add_argument('--img_encoder', type=str, default='resnet50')
    parser.add_argument('--text_encoder', type=str, default='emilyalsentzer/Bio_ClinicalBERT')
    parser.add_argument('--cache_dir', type=str, default='./pretrains/biobert')
    parser.add_argument('--out_dim', type=int, default=512)
    parser.add_argument('--soft_label', action='store_true')
    parser.add_argument('--way', type=str)


    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--optim', default='adamw')
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--adam_eps', type=float, default=1e-6)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--warmup_steps',type=int,default=20)
    parser.add_argument('--alpha_weight', type=float, default=0.5)

    # Parameters Save Name
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--print', type=bool, default=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_path', type=str)
    parser.add_argument('--resume_dir', type=str)

    # Multi contrastive learning
    parser.add_argument('--threshold', type=float, default=0.98)
    parser.add_argument('--threshold1', type=float, default=0.97)



    # Parse the arguments.
    if parse:
        args = parser.parse_args()
    # For interative engironmnet (ex. jupyter)
    else:
        args = parser.parse_known_args()[0]

    return args



class Trainer(object):
    def __init__(self,args,train_loader=None, val_loader=None, test_loader=None, train=True):
        super(Trainer,self).__init__()
        

        self.args = args
        self.model = ContrasMed(args)
        self.tokenizer = BertTokenizer.from_pretrained(args.text_encoder, cache_dir = args.cache_dir, local_files_only=True)
        self.print = self.args.print
        self.start_epoch = 0
        if self.args.soft_label:
            self.topK = [args.topk]
            self.threshold = [args.threshold, args.threshold1]
            # self.threshold = args.threshold
            # self.topK = [1,4]
        
        if self.args.gpu != 0:
            self.print = False

        if self.print:
            self.logger = utils.Logger(os.path.join(args.task_path,'task.log')).get_logger()
            self.logger.info(args)
            self.writer = SummaryWriter(log_dir=args.task_path)
            
        
        if args.resume:
            print('resume model weights')
            map_location = {'cuda:%d' % 0: 'cuda:%d' % args.gpu}
            checkpoint = torch.load(args.resume_path, map_location = map_location) 
            if print: # only load ckpt at rank 0, then DDP could help distribute weights to other GPUs
                ckpt = checkpoint['model'] 
                for i in list(ckpt.keys()):
                    if i.startswith('module.'):
                        ckpt[i[len('module.'):]] = ckpt[i]
                        del ckpt[i] 
                self.model.load_state_dict(ckpt, strict=True) 
            
        # Use a barrier() to make sure that process 1-N loads the model after process 0
        dist.barrier()
        
        self.model = nn.SyncBatchNorm.convert_sync_batchnorm(self.model).cuda(args.gpu)
        self.model = DDP(self.model,device_ids=[args.gpu],find_unused_parameters=True)


        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.lr, eps=self.args.adam_eps)

        self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer,args.warmup_steps, args.epochs)

        if args.resume:
            print('resume optimizer, lr_scheduler, epoch....')
            self.optimizer.load_state_dict(checkpoint['optimizer'])  
            self.start_epoch = checkpoint['epoch'] + 1
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        dist.barrier()

        self.train_loader = train_loader
        self.eval_loader = val_loader
        self.test_loader = test_loader
        if args.soft_label:
            self.criterion = HierarchicalLoss(temperature=0.1, alpha_weight=args.alpha_weight)
            self.criterion1 = NTXentLoss(self.args.gpu, temperature=0.1, alpha_weight=args.alpha_weight)

        else:
            self.criterion = NTXentLoss(self.args.gpu, temperature=0.1, alpha_weight=args.alpha_weight)
            self.criterion1 = NTXentLoss(self.args.gpu, temperature=0.1, alpha_weight=args.alpha_weight)
        
 
    def train(self):

        loss_meter = utils.LossMeter()
        
        # fp16
        # self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
        #                                       opt_level='O1')
        best_valid_loss = np.inf
        best_epoch = 0
        

        for epoch in range(self.start_epoch, self.args.epochs):
            
            
            train_loss = 0
            self.model.train()
            self.train_loader.sampler.set_epoch(epoch)
            start_time = time.time()
                        
            for step_i, (img, txt, index) in enumerate(self.train_loader):

                token_inputs = self.tokenizer(txt, padding=True, truncation=True, return_tensors='pt')
                img, txt  = img.cuda(self.args.gpu), token_inputs.to(self.args.gpu)

                if self.args.soft_label:

                    f_img, f_txt, labels  = self.model(img, txt, self.args.way, True, self.threshold)

                    loss = self.criterion(f_img, f_txt, labels)

                
                else:

                    f_img, f_txt = self.model(img,txt)
                    loss = self.criterion(f_img, f_txt)

                    


                self.optimizer.zero_grad()
                # with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                #     scaled_loss.backward()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),0.25)

                train_loss += loss.item()
                self.optimizer.step()
                    
                lr = self.lr_scheduler.get_last_lr()[0]  
                loss_meter.update(loss.item())
                desc_str = f'Epoch {epoch} | LR {lr:.6f}'
                desc_str += f' | Loss {loss_meter.val:4f}'

            end_time = time.time()

            if self.print:
                # self.logger.info(f'training time: {end_time - start_time}')
                self.logger.info(desc_str)
                self.writer.add_scalar('lr', lr, global_step=epoch)
                self.writer.add_scalar('train_loss', train_loss/len(self.train_loader), global_step=epoch)

            
            self.lr_scheduler.step()      
        
            
            if self.print:
                val_time_start = time.time()
                val_loss = self.validate(self.eval_loader)
                val_time_end = time.time()
                # self.logger.info(f'val time: {val_time_end - val_time_start}')
                if val_loss < best_valid_loss:
                    best_valid_loss = val_loss
                    best_epoch = epoch
                self.logger.info(f'Best Min Loss: {best_valid_loss} at epoch {best_epoch}')
                r5 = self.predict(self.test_loader)


    def validate(self,dataloader):
        
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for step_i, (img, txt,index) in enumerate(dataloader):


                token_inputs = self.tokenizer(txt,padding=True,truncation=True, return_tensors='pt')

                img, txt = img.cuda(self.args.gpu), token_inputs.to(self.args.gpu)

 
                f_img, f_txt = self.model(img,txt)

                loss = self.criterion1(f_img, f_txt)

                val_loss += loss.item()

            eval_loss = val_loss/len(dataloader)

            self.logger.info('------------------EVAL------------------')
            self.logger.info(f'eval loss: {eval_loss}')
        

        return eval_loss

    def predict(self,dataloader):
        
        self.model.eval()
        val_loss = 0
        img_embs = []
        txt_embs = []
        with torch.no_grad():
            for step_i, (img, txt,index) in enumerate(dataloader):


                token_inputs = self.tokenizer(txt,padding=True,truncation=True, return_tensors='pt')

                img, txt = img.cuda(self.args.gpu), token_inputs.to(self.args.gpu)

 
                f_img, f_txt = self.model(img,txt)

                # loss = self.criterion(f_img, f_txt)

                f_img = F.normalize(f_img, p=2, dim=1)
                f_txt = F.normalize(f_txt, p=2, dim=1)

                
                img_embs.extend(f_img.data.cpu().numpy().copy())
                txt_embs.extend(f_txt.data.cpu().numpy().copy())
                

                # val_loss += loss.item()

            # eval_loss = val_loss/len(dataloader)

            self.logger.info('------------------Test------------------')
            # self.logger.info(f'eval loss: {eval_loss}')
            (r1,r5,r10,r50,r100,medr,meanr) = utils.i2t(np.array(img_embs), np.array(txt_embs))
            self.logger.info("Image to text: %.1f, %.1f, %.1f, %.1f , %.1f, %.1f, %.1f" %
                 (r1,r5,r10,r50,r100,medr,meanr))

            (r1i,r5i,r10i,r50i,r100i,medri,meanri) = utils.t2i(np.array(img_embs), np.array(txt_embs))
            self.logger.info("Text to image: %.1f, %.1f, %.1f, %.1f , %.1f, %.1f, %.1f" %
                 (r1i,r5i,r10i,r50i,r100i,medri,meanri))

        return r5

    
    def save(self, epoch,path):

        checkpoint = {
            "model": self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            "epoch": epoch,
            'lr_scheduler': self.lr_scheduler.state_dict()
        }
        torch.save(checkpoint, path)

        

def accuracy(output, target, topk=(1,3,5)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def main_worker(gpu, args):

    # GPU is assigned
    args.gpu = gpu
    args.rank = gpu
    print(f'Process Launching at GPU {args.gpu}')

    
    torch.cuda.set_device(args.gpu)
    dist.init_process_group(backend='nccl')

    train_loader = get_data_loader(
        args,
        mode='train', batch_size=args.batch_size,
        num_workers=4
    )

    eval_loader = get_data_loader(
        args,
        mode='val', batch_size=args.batch_size,
        num_workers=4
    )


    test_loader = get_data_loader(
        args,
        mode='test', batch_size=args.batch_size,
        num_workers=4
    )


    trainer = Trainer(args, train_loader, eval_loader, test_loader, train=True)
    trainer.train()



if __name__ == "__main__":

    args = parse_args()
    print('task start.....')

    # Set seeds
    cudnn.benchmark = True
    cudnn.deterministic = False
    seed = args.seed + args.local_rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
    if args.local_rank in [0, -1]:
        if not args.resume:
            comments = []
            if args.comment != '':
                comments.append(args.comment)
            comment = '_'.join(comments)

            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M')

            run_name = f'{current_time}_{args.world_size}GPU'
            if len(comments) > 0:
                run_name += f'_{comment}'

            args.task_path = f'{args.output}/{run_name}'
            utils.task_dir(args.task_path)
        else:
            args.task_path = args.resume_dir
       
    # device = torch.device('cuda:0')
    # args.device = device
    main_worker(args.local_rank, args)