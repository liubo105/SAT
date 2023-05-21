from cmath import inf
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
from models.gloria_model import GLoRIA
from modules.contras_loss import HierarchicalLoss, NTXentLoss,MixLoss
from transformers import BertTokenizer
import torch.nn.functional as F
from modules import utils
import argparse
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
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
    parser.add_argument('--soft_label', action='store_true') # AAAI
    parser.add_argument('--shuffle_txt', action='store_true') # EMNLP
    parser.add_argument('--mix', action='store_true') # EMNLP
    parser.add_argument('--mix_number', type=int, default=32) # EMNLP

    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--optim', default='adamw')
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--adam_eps', type=float, default=1e-6)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--warmup_steps',type=int,default=5)
    parser.add_argument('--alpha_weight', type=float, default=0.5)

    # Parameters Save Name
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--print', type=bool, default=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_path', type=str)
    parser.add_argument('--resume_dir', type=str)

    # Multi/Soft contrastive learning
    parser.add_argument('--threshold', type=float, default=0.98)



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
        self.model = GLoRIA()
        self.tokenizer = BertTokenizer.from_pretrained(args.text_encoder, cache_dir = args.cache_dir, local_files_only=True)
        self.print = self.args.print
        self.start_epoch = 0
        
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

        # self.optimizer = torch.optim.Adam(self.model.parameters(),
        #                   lr=self.args.lr, weight_decay=1e-6,betas=(0.5, 0.999))

        self.optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.lr, eps=self.args.adam_eps)

        self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, args.warmup_steps, args.epochs)

        # self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     self.optimizer, factor=0.5, patience=10, mode='min'
        # )
        self.scaler = GradScaler()

        if args.resume:
            print('resume optimizer, lr_scheduler, epoch....')
            self.optimizer.load_state_dict(checkpoint['optimizer'])  
            self.start_epoch = checkpoint['epoch'] + 1
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        dist.barrier()

        self.train_loader = train_loader
        self.eval_loader = val_loader
        self.test_loader = test_loader
 
    def train(self):

        loss_meter = utils.LossMeter()
        
        # fp16
        # self.model, self.optimizer = amp.initialize(self.model, self.optimizer,
        #                                       opt_level='O1')
        best_valid_loss = np.inf
        

        for epoch in range(self.start_epoch, self.args.epochs):

            
            train_loss = 0
            val_loss = 0 
            self.model.train()
            self.train_loader.sampler.set_epoch(epoch)
        
            sum_z = 0
            sum_a = 1e-9
            
            start_time = time.time()
            for step_i, (img, txt, index) in enumerate(self.train_loader):

                with autocast():
                    token_inputs = self.tokenizer(txt, padding=True, truncation=True, return_tensors='pt')
                    img, txt  = img.cuda(self.args.gpu), token_inputs.to(self.args.gpu)
                    img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, idx, probs = self.model(img,txt) 
                    # bsz = img.shape[0]
                    # z = torch.sum(probs>0) - bsz

                    # sum_z += z
                    # sum_a += bsz*4
                    

                    loss, attn_maps = self.model.module.calc_loss(
                            img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, idx, probs)

                
                self.optimizer.zero_grad()

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                # loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(),0.25)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                # self.optimizer.step()
                    
                train_loss += loss.item()
                loss_meter.update(loss.item())
            end_time = time.time()


            lr = self.lr_scheduler.get_last_lr()[0]  
            # lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            desc_str = f'Epoch {epoch} | LR {lr:.6f}'
            desc_str += f' | Loss {loss_meter.val:4f}'
            desc_str += f' | ratio {sum_z/sum_a}'


            if self.print:
                self.logger.info(f'training time: {end_time - start_time}')
                self.logger.info(desc_str)
                self.writer.add_scalar('lr', lr, global_step=epoch)
                self.writer.add_scalar('train_loss', train_loss/len(self.train_loader), global_step=epoch)
  
        
            self.lr_scheduler.step()
            # if self.args.gpu == 0:
            #     val_loss = self.eval(self.eval_loader)
            #     # self.writer.add_scalar('validation_loss', valid_loss, global_step=epoch)

            #     if r5 < best_valid_loss:
            #         best_valid_loss = r5
            #         self.save(epoch, os.path.join(args.task_path, f'best_model.pth'))
            #     # if valid_loss < best_valid_loss:
            #     #     best_valid_loss = valid_loss
            #     #     self.save(epoch, os.path.join(args.task_path, f'best_model.pth'))
                
            #     if epoch % self.args.eval_interval == 0:
            #         self.save(epoch, os.path.join(args.task_path, f'{epoch}_model.pth'))

            # print(f'before barrier... as {self.args.gpu}')
            # print(f'before scheduler... as {self.args.gpu}')

            # dist.broadcast(val_loss, 0, async_op=False)

            # self.lr_scheduler.step(val_loss)  

            torch.cuda.empty_cache()
            if self.print:
                val_start_time = time.time()
                val_loss = self.eval(self.eval_loader)
                val_end_time = time.time()
                self.logger.info(f'val time: {val_end_time - val_start_time}')
                self.logger.info(f'validate loss: {val_loss}')
                r5 = self.predict(self.test_loader)
                # self.save(epoch, os.path.join(args.task_path, f'{epoch}_model.pth'))




    def eval(self, dataloader):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for step_i, (img, txt,index) in enumerate(dataloader):

                token_inputs = self.tokenizer(txt, padding=True, truncation=True, return_tensors='pt')
                img, txt  = img.cuda(self.args.gpu), token_inputs.to(self.args.gpu)
                
                img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents,_,_ = self.model(img,txt,False)   
                loss, attn_maps = self.model.module.calc_loss(
                                    img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, None, None)
                val_loss += loss.item()
        
        return val_loss/len(dataloader)

                



    def predict(self, dataloader):
        
        self.model.eval()
        val_loss = 0
        img_embs_l = []
        img_embs_g = []
        txt_embs_l = []
        txt_embs_g = []
        caps_len = []
        with torch.no_grad():
            for step_i, (img, txt,index) in enumerate(dataloader):

                token_inputs = self.tokenizer(txt,padding=True,truncation=True, return_tensors='pt')

                img, txt = img.cuda(self.args.gpu), token_inputs.to(self.args.gpu)

 
                img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents,_,_ = self.model(img,txt,False)   

                # loss = self.criterion(f_img, f_txt)
                cap_lens = [len([w for w in sent if not w.startswith("[")]) for sent in sents]
                caps_len += cap_lens

                img_embs_l.append(img_emb_l)
                img_embs_g.append(img_emb_g.cpu())
                txt_embs_l.extend(text_emb_l)
                txt_embs_g.append(text_emb_g.cpu())
            


            
            img_embs_l = torch.cat(img_embs_l,0) 
            img_embs_g = torch.cat(img_embs_g,0)
            # txt_embs_l = torch.cat(txt_embs_l,1)
            txt_embs_g = torch.cat(txt_embs_g,0)
            i2t_r1,i2t_r5,i2t_r10,i2t_r50,i2t_r100,i2t_medr,i2t_meanr,t2i_r1,t2i_r5,t2i_r10,t2i_r50,t2i_r100,t2i_medr,t2i_meanr = \
            self.model.module.cross_modal_retrieval(img_embs_l, img_embs_g, txt_embs_l, txt_embs_g, caps_len)

            self.logger.info('------------------EVAL------------------')
            # self.logger.info(f'eval loss: {eval_loss}')
            # (r1,r5,r10,r50,r100,medr,meanr) = utils.i2t(np.array(img_embs), np.array(txt_embs))
            self.logger.info("Image to text: %.1f, %.1f, %.1f, %.1f , %.1f, %.1f, %.1f" %
                 (i2t_r1,i2t_r5,i2t_r10,i2t_r50,i2t_r100,i2t_medr,i2t_meanr))

            # (r1i,r5i,r10i,r50i,r100i,medri,meanri) = utils.t2i(np.array(img_embs), np.array(txt_embs))
            self.logger.info("Text to image: %.1f, %.1f, %.1f, %.1f , %.1f, %.1f, %.1f" %
                 (t2i_r1,t2i_r5,t2i_r10,t2i_r50,t2i_r100,t2i_medr,t2i_meanr))

        return i2t_r5

    # def predict(self,dataloader):
        
    #     self.model.eval()
    #     val_loss = 0
    #     aiacc_1,aiacc_3,aiacc_5 = 0,0,0
    #     atacc_1,atacc_3,atacc_5 = 0,0,0
    #     criterion = torch.nn.CrossEntropyLoss().cuda(self.args.gpu)
    #     with torch.no_grad():
    #         for step_i, (img, txt, index) in enumerate(
    #                                             tqdm(dataloader,ncols=180,desc='Prediction',disable= not self.print)):


    #             token_inputs = self.tokenizer(txt,padding=True,truncation=True, return_tensors='pt')

    #             img, txt = img.cuda(self.args.gpu), token_inputs.to(self.args.gpu)

    #             f_img, f_txt = self.model(img,txt,False)
    #             f_img = F.normalize(f_img, p=2, dim=1)
    #             f_txt = F.normalize(f_txt, p=2, dim=1)

    #             batch_size = f_img.shape[0]

    #             labels = torch.arange(start=0, end=batch_size, dtype=torch.long)
    #             labels = labels.cuda(self.args.gpu)

    #             logits_i2t = torch.matmul(f_img, torch.transpose(f_txt,0, 1)) / 0.1
    #             logits_t2i = torch.matmul(f_txt, torch.transpose(f_img,0, 1)) / 0.1



    #             loss = 0.5*criterion(logits_i2t,labels) + 0.5*criterion(logits_t2i,labels)

    #             accp_img = accuracy(logits_i2t, labels)
    #             accp_txt = accuracy(logits_t2i, labels)
    #             iacc_1,iacc_3,iacc_5 = accp_img[0][0],accp_img[1][0],accp_img[2][0]
    #             tacc_1,tacc_3,tacc_5 = accp_txt[0][0],accp_txt[1][0],accp_txt[2][0]

    #             aiacc_1+= iacc_1
    #             aiacc_3+= iacc_3
    #             aiacc_5+= iacc_5

    #             atacc_1+=tacc_1
    #             atacc_3+=tacc_3
    #             atacc_5+=tacc_5

                
    #             val_loss += loss.item()

    #         len_ = len(dataloader)
    #         eval_loss = val_loss/len_

    #         print('-----------________________EVAL________________-------------')
    #         print(f'eval: iacc@1:{aiacc_1/len_} iacc@3:{aiacc_3/len_} iacc@5:{aiacc_5/len_} ; tacc@1:{atacc_1/len_} tacc@3:{atacc_3/len_} tacc@5:{atacc_5/len_}')
    #         print(f'eval loss: {eval_loss}')
            


    #     return eval_loss

    
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
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    
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