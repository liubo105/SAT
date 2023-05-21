import torch.nn as nn
from modules.resnet import resnet50
from transformers import BertModel
import torch
import torch.nn.functional as F
import math
import numpy as np

class ContrasMed(nn.Module):

    def __init__(self,args):
        super().__init__()

        self.img_encoder = resnet50(pretrained=True)
        in_dim = self.img_encoder.fc.weight.shape[1]
        self.img_encoder.fc = nn.Sequential(
                        nn.Linear(in_dim,in_dim),
                        nn.ReLU(),
                        nn.Linear(in_dim, args.out_dim)
                        )
    

        self.bert = BertModel.from_pretrained(args.text_encoder, cache_dir = args.cache_dir, local_files_only=True)

        # Follow ConVIRT, freeze embeddings and first 6 layers of BERT
        for embeding in self.bert.embeddings.parameters():
            embeding.requires_grad = False

        for i in range(6):
            for param in self.bert.encoder.layer[i].parameters():
                param.requires_grad = False
                
        self.bert_proj = nn.Sequential(
                        nn.Linear(768,768),
                        nn.ReLU(),
                        nn.Linear(768, args.out_dim)
                        )
        self.args = args
        if args.soft_label:
            self.tool_bert = BertModel.from_pretrained(args.text_encoder, cache_dir = args.cache_dir, local_files_only=True)
        # self.threshold = args.threshold


    def mean_pooling(self, model_output, attention_mask):
        """
        Mean Pooling - Take attention mask into account for correct averaging
        Reference: https://www.sbert.net/docs/usage/computing_sentence_embeddings.html

        """
        token_embeddings = model_output  

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def token_pooling(self,model_output):
        return model_output[:,0]


    def sum_pooling(self,model_output,attention_mask):
        """
        Sum Pooling - Take attention mask into account for correct averaging
        Reference: https://www.sbert.net/docs/usage/computing_sentence_embeddings.html

        """
        token_embeddings = model_output  

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
       
        return sum_embeddings

    def max_pooling(self, model_output, attention_mask):
        
        token_embeddings = model_output  

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        token_embeddings[input_mask_expanded == 0] = -1e9

        return torch.max(token_embeddings, 1)[0]


    def getTextSoftTarget(self, raw_txt, way, threshold):


        selected = []
        self.tool_bert.eval()
        with torch.no_grad():
            f_txt = self.tool_bert(**raw_txt)

            # using mean features  
            if way == 'max':
                f_txt = self.max_pooling(f_txt[0], raw_txt['attention_mask'])
            elif way == 'mean':
                f_txt = self.mean_pooling(f_txt[0], raw_txt['attention_mask'])
            elif way == 'sum':
                f_txt = self.sum_pooling(f_txt[0], raw_txt['attention_mask'])
            elif way == 'token':
                f_txt = self.token_pooling(f_txt[0])


            # l2 normalization
            f_txt = F.normalize(f_txt, p=2, dim=1)
            # cosin score
            scores = torch.matmul(f_txt,f_txt.transpose(-2,-1)) 
            return (scores, threshold)


    
    def encode_txt(self, text):

        f_txt = self.bert(**text)
        # f_txt = self.mean_pooling(f_txt[0],text['attention_mask'])
        f_txt = self.max_pooling(f_txt[0], text['attention_mask'])
        f_txt = self.bert_proj(f_txt)
        return f_txt

    def encode_img(self, img):
        f_img = self.img_encoder(img)
        return f_img


    def feature_mix(self,f_img,f_txt):

        

        b, s = f_img.shape
        choices = list(range(b))
        

        # for i in range():

        extra_num = self.args.mix_number
        img_idx = torch.tensor([list(np.random.choice(choices, 2,replace=False)) for i in range(extra_num)]).cuda()
        # img_idx = torch.multinomial(img_idx, 2, replacement=False)
        img_prob = torch.tensor(np.random.uniform(0, 1, extra_num)).reshape(extra_num,-1).float().cuda()
        img_gen_a = torch.index_select(f_img,0,img_idx[:,0])
        img_gen_b = torch.index_select(f_img,0,img_idx[:,1])
        f_img_gen = img_gen_a*img_prob + img_gen_b*(1-img_prob)


        txt_idx = torch.tensor([list(np.random.choice(choices, 2,replace=False)) for i in range(extra_num)]).cuda()
        # txt_idx = torch.multinomial(img_idx, 2, replacement=False)
        txt_prob = torch.tensor(np.random.uniform(0, 1, extra_num)).reshape(extra_num,-1).float().cuda()
        txt_gen_a = torch.index_select(f_txt,0,txt_idx[:,0])
        txt_gen_b = torch.index_select(f_txt,0,txt_idx[:,1])
        f_txt_gen = txt_gen_a*txt_prob + txt_gen_b*(1-txt_prob)

        return f_img_gen, f_txt_gen



    def forward(self, img, text, way='max', soft_label=False, threshold=None, mix=None):
        
        f_img = self.encode_img(img)

        f_txt = self.encode_txt(text)

        if soft_label:
            labels = self.getTextSoftTarget(text,way,threshold)

            return f_img, f_txt, labels
            
        return f_img, f_txt

