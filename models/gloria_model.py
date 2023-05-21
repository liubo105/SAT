import torch
import torch.nn as nn
import cv2
import re
import numpy as np
from modules.resnet import resnet50
import torch.nn.functional as F
from sklearn import metrics

from PIL import Image
from modules.gloria_loss import *
from transformers import BertModel, BertTokenizer
from omegaconf import OmegaConf
from tqdm import tqdm

class BertEncoder(nn.Module):
    def __init__(self, cfg):
        super(BertEncoder, self).__init__()

        self.last_n_layers = cfg.model.text.last_n_layers
        self.aggregate_method = cfg.model.text.aggregate_method
        self.norm = cfg.model.text.norm
        self.embedding_dim = cfg.model.text.embedding_dim
        self.freeze_bert = cfg.model.text.freeze_bert
        self.agg_tokens = cfg.model.text.agg_tokens

        self.model = BertModel.from_pretrained(cfg.model.text.bert_type, cache_dir = cfg.model.text.cache_dir, local_files_only=True, output_hidden_states=True)

        self.tokenizer = BertTokenizer.from_pretrained(cfg.model.text.bert_type, cache_dir = cfg.model.text.cache_dir, local_files_only=True)
        self.idxtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}

        self.emb_global, self.emb_local = None, None

        if self.freeze_bert is True:
            print("Freezing BERT model")
            for param in self.model.parameters():
                param.requires_grad = False

    def aggregate_tokens(self, embeddings, caption_ids):

        batch_size, num_layers, num_words, dim = embeddings.shape
        embeddings = embeddings.permute(0, 2, 1, 3)
        agg_embs_batch = []
        sentences = []

        # loop over batch
        for embs, caption_id in zip(embeddings, caption_ids):

            agg_embs = []
            token_bank = []
            words = []
            word_bank = []

            # loop over sentence
            for word_emb, word_id in zip(embs, caption_id):

                word = self.idxtoword[word_id.item()]

                if word == "[SEP]":
                    new_emb = torch.stack(token_bank)
                    new_emb = new_emb.sum(axis=0)
                    agg_embs.append(new_emb)
                    words.append("".join(word_bank))

                    agg_embs.append(word_emb)
                    words.append(word)
                    break

                if not word.startswith("##"):
                    if len(word_bank) == 0:
                        token_bank.append(word_emb)
                        word_bank.append(word)
                    else:
                        new_emb = torch.stack(token_bank)
                        new_emb = new_emb.sum(axis=0)
                        agg_embs.append(new_emb)
                        words.append("".join(word_bank))

                        token_bank = [word_emb]
                        word_bank = [word]
                else:
                    if word.startswith("##"):
                        token_bank.append(word_emb)
                        word_bank.append(word[2:])

            agg_embs = torch.stack(agg_embs)
            padding_size = num_words - len(agg_embs)
            paddings = torch.zeros(padding_size, num_layers, dim)
            paddings = paddings.to(agg_embs.device)
            words = words + ["[PAD]"] * padding_size

            agg_embs_batch.append(torch.cat([agg_embs, paddings]))
            sentences.append(words)

        agg_embs_batch = torch.stack(agg_embs_batch)
        agg_embs_batch = agg_embs_batch.permute(0, 2, 1, 3)
        return agg_embs_batch, sentences

    def forward(self, ids, attn_mask, token_type):

        outputs = self.model(ids, attn_mask, token_type)

        # aggregate intermetidate layers
        if self.last_n_layers > 1:
            all_embeddings = outputs[2]
            embeddings = torch.stack(
                all_embeddings[-self.last_n_layers :]
            )  # layers, batch, sent_len, embedding size

            embeddings = embeddings.permute(1, 0, 2, 3)

            if self.agg_tokens:
                embeddings, sents = self.aggregate_tokens(embeddings, ids)
            else:
                sents = [[self.idxtoword[w.item()] for w in sent] for sent in ids]

            sent_embeddings = embeddings.mean(axis=2)

            if self.aggregate_method == "sum":
                word_embeddings = embeddings.sum(axis=1) # batch, words, 768
                sent_embeddings = sent_embeddings.sum(axis=1) # batch, 768
            elif self.aggregate_method == "mean":
                word_embeddings = embeddings.mean(axis=1)
                sent_embeddings = sent_embeddings.mean(axis=1)
            else:
                print(self.aggregate_method)
                raise Exception("Aggregation method not implemented")

        # use last layer
        else:
            word_embeddings, sent_embeddings = outputs[0], outputs[1]

        batch_dim, num_words, feat_dim = word_embeddings.shape
        word_embeddings = word_embeddings.view(batch_dim * num_words, feat_dim)
        if self.emb_local is not None:
            word_embeddings = self.emb_local(word_embeddings)
        word_embeddings = word_embeddings.view(batch_dim, num_words, self.embedding_dim)
        word_embeddings = word_embeddings.permute(0, 2, 1)

        if self.emb_global is not None:
            sent_embeddings = self.emb_global(sent_embeddings)

        if self.norm is True:
            word_embeddings = word_embeddings / torch.norm(
                word_embeddings, 2, dim=1, keepdim=True
            ).expand_as(word_embeddings)
            sent_embeddings = sent_embeddings / torch.norm(
                sent_embeddings, 2, dim=1, keepdim=True
            ).expand_as(sent_embeddings)

        return word_embeddings, sent_embeddings, sents



def resnet_50(pretrained=True):
    model = resnet50(pretrained=pretrained)
    feature_dims = model.fc.in_features
    model.fc = nn.Identity()
    return model, feature_dims, 1024

class ImageEncoder(nn.Module):
    def __init__(self, cfg):
        super(ImageEncoder, self).__init__()

        self.output_dim = cfg.model.text.embedding_dim
        self.norm = cfg.model.vision.norm

        self.model, self.feature_dim, self.interm_feature_dim = resnet_50(
            pretrained=cfg.model.vision.pretrained
        )

        self.global_embedder = nn.Linear(self.feature_dim, self.output_dim)
        self.local_embedder = nn.Conv2d(
            self.interm_feature_dim,
            self.output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # if cfg.model.ckpt_path is not None:
        #     self.init_trainable_weights()

        if cfg.model.vision.freeze_cnn:
            print("Freezing CNN model")
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, get_local=False):
        # --> fixed-size input: batch x 3 x 299 x 299
        if "resnet" or "resnext" in self.cfg.model.vision.model_name:
            global_ft, local_ft = self.resnet_forward(x, extract_features=True)
        elif "densenet" in self.cfg.model.vision.model_name:
            global_ft, local_ft = self.dense_forward(x, extract_features=True)

        if get_local:
            return global_ft, local_ft
        else:
            return global_ft

    def generate_embeddings(self, global_features, local_features):

        global_emb = self.global_embedder(global_features)
        local_emb = self.local_embedder(local_features)

        if self.norm is True:
            local_emb = local_emb / torch.norm(
                local_emb, 2, dim=1, keepdim=True
            ).expand_as(local_emb)
            global_emb = global_emb / torch.norm(
                global_emb, 2, dim=1, keepdim=True
            ).expand_as(global_emb)

        return global_emb, local_emb

    def resnet_forward(self, x, extract_features=False):

        # --> fixed-size input: batch x 3 x 299 x 299
        x = nn.Upsample(size=(299, 299), mode="bilinear", align_corners=True)(x)

        x = self.model.conv1(x)  # (batch_size, 64, 150, 150)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)  # (batch_size, 64, 75, 75)
        x = self.model.layer2(x)  # (batch_size, 128, 38, 38)
        x = self.model.layer3(x)  # (batch_size, 256, 19, 19)
        local_features = x
        x = self.model.layer4(x)  # (batch_size, 512, 10, 10)

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        return x, local_features

    def densenet_forward(self, x, extract_features=False):
        pass

    def init_trainable_weights(self):
        initrange = 0.1
        self.emb_features.weight.data.uniform_(-initrange, initrange)
        self.emb_cnn_code.weight.data.uniform_(-initrange, initrange)



class GLoRIA(nn.Module):
    def __init__(self):
        super(GLoRIA, self).__init__()

        self.cfg = OmegaConf.load('/apdcephfs/share_1290796/kelvinbliu/MedRp/models/config_gloria.yaml')
        self.text_encoder = BertEncoder(self.cfg)
        self.img_encoder = ImageEncoder(self.cfg)

        self.local_loss = local_loss
        self.global_loss = global_loss
        self.local_loss_weight = self.cfg.model.gloria.local_loss_weight
        self.global_loss_weight = self.cfg.model.gloria.global_loss_weight

        self.temp1 = self.cfg.model.gloria.temp1
        self.temp2 = self.cfg.model.gloria.temp2
        self.temp3 = self.cfg.model.gloria.temp3
        self.batch_size = self.cfg.train.batch_size

        self.tokenizer = BertTokenizer.from_pretrained(self.cfg.model.text.bert_type, cache_dir = self.cfg.model.text.cache_dir, local_files_only=True)
        self.ixtoword = {v: k for k, v in self.tokenizer.get_vocab().items()}
        self.soft_label = self.cfg.model.gloria.soft_label

        if self.soft_label:
            self.topk = self.cfg.model.gloria.topk
            self.threshold = [self.cfg.model.gloria.threshold,self.cfg.model.gloria.threshold1]
            self.tool_bert = BertModel.from_pretrained(self.cfg.model.text.bert_type, cache_dir = self.cfg.model.text.cache_dir, local_files_only=True)

    def text_encoder_forward(self, caption_ids, attention_mask, token_type_ids):
        text_emb_l, text_emb_g, sents = self.text_encoder(
            caption_ids, attention_mask, token_type_ids
        )
        return text_emb_l, text_emb_g, sents

    def image_encoder_forward(self, imgs):
        img_feat_g, img_emb_l = self.img_encoder(imgs, get_local=True)
        img_emb_g, img_emb_l = self.img_encoder.generate_embeddings(
            img_feat_g, img_emb_l
        )

        return img_emb_l, img_emb_g #b,768,19,19 ; b,768

    def _calc_local_loss(self, img_emb_l, text_emb_l, sents, idx, probs):

        cap_lens = [
            len([w for w in sent if not w.startswith("[")]) + 1 for sent in sents
        ]
        l_loss0, l_loss1, attn_maps = self.local_loss(
            img_emb_l,
            text_emb_l,
            cap_lens,
            temp1=self.temp1,
            temp2=self.temp2,
            temp3=self.temp3,
            idx = idx,
            probs = probs
        )
        return l_loss0, l_loss1, attn_maps

    def _calc_global_loss(self, img_emb_g, text_emb_g, idx, probs):
        g_loss0, g_loss1 = self.global_loss(img_emb_g, text_emb_g, temp3=self.temp3, idx=idx, probs=probs)
        return g_loss0, g_loss1

    def calc_loss(self, img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, idx, probs):

        l_loss0, l_loss1, attn_maps = self._calc_local_loss(
            img_emb_l, text_emb_l, sents, idx, probs
        )
        g_loss0, g_loss1 = self._calc_global_loss(img_emb_g, text_emb_g, idx, probs)

        # weighted loss
        loss = 0
        loss += (l_loss0 + l_loss1) * self.local_loss_weight
        loss += (g_loss0 + g_loss1) * self.global_loss_weight

        return loss, attn_maps

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


    def sum_pooling(self,model_output,attention_mask):
        """
        Sum Pooling - Take attention mask into account for correct averaging
        Reference: https://www.sbert.net/docs/usage/computing_sentence_embeddings.html

        """
        token_embeddings = model_output  

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
       
        return sum_embeddings

    def token_pooling(self,model_output):
        return model_output[:,0]

    def max_pooling(self, model_output, attention_mask):
        
        token_embeddings = model_output

        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        token_embeddings[input_mask_expanded == 0] = -1e9

        return torch.max(token_embeddings, 1)[0]


    def getTextSoftTarget(self, raw_txt, topK, threshold):

        self.tool_bert.eval()
        with torch.no_grad():
            f_txt = self.tool_bert(**raw_txt)

            # using mean features  
            # f_txt = self.max_pooling(f_txt[0], raw_txt['attention_mask'])
            f_txt = self.token_pooling(f_txt[0])
            # f_txt = self.mean_pooling(f_txt[0], raw_txt['attention_mask'])

            f_txt = F.normalize(f_txt, p=2, dim=1)
            batch_size, d_k = f_txt.shape
            scores = torch.matmul(f_txt,f_txt.transpose(-2,-1)) 
            # val, idx = torch.topk(scores,topK,dim=-1)
            # filter = torch.masked_fill(val, val<threshold, 0)

            # return (idx, filter)
            return (scores, threshold)


    def forward(self, img, txt, train=True):

        # img encoder branch
        idx = None 
        filter = None
        if self.soft_label and train:
            idx, filter = self.getTextSoftTarget(txt, self.topk, self.threshold)


        img_emb_l, img_emb_g = self.image_encoder_forward(img)

        # text encorder branch
        text_emb_l, text_emb_g, sents = self.text_encoder_forward(
            txt["input_ids"], txt["attention_mask"], txt["token_type_ids"]
        )

        return img_emb_l, img_emb_g, text_emb_l, text_emb_g, sents, idx, filter

    def get_global_similarities(self, img_emb_g, text_emb_g):
        img_emb_g = img_emb_g.numpy()
        text_emb_g = text_emb_g.numpy()
        global_similarities = metrics.pairwise.cosine_similarity(img_emb_g, text_emb_g)
        global_similarities = torch.Tensor(global_similarities)
        return global_similarities

    def get_local_similarities(self, img_emb_l, text_emb_l, cap_lens):

        batch_size = img_emb_l.shape[0]
        similarities = []
        for i in range(len(text_emb_l)):
            words_num = cap_lens[i]
            text_emb = text_emb_l[i]
            word = (
                text_emb[:, 1 : words_num + 1].unsqueeze(0).contiguous()
            )  # [1, 768, 25]

            word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
            context = img_emb_l  # [48, 768, 19, 19]

            weiContext, attn = attention_fn(
                word, context, 4.0
            )  # [48, 768, 25], [48, 25, 19, 19]

            word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
            weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

            word = word.view(batch_size * words_num, -1)  # [1200, 768]
            weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]
            #
            row_sim = cosine_similarity(word, weiContext)
            row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

            row_sim.mul_(5.0).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)

            row_sim = torch.log(row_sim)

            similarities.append(row_sim)

        local_similarities = torch.cat(similarities, 1).cpu()

        return local_similarities

    def get_attn_maps(self, img_emb_l, text_emb_l, sents):
        _, _, attn_maps = self._calc_local_loss(img_emb_l, text_emb_l, sents)
        return attn_maps

    # def plot_attn_maps(self, attn_maps, imgs, sents, epoch_idx=0, batch_idx=0):

    #     img_set, _ = utils.build_attention_images(
    #         imgs,
    #         attn_maps,
    #         max_word_num=self.cfg.data.text.word_num,
    #         nvis=self.cfg.train.nvis,
    #         rand_vis=self.cfg.train.rand_vis,
    #         sentences=sents,
    #     )

    #     if img_set is not None:
    #         im = Image.fromarray(img_set)
    #         fullpath = (
    #             f"{self.cfg.output_dir}/"
    #             f"attention_maps_epoch{epoch_idx}_"
    #             f"{batch_idx}.png"
    #         )
    #         im.save(fullpath)

    def process_class_prompts(self, class_prompts, device):

        cls_2_processed_txt = {}
        for k, v in class_prompts.items():
            cls_2_processed_txt[k] = self.process_text(v, device)

        return cls_2_processed_txt

    def cross_modal_retrieval(self, img_emb_l, img_emb_g, text_emb_l, text_emb_g, cap_lens):

        print('retrieval....')
        global_similarities = self.get_global_similarities(img_emb_g, text_emb_g)
        local_similarities = self.get_local_similarities(img_emb_l, text_emb_l, cap_lens)
        
        norm = lambda x: (x - x.mean(axis=0)) / (x.std(axis=0))
        similarities = np.stack(
            [norm(local_similarities), norm(global_similarities)]
        )
        # similarities = np.stack([local_similarities, global_similarities])
        similarities = similarities.mean(axis=0)
        batch_size = len(similarities)
        idx = np.array(range(batch_size)).reshape(-1,1)


        # similarites matrix: img x txt
        # i2t
        i2t_index_list = np.argsort(similarities)[:,::-1]
        i2t_ranks = np.where(idx == i2t_index_list)[1]
        i2t_ranks = np.array(i2t_ranks)
        
        i2t_r1 = 100.0 * len(np.where(i2t_ranks < 1)[0]) / len(i2t_ranks)
        i2t_r5 = 100.0 * len(np.where(i2t_ranks < 5)[0]) / len(i2t_ranks)
        i2t_r10 = 100.0 * len(np.where(i2t_ranks < 10)[0]) / len(i2t_ranks)
        i2t_r50 = 100.0 * len(np.where(i2t_ranks < 50)[0]) / len(i2t_ranks)
        i2t_r100 = 100.0 * len(np.where(i2t_ranks < 100)[0]) / len(i2t_ranks)
        i2t_medr = np.floor(np.median(i2t_ranks)) + 1
        i2t_meanr = i2t_ranks.mean() + 1

        # t2i
        t_similarities = similarities.T
        t2i_index_list = np.argsort(t_similarities)[:,::-1]
        t2i_ranks = np.where(idx == t2i_index_list)[1]
        t2i_ranks = np.array(t2i_ranks)
        
        t2i_r1 = 100.0 * len(np.where(t2i_ranks < 1)[0]) / len(t2i_ranks)
        t2i_r5 = 100.0 * len(np.where(t2i_ranks < 5)[0]) / len(t2i_ranks)
        t2i_r10 = 100.0 * len(np.where(t2i_ranks < 10)[0]) / len(t2i_ranks)
        t2i_r50 = 100.0 * len(np.where(t2i_ranks < 50)[0]) / len(t2i_ranks)
        t2i_r100 = 100.0 * len(np.where(t2i_ranks < 100)[0]) / len(t2i_ranks)
        t2i_medr = np.floor(np.median(t2i_ranks)) + 1
        t2i_meanr = t2i_ranks.mean() + 1

        return i2t_r1,i2t_r5,i2t_r10,i2t_r50,i2t_r100,i2t_medr,i2t_meanr,t2i_r1,t2i_r5,t2i_r10,t2i_r50,t2i_r100,t2i_medr,t2i_meanr








