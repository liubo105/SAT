"""
Adapted from: https://github.com/mrlibw/ControlGAN
"""

import torch
import torch.nn as nn

from torch.autograd import Variable


def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()


def attention_fn(query, context, temp1):
    """
    query: batch x ndf x queryL / word  [48, 768, 25]
    context: batch x ndf x ih x iw (sourceL=ihxiw) / img [48, 768, 19, 19]
    mask: batch_size x sourceL
    """
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous() # 48，19*19， 768

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query)
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax(dim=-1)(attn)

    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)

    attn = attn * temp1
    attn = nn.Softmax(dim=-1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)

    return weightedContext, attn.view(batch_size, -1, ih, iw)


def softXEnt(target, logits):
    """
    From the pytorch discussion Forum:
    https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501 
    """
    logprobs = torch.nn.functional.log_softmax(logits, dim = -1)
    loss = -(target * logprobs).sum() / logits.shape[0]
    return loss

def softXEntPenalty(target, logits,penalty):
        """
        From the pytorch discussion Forum:
        https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501 
        """
        logprobs = torch.nn.functional.log_softmax(logits, dim = -1)
        loss = -(target * logprobs * penalty).sum() / logits.shape[0]
        return loss

def global_loss(cnn_code, rnn_code, eps=1e-8, temp3=10.0, idx=None, probs=None):

    batch_size = cnn_code.shape[0]

    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * temp3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    scores1 = scores0.transpose(0, 1)

    if idx is None:
        print('normal gloria global')
        labels = Variable(torch.LongTensor(range(batch_size))).to(scores0.device)
        loss0 = nn.CrossEntropyLoss()(scores0, labels)  # labels: arange(batch_size)
        loss1 = nn.CrossEntropyLoss()(scores1, labels)

    else:
        print('soft gloria global')
        loss0 = 0
        loss1 = 0
        scores = idx
        threshold1, threshold2 = probs

        for layer,i in enumerate(scores):
            pos  = (i>threshold1).nonzero().squeeze(-1)
            neg = (i<=threshold2).nonzero().squeeze(-1)
            neg_i2t = scores0[layer][neg]
            neg_t2i = scores1[layer][neg]
            loss_i2t = 0
            loss_t2i = 0

            for j in pos:
                pos_i2t = scores0[layer][j].unsqueeze(-1)
                pos_t2i = scores1[layer][j].unsqueeze(-1)
                new_i2t = torch.cat([pos_i2t,neg_i2t])
                new_t2i = torch.cat([pos_t2i,neg_t2i])
                targets = torch.zeros_like(new_i2t,dtype=torch.long)
                targets[0] = 1
                loss_i2t += softXEnt(targets,new_i2t)
                loss_t2i += softXEnt(targets,new_t2i)
    
            loss_i2t /= len(pos)
            loss_t2i /= len(pos)
            loss0 += loss_i2t
            loss1 += loss_t2i

        loss0 /= batch_size
        loss1 /= batch_size

        # dynamic filtering
        # neg_idx = torch.ones((batch_size, batch_size),dtype=torch.bool).cuda()
        # neg_idx = neg_idx.scatter(1,idx,False)
        # neg_i2t = scores0[neg_idx].reshape(batch_size,-1)
        # neg_t2i = scores1[neg_idx].reshape(batch_size,-1)
        # loss0 = 0
        # loss1 = 0
            
        # # for i in range(idx.shape[1]):
        # for i in range(1):

        #     sub_idx = idx[:,i].reshape(batch_size,-1)
        #     penalty = probs[:,i].reshape(batch_size,-1)
        #     pos_idx = torch.zeros((batch_size, batch_size),dtype=torch.bool).cuda()
        #     pos_idx = pos_idx.scatter(1,sub_idx,True)
        #     pos_i2t = scores0[pos_idx].reshape(batch_size,-1)
        #     new_logits_i2t = torch.cat([pos_i2t,neg_i2t],dim=1)
        #     targets = torch.zeros_like(new_logits_i2t,dtype=torch.long)
        #     targets[:,0] = 1
        #     loss_i2t = softXEntPenalty(targets,new_logits_i2t,penalty)
        #     # loss_i2t = softXEnt(targets,new_logits_i2t)
            
        #     pos_t2i = scores1[pos_idx].reshape(batch_size,-1)
        #     new_logits_t2i = torch.cat([pos_t2i,neg_t2i],dim=1)
        #     loss_t2i = softXEntPenalty(targets,new_logits_t2i,penalty)
        #     # loss_t2i = softXEnt(targets,new_logits_t2i)

        #     loss0 += loss_i2t 
        #     loss1 += loss_t2i

    return loss0, loss1


def local_loss(
    img_features, words_emb, cap_lens, temp1=4.0, temp2=5.0, temp3=10.0, agg="sum", idx=None, probs=None
):

   
    batch_size = img_features.shape[0]

    att_maps = []
    similarities = []
    # cap_lens = cap_lens.data.tolist()
    for i in range(words_emb.shape[0]):

        # Get the i-th text description in current mini-batch
        words_num = cap_lens[i]  # 25
        # TODO: remove [SEP]
        # word = words_emb[i, :, 1:words_num+1].unsqueeze(0).contiguous()    # [1, 768, 25]
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 768, 25]
        word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
        context = img_features  # [48, 768, 19, 19]

        weiContext, attn = attention_fn(
            word, context, temp1
        )  # [48, 768, 25], [48, 25, 19, 19]

        att_maps.append(
            attn[i].unsqueeze(0).contiguous()
        )  # add attention for curr index  [25, 19, 19]
        word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
        weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

        word = word.view(batch_size * words_num, -1)  # [1200, 768]
        weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]

        row_sim = cosine_similarity(word, weiContext)
        row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

        row_sim.mul_(temp2).exp_()
        if agg == "sum":
            row_sim = row_sim.sum(dim=1, keepdim=True)  # [48, 1]
        else:
            row_sim = row_sim.mean(dim=1, keepdim=True)  # [48, 1]
        row_sim = torch.log(row_sim)

        similarities.append(row_sim)

    similarities = torch.cat(similarities, 1)  #
    similarities = similarities * temp3
    similarities1 = similarities.transpose(0, 1)  # [48, 48]

    if idx is None:
        print('normal gloria local')
        labels = Variable(torch.LongTensor(range(batch_size))).to(similarities.device)
        loss0 = nn.CrossEntropyLoss()(similarities, labels)  # labels: arange(batch_size)
        loss1 = nn.CrossEntropyLoss()(similarities1, labels)

    else:
        print('soft gloria local')
        loss0 = 0
        loss1 = 0
        scores = idx
        threshold1, threshold2 = probs

        for layer,i in enumerate(scores):
            pos  = (i>threshold1).nonzero().squeeze(-1)
            neg = (i<=threshold2).nonzero().squeeze(-1)
            neg_i2t = similarities[layer][neg]
            neg_t2i = similarities1[layer][neg]
            loss_i2t = 0
            loss_t2i = 0

            for j in pos:
                pos_i2t = similarities[layer][j].unsqueeze(-1)
                pos_t2i = similarities1[layer][j].unsqueeze(-1)
                new_i2t = torch.cat([pos_i2t,neg_i2t])
                new_t2i = torch.cat([pos_t2i,neg_t2i])
                targets = torch.zeros_like(new_i2t,dtype=torch.long)
                targets[0] = 1
                loss_i2t += softXEnt(targets,new_i2t)
                loss_t2i += softXEnt(targets,new_t2i)
    
            loss_i2t /= len(pos)
            loss_t2i /= len(pos)
            loss0 += loss_i2t
            loss1 += loss_t2i

        loss0 /= batch_size
        loss1 /= batch_size

        # neg_idx = torch.ones((batch_size, batch_size),dtype=torch.bool).cuda()
        # neg_idx = neg_idx.scatter(1,idx,False)
        # neg_i2t = similarities[neg_idx].reshape(batch_size,-1)
        # neg_t2i = similarities1[neg_idx].reshape(batch_size,-1)
        # loss0 = 0
        # loss1 = 0
            
        # # for i in range(idx.shape[1]):
        # for i in range(1):

        #     sub_idx = idx[:,i].reshape(batch_size,-1)
        #     penalty = probs[:,i].reshape(batch_size,-1)
        #     pos_idx = torch.zeros((batch_size, batch_size),dtype=torch.bool).cuda()
        #     pos_idx = pos_idx.scatter(1,sub_idx,True)
        #     pos_i2t = similarities[pos_idx].reshape(batch_size,-1)
        #     new_logits_i2t = torch.cat([pos_i2t,neg_i2t],dim=1)
        #     targets = torch.zeros_like(new_logits_i2t,dtype=torch.long)
        #     targets[:,0] = 1
        #     loss_i2t = softXEntPenalty(targets,new_logits_i2t,penalty)
        #     # loss_i2t = softXEnt(targets,new_logits_i2t)
            
        #     pos_t2i = similarities1[pos_idx].reshape(batch_size,-1)
        #     new_logits_t2i = torch.cat([pos_t2i,neg_t2i],dim=1)
        #     loss_t2i = softXEntPenalty(targets,new_logits_t2i,penalty)
        #     # loss_t2i = softXEnt(targets,new_logits_t2i)

        #     loss0 += loss_i2t 
        #     loss1 += loss_t2i





    return loss0, loss1, att_maps
