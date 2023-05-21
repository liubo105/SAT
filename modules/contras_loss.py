import torch
import torch.nn.functional as F

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, temperature, alpha_weight):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.alpha_weight = alpha_weight
        self.device =device



    def softXEnt(self, target, logits):
        """
        From the pytorch discussion Forum:
        https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501 
        """
        logprobs = torch.nn.functional.log_softmax(logits, dim = -1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def forward(self, zis, zjs,
                    norm=True):
        temperature = self.temperature
        """
        Pytorch implementation of the loss  SimCRL function by googleresearch: https://github.com/google-research/simclr
        @article{chen2020simple,
                title={A Simple Framework for Contrastive Learning of Visual Representations},
                author={Chen, Ting and Kornblith, Simon and Norouzi, Mohammad and Hinton, Geoffrey},
                journal={arXiv preprint arXiv:2002.05709},
                year={2020}
                }
        @article{chen2020big,
                title={Big Self-Supervised Models are Strong Semi-Supervised Learners},
                author={Chen, Ting and Kornblith, Simon and Swersky, Kevin and Norouzi, Mohammad and Hinton, Geoffrey},
                journal={arXiv preprint arXiv:2006.10029},
                year={2020}
                }
        """
        """Compute loss for model.
        Args:
        hidden: hidden vector (`Tensor`) of shape (2 * bsz, dim).
        hidden_norm: whether or not to use normalization on the hidden vector.
        temperature: a `floating` number for temperature scaling.
        tpu_context: context information for tpu.
        weights: a weighting number or vector.
        Returns:
        A loss scalar.
        The logits for contrastive prediction task.
        The labels for contrastive prediction task.
        """
        # Get (normalized) hidden1 and hidden2.
        if norm:
            zis = F.normalize(zis, p=2, dim=1)
            zjs = F.normalize(zjs, p=2, dim=1)
            

        hidden1, hidden2 = zis, zjs
        batch_size = hidden1.shape[0]

        hidden1_large = hidden1
        hidden2_large = hidden2

        labels = F.one_hot(torch.arange(start=0, end=batch_size, dtype=torch.int64), num_classes=batch_size).float()
        labels = labels.cuda(self.device)
        
        """
        Different from Image-Image contrastive learning
        In the case of Image-Text contrastive learning we do not compute the similarity function between the Image-Image and Text-Text pairs  
        """
        # logits_aa = torch.matmul(hidden1, torch.transpose(hidden1_large,0, 1)) / temperature
        # logits_aa = logits_aa - masks * LARGE_NUM
        # logits_bb = torch.matmul(hidden2,  torch.transpose(hidden2_large,0, 1)) / temperature
        # logits_bb = logits_bb - masks * LARGE_NUM
        logits_ab = torch.matmul(hidden1, torch.transpose(hidden2_large,0, 1)) / temperature
        logits_ba = torch.matmul(hidden2, torch.transpose(hidden1_large,0, 1)) / temperature

        loss_a = self.softXEnt(labels, logits_ab)
        loss_b = self.softXEnt(labels, logits_ba)

        return loss_a + loss_b



class HierarchicalLoss(torch.nn.Module):

    def __init__(self, temperature, alpha_weight):
        super(HierarchicalLoss, self).__init__()
        self.temperature = temperature
        self.alpha_weight = alpha_weight


    def softXEntPenalty(self, target, logits,penalty):
        """
        From the pytorch discussion Forum:
        https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501 
        """
        logprobs = torch.nn.functional.log_softmax(logits, dim = -1)
        loss = -(target * logprobs * penalty).sum() / logits.shape[0]
        return loss

    def softXEnt(self, target, logits):
        """
        From the pytorch discussion Forum:
        https://discuss.pytorch.org/t/soft-cross-entropy-loss-tf-has-it-does-pytorch-have-it/69501 
        """
        logprobs = torch.nn.functional.log_softmax(logits, dim = -1)
        loss = -(target * logprobs).sum() / logits.shape[0]
        return loss

    def forward(self, f_img, f_txt, labels):
        temperature = self.temperature
        alpha = self.alpha_weight
        loss = 0
        
        f_img = F.normalize(f_img, p=2, dim=1)
        f_txt = F.normalize(f_txt, p=2, dim=1)

        logits_i2t = torch.matmul(f_img, torch.transpose(f_txt,0, 1)) / temperature
        logits_t2i = torch.matmul(f_txt, torch.transpose(f_img,0, 1)) / temperature

        h, w = logits_i2t.shape # batch x batch


        # #loss threshold
        scores, threshold = labels
        threshold1, threshold2 = threshold

        for layer,i in enumerate(scores):
            pos  = (i>threshold1).nonzero().squeeze(-1)
            neg = (i<=threshold2).nonzero().squeeze(-1)
            neg_i2t = logits_i2t[layer][neg]
            neg_t2i = logits_t2i[layer][neg]
            loss_sep = 0
            for j in pos:
                pos_i2t = logits_i2t[layer][j].unsqueeze(-1)
                pos_t2i = logits_t2i[layer][j].unsqueeze(-1)
                new_i2t = torch.cat([pos_i2t,neg_i2t])
                new_t2i = torch.cat([pos_t2i,neg_t2i])
                targets = torch.zeros_like(new_i2t,dtype=torch.long)
                targets[0] = 1
                loss_i2t = self.softXEnt(targets,new_i2t)
                loss_t2i = self.softXEnt(targets,new_t2i)
                loss_sep += alpha*loss_i2t + (1-alpha)*loss_t2i
            loss_sep /= len(pos)
            loss += loss_sep

        loss /= h

        return loss


