# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import math
from semilearn.algorithms.utils import concat_all_gather
from semilearn.algorithms.hooks import MaskingHook


class FreeMatchThresholingHook(MaskingHook):
    """
    SAT in FreeMatch
    """
    def __init__(self, num_classes, momentum=0.999, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.m = momentum
        # self.relation_matrix = relation_matrix
        self.mod_relation = None
        self.p_model = torch.ones((self.num_classes)) # / self.num_classes
        self.label_hist = torch.ones((self.num_classes)) # / self.num_classes
        self.time_p = self.p_model.mean()
        self.max_ambiguity = self.get_max_ambiguity(num_classes)
    
    @torch.no_grad()
    def update(self, algorithm, probs_x_ulb, relation_matrix):
        if algorithm.distributed and algorithm.world_size > 1:
            probs_x_ulb = concat_all_gather(probs_x_ulb)
        max_probs, max_idx = torch.max(probs_x_ulb, dim=-1,keepdim=True)

        if algorithm.use_quantile:
            self.time_p = self.time_p * self.m + (1 - self.m) * torch.quantile(max_probs,0.8) #* max_probs.mean()
        else:
            self.time_p = self.time_p * self.m + (1 - self.m) * max_probs.mean()
        
        if algorithm.clip_thresh:
            self.time_p = torch.clip(self.time_p, 0.8, 0.95)
            self.p_model = torch.clip(self.p_model, 0.6, 1)

        self.p_model = self.p_model * self.m + (1 - self.m) * probs_x_ulb.mean(dim=0)
        hist = torch.bincount(max_idx.reshape(-1), minlength=self.p_model.shape[0]).to(self.p_model.dtype) 
        self.label_hist = self.label_hist * self.m + (1 - self.m) * (hist / hist.sum())
        self.mod_relation = torch.diag(relation_matrix)

        algorithm.p_model = self.p_model 
        algorithm.label_hist = self.label_hist 
        algorithm.time_p = self.time_p 
        
    
    def get_max_ambiguity(self, num_class):
        uniform_distribution = torch.ones((1, num_class)) / num_class
        max_ambiguity = -uniform_distribution * torch.log(uniform_distribution+1e-7)
        return max_ambiguity.sum(1)
    
    def get_time_p(self, algorithm):
        return self.time_p

    @torch.no_grad()
    def masking(self, algorithm, logits_x_ulb, relation_matrix, softmax_x_ulb=True, *args, **kwargs):
        if not self.p_model.is_cuda:
            self.p_model = self.p_model.to(logits_x_ulb.device)
        if not self.label_hist.is_cuda:
            self.label_hist = self.label_hist.to(logits_x_ulb.device)
        if not self.time_p.is_cuda:
            self.time_p = self.time_p.to(logits_x_ulb.device)
        if not self.max_ambiguity.is_cuda:
            self.max_ambiguity = self.max_ambiguity.to(logits_x_ulb.device)

        if softmax_x_ulb:
            probs_x_ulb = torch.softmax(logits_x_ulb.detach(), dim=-1)
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.detach()

        self.update(algorithm, probs_x_ulb, relation_matrix)

        class_num = relation_matrix.shape[0]
        top_class_num = math.floor(class_num / 2)
        
        sample_level_bias = 0
        ambiguity = -probs_x_ulb * torch.log(probs_x_ulb+1e-7)
        ambiguity = ambiguity.sum(1) + sample_level_bias
        
        top_related_class = torch.topk(relation_matrix, top_class_num, dim=-1)[1]
        pred_ulb_labels = torch.max(probs_x_ulb, dim=-1)[1]
        top_related_idx = top_related_class[pred_ulb_labels]
        # 跑起来再查一下这里到底对不对
        prob_top_related = torch.gather(probs_x_ulb, 1, top_related_idx).sum(dim=-1)
        
        # sample_level_coff = 1 / (ambiguity * prob_top_related)
        coff = torch.exp(-2 * (prob_top_related - 0.3)).to(logits_x_ulb.device)
        sample_level_coff = coff + (1-coff) * ambiguity / self.max_ambiguity
        sample_level_coff[sample_level_coff > 1] = 1

        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        mod = self.p_model / torch.max(self.p_model, dim=-1)[0]
        # mod_relation = self.mod_relation / torch.max(self.mod_relation, dim=-1)[0]
        min_thres = 0.5
        if self.every_n_iters(algorithm, algorithm.num_log_iter):
            algorithm.print_fn('The Threshold of global, class-level, max and min sample-level are {}, {}, {}, {}'.format(self.time_p, mod, torch.max(sample_level_coff), torch.min(sample_level_coff)))
            algorithm.print_fn('The confusion matrix is {}'.format(relation_matrix.detach().data))
        # mask = max_probs.ge(min_thres + (1-min_thres) * self.time_p * mod[max_idx] * sample_level_coff).to(max_probs.dtype)
        mask = max_probs.ge(min_thres + self.time_p * (mod[max_idx] * 0.25 + sample_level_coff * 0.25)).to(max_probs.dtype)
        return mask
