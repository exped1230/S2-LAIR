import torch
import torch.nn.functional as F
import torch.nn as nn
from .utils import FreeMatchThresholingHook
from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
from torch.distributions.normal import Normal
from diffusion.utils import *
from diffusion.ddim import *
import numpy as np
import time

# TODO: move these to .utils or algorithms.utils.loss
def replace_inf_to_zero(val):
    val[val == float('inf')] = 0.0
    return val

def entropy_loss(mask, logits_s, prob_model, label_hist):
    mask = mask.bool()

    # select samples
    logits_s = logits_s[mask]

    prob_s = logits_s.softmax(dim=-1)
    _, pred_label_s = torch.max(prob_s, dim=-1)

    hist_s = torch.bincount(pred_label_s, minlength=logits_s.shape[1]).to(logits_s.dtype)
    hist_s = hist_s / hist_s.sum()

    # modulate prob model 
    prob_model = prob_model.reshape(1, -1)
    label_hist = label_hist.reshape(1, -1)
    # prob_model_scaler = torch.nan_to_num(1 / label_hist, nan=0.0, posinf=0.0, neginf=0.0).detach()
    prob_model_scaler = replace_inf_to_zero(1 / label_hist).detach()
    mod_prob_model = prob_model * prob_model_scaler
    mod_prob_model = mod_prob_model / mod_prob_model.sum(dim=-1, keepdim=True)

    # modulate mean prob
    mean_prob_scaler_s = replace_inf_to_zero(1 / hist_s).detach()
    # mean_prob_scaler_s = torch.nan_to_num(1 / hist_s, nan=0.0, posinf=0.0, neginf=0.0).detach()
    mod_mean_prob_s = prob_s.mean(dim=0, keepdim=True) * mean_prob_scaler_s
    mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum(dim=-1, keepdim=True)

    loss = mod_prob_model * torch.log(mod_mean_prob_s + 1e-12)
    loss = loss.sum(dim=1)
    return loss.mean(), hist_s.mean()


# class Diffusion_Linear(nn.Module):
#     def __init__(self, num_in, num_out, n_steps):
#         super(Diffusion_Linear, self).__init__()
#         self.num_out = num_out
#         self.lin = nn.Linear(num_in, num_out)
#         self.embed = nn.Embedding(n_steps, num_out)
#         self.embed.weight.data.uniform_()

#     def forward(self, x, t):
#         out = self.lin(x)
#         gamma = self.embed(t)
#         out = gamma.view(-1, self.num_out) * out
#         return out

class CoMatch_Net(nn.Module):
    def __init__(self, base, proj_size=128):
        super(CoMatch_Net, self).__init__()
        self.backbone = base
        self.num_features = base.num_features
        
        self.mlp_proj = nn.Sequential(*[
            nn.Linear(self.num_features, self.num_features),
            nn.ReLU(inplace=False),
            nn.Linear(self.num_features, proj_size)
        ])
        
    def l2norm(self, x, power=2):
        norm = x.pow(power).sum(1, keepdim=True).pow(1. / power)
        out = x.div(norm)
        return out
    
    def forward(self, x, **kwargs):
        feat = self.backbone(x, only_feat=True)
        logits = self.backbone(feat, only_fc=True)
        feat_proj = self.l2norm(self.mlp_proj(feat))
        return {'logits':logits, 'feat':feat_proj}

    def group_matcher(self, coarse=False):
        matcher = self.backbone.group_matcher(coarse, prefix='backbone.')
        return matcher

# class Diffusion_model(nn.Module):
#     def __init__(self, proj_size=64, num_class=8, diffusion_feature_dim=512, num_timesteps=1000):
#         super(Diffusion_model, self).__init__()
#         self.diffusion_linear1 = Diffusion_Linear(proj_size, diffusion_feature_dim, num_timesteps)
#         self.unetnorm1 = nn.BatchNorm1d(diffusion_feature_dim)
#         self.diffusion_linear2 = Diffusion_Linear(diffusion_feature_dim, diffusion_feature_dim, num_timesteps)
#         self.unetnorm2 = nn.BatchNorm1d(diffusion_feature_dim)
#         self.diffusion_linear3 = Diffusion_Linear(diffusion_feature_dim, diffusion_feature_dim, num_timesteps)
#         self.unetnorm3 = nn.BatchNorm1d(diffusion_feature_dim)
#         self.linear4 = nn.Linear(diffusion_feature_dim, num_class*2)

#     def forward(self, x, t=None):
#             y = self.diffusion_linear1(x, t)
#             y = self.unetnorm1(y)
#             y = F.softplus(y)
#             y = self.diffusion_linear2(y, t)
#             y = self.unetnorm2(y)
#             y = F.softplus(y)
#             y = self.diffusion_linear3(y, t)
#             y = self.unetnorm3(y)
#             y = F.softplus(y)
#             return {'noise':self.linear4(y)}




@ALGORITHMS.register('main_retain')
class Main(AlgorithmBase):
    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger) 
        self.init(T=args.T, hard_label=args.hard_label, ema_p=args.ema_p, use_quantile=args.use_quantile, clip_thresh=args.clip_thresh)
        self.lambda_e = args.ent_loss_ratio
        
        self.prototype = torch.zeros(self.num_classes, args.proj_size)
        self.prototype = self.prototype.to(args.gpu)
        self.relation_matrix = torch.zeros(self.num_classes, self.num_classes).fill_(1/self.num_classes)
        self.relation_matrix = self.relation_matrix.to(args.gpu)
        self.k = args.k
        self.update_m = args.update_m       

        self.device = args.gpu
        
        self.ulb_pseudo_label_acc = 0.0
        self.pseudo_class_util_ratio = [0] * self.num_classes
        self.real_class_util_ratio = [0] * self.num_classes
        self.class_ulb_pseudo_label_acc = [0] * self.num_classes
        self.class_ulb_pseudo_label_precision = [0] * self.num_classes
        

    def init(self, T, hard_label=True, ema_p=0.999, use_quantile=True, clip_thresh=False):
        self.T = T
        self.use_hard_label = hard_label
        self.ema_p = ema_p
        self.use_quantile = use_quantile
        self.clip_thresh = clip_thresh

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FreeMatchThresholingHook(num_classes=self.num_classes, momentum=self.args.ema_p), "MaskingHook")
        super().set_hooks()

    def set_model(self):
        model = super().set_model()
        # 后面看继续用CoMatch_Net还是直接用回ResNet50
        model = CoMatch_Net(model, proj_size=self.args.proj_size)
        return model

    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = CoMatch_Net(ema_model, proj_size=self.args.proj_size)
        ema_model.load_state_dict(self.check_prefix_state_dict(self.model.state_dict()))
        return ema_model
    
    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s, y_ulb):
        num_lb = y_lb.shape[0]

        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb) 
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb':feats_x_lb, 'x_ulb_w':feats_x_ulb_w, 'x_ulb_s':feats_x_ulb_s}
            
            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            # calculate mask 这里根据阈值生成mask的hook要重写一下
            mask = self.call_hook("masking", "MaskingHook", relation_matrix=self.relation_matrix, logits_x_ulb=logits_x_ulb_w)

            # generate unlabeled targets using pseudo label hook 生成伪标签的重写一下
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=logits_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T)
            
            with torch.no_grad():
                features_lb = feats_x_lb.detach()
                feature_ulb_w = feats_x_ulb_w.detach()

                lb_probs = torch.softmax(logits_x_lb, dim=1)
                _, lb_guess = torch.max(lb_probs, dim=1)

                for cur_class in range(self.num_classes):
                    class_lb_mask = (y_lb == cur_class) & (lb_guess == y_lb)
                    class_ulb_mask = (pseudo_label == cur_class) & (mask == 1)
                    # if (class_lb_mask.sum() + class_ulb_mask.sum()) == 0:
                    #     continue
                    # 可以试一下要不要把达到置信度的无标签样本加进来
                    lambda_prototype = 0.9
                    if class_lb_mask.sum() != 0:
                        cur_class_feature = features_lb[class_lb_mask].sum(0) / class_lb_mask.sum()
                        self.prototype[cur_class] = lambda_prototype * self.prototype[cur_class] + (1-lambda_prototype) * cur_class_feature
                    # 可以试一下要不要把达到置信度的无标签样本加进来
                    # if class_ulb_mask.sum() != 0:
                    #     cur_class_feature = feature_ulb_w[class_ulb_mask].sum(0) / class_ulb_mask.sum()
                    #     self.prototype[cur_class] = lambda_prototype * self.prototype[cur_class] + (1-lambda_prototype) * cur_class_feature
                relation_T = 0.2
                if self.update_m == 'L2':
                    L2_dis = torch.norm(self.prototype[:, None].repeat(1, self.num_classes, 1) - self.prototype[None, :].repeat(self.num_classes, 1, 1), dim=2, p=2) / relation_T
                    new_relation_matrix = torch.exp(-L2_dis)
                elif self.update_m == 'L1':
                    L1_dis = torch.sum(self.prototype[:, None].repeat(1, self.num_classes, 1) - self.prototype[None, :].repeat(self.num_classes, 1, 1), dim=2) / relation_T
                    new_relation_matrix = torch.exp(-L1_dis)
                elif self.update_m == 'cos':
                    prototype_tmp = torch.norm(self.prototype, p=2, dim=1, keepdim=True).expand_as(self.prototype) + 1e-12
                    prototype_tmp = self.prototype / prototype_tmp
                    # 跑起来以后还得查一下cos_sim，之前对角线不是明显最大的过于不合理
                    cos_sim = torch.mm(prototype_tmp, prototype_tmp.T) / relation_T
                    for i in range(self.num_classes): cos_sim[i, i] = 1 / relation_T
                    new_relation_matrix = torch.exp(cos_sim)

                lambda_relation_matrix = 0.9
                new_relation_matrix = new_relation_matrix / torch.sum(new_relation_matrix, dim=1, keepdim=True).expand_as(new_relation_matrix)
                self.relation_matrix = lambda_relation_matrix * self.relation_matrix + (1-lambda_relation_matrix) * new_relation_matrix
            
            unsup_loss = self.consistency_loss(logits_x_ulb_s, pseudo_label, 'ce', mask=mask)
            
            # calculate entropy loss, 后面可以在消融的几个数据集上再测一下加这个到底有没有用
            if mask.sum() > 0:
               ent_loss, _ = entropy_loss(mask, logits_x_ulb_s, self.p_model, self.label_hist)
            else:
               ent_loss = 0.0
            total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_e * ent_loss
            # print('total_loss:', total_loss)
            
            self.ulb_pseudo_label_acc = ((pseudo_label == y_ulb) * mask).sum() / (mask.sum() + 1e-8)
            for cur_class in range(self.num_classes):
                self.pseudo_class_util_ratio[cur_class] = round(float(((pseudo_label == cur_class) * mask).sum() / (mask.shape[0] + 1e-8)), 4)
                self.real_class_util_ratio[cur_class] = round(float(((y_ulb == cur_class) * mask).sum() / (mask.shape[0] + 1e-8)), 4)
                self.class_ulb_pseudo_label_acc[cur_class] = round(float(((y_ulb == cur_class) * (pseudo_label == cur_class) * mask).sum() / (mask.sum() + 1e-8)), 4)
                self.class_ulb_pseudo_label_precision[cur_class] = round(float(((pseudo_label == cur_class) * (y_ulb == cur_class) * mask).sum() / (((y_ulb == cur_class) * mask).sum() + 1e-8)), 4)

        # f = open('utility_record.txt', 'a')
        # f.write('{}\n'.format(str(mask.float().mean().item())))
        # f.close()
        
        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(), 
                                         unsup_loss=unsup_loss.item(), 
                                         total_loss=total_loss.item(), 
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['p_model'] = self.hooks_dict['MaskingHook'].p_model.cpu()
        save_dict['time_p'] = self.hooks_dict['MaskingHook'].time_p.cpu()
        save_dict['label_hist'] = self.hooks_dict['MaskingHook'].label_hist.cpu()
        # save_dict['mod_relation'] = self.hooks_dict['MaskingHook'].mod_relation.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['MaskingHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].time_p = checkpoint['time_p'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].label_hist = checkpoint['label_hist'].cuda(self.args.gpu)
        # self.hooks_dict['MaskingHook'].mod_relation = checkpoint['mod_relation'].cuda(self.args.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--ema_p', float, 0.999),
            SSL_Argument('--ent_loss_ratio', float, 0.01),
            SSL_Argument('--use_quantile', str2bool, False),
            SSL_Argument('--clip_thresh', str2bool, False),
            SSL_Argument('--proj_size', int, 64),
            SSL_Argument('--k', int, 10),
            # SSL_Argument('--beta_schedule', str, 'linear'),
            # SSL_Argument('--beta_start', float, 0.0001),
            # SSL_Argument('--beta_end', float, 0.02),
            # SSL_Argument('--timesteps', int, 1000),
            # SSL_Argument('--diffusion_ce_loss', str2bool, False),
            # SSL_Argument('--diffusion_lambda_ce', float, 0.01),
            # SSL_Argument('--diffusion_feature2label_loss_coff', float, 0.1),
            # SSL_Argument('--diffusion_feature_dim', int, 64),
            # SSL_Argument('--use_diffusion', str2bool, False),
            # #ddim
            # SSL_Argument('--ddim', str2bool, True),
            # SSL_Argument('--skip', int, 100),
            # SSL_Argument('--skip-type', str, 'uniform'),
            # SSL_Argument('--eta', float, 0),
        ]