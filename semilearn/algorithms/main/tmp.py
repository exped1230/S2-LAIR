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


class Diffusion_Linear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super(Diffusion_Linear, self).__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        out = self.lin(x)
        gamma = self.embed(t)
        out = gamma.view(-1, self.num_out) * out
        return out

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

class Diffusion_model(nn.Module):
    def __init__(self, proj_size=64, num_class=8, diffusion_feature_dim=512, num_timesteps=1000):
        super(Diffusion_model, self).__init__()
        self.diffusion_linear1 = Diffusion_Linear(proj_size, diffusion_feature_dim, num_timesteps)
        self.unetnorm1 = nn.BatchNorm1d(diffusion_feature_dim)
        self.diffusion_linear2 = Diffusion_Linear(diffusion_feature_dim, diffusion_feature_dim, num_timesteps)
        self.unetnorm2 = nn.BatchNorm1d(diffusion_feature_dim)
        self.diffusion_linear3 = Diffusion_Linear(diffusion_feature_dim, diffusion_feature_dim, num_timesteps)
        self.unetnorm3 = nn.BatchNorm1d(diffusion_feature_dim)
        self.linear4 = nn.Linear(diffusion_feature_dim, num_class*2)

    def forward(self, x, t=None):
            y = self.diffusion_linear1(x, t)
            y = self.unetnorm1(y)
            y = F.softplus(y)
            y = self.diffusion_linear2(y, t)
            y = self.unetnorm2(y)
            y = F.softplus(y)
            y = self.diffusion_linear3(y, t)
            y = self.unetnorm3(y)
            y = F.softplus(y)
            return {'noise':self.linear4(y)}




@ALGORITHMS.register('freematch')
class FreeMatch(AlgorithmBase):
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
        self.num_timesteps = args.timesteps
        self.use_diffusion = args.use_diffusion
        if self.use_diffusion:
        #ddpm
            betas = make_beta_schedule(schedule=args.beta_schedule, num_timesteps=self.num_timesteps,
                                    start=args.beta_start, end=args.beta_end)
            betas = self.betas = betas.float().to(args.gpu)
            self.betas_sqrt = torch.sqrt(betas)
            alphas = 1.0 - betas
            self.alphas = alphas
            self.one_minus_betas_sqrt = torch.sqrt(alphas)
            alphas_cumprod = alphas.cumprod(dim=0)
            self.alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
            self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
            if args.beta_schedule == "cosine":
                self.one_minus_alphas_bar_sqrt *= 0.9999  # avoid division by 0 for 1/sqrt(alpha_bar_t) during inference
            alphas_cumprod_prev = torch.cat(
                [torch.ones(1).to(args.gpu), alphas_cumprod[:-1]], dim=0
            )
            self.alphas_cumprod_prev = alphas_cumprod_prev
            # self.diffusion_model = None
            self.diffusion_model = self.diffusion_model.to(args.gpu)
            self.use_diffusion_ce_loss = args.diffusion_ce_loss
            self.diffusion_lambda_ce = args.diffusion_lambda_ce
            self.diffusion_feature2label_loss_coff = args.diffusion_feature2label_loss_coff
            self.noise_guidence = False

            #ddim
            self.posterior_variance = (
                betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
            )   

            if args.skip_type == "uniform":
                skip = self.num_timesteps // self.args.timesteps
                self.seq = range(0, self.num_timesteps, skip)
            elif args.skip_type == "quad":
                self.seq = (
                    np.linspace(
                        0, np.sqrt(self.num_timesteps * 0.8), self.args.timesteps
                    )
                    ** 2
                )
                self.seq = [int(s) for s in list(self.seq)]
            else:
                raise NotImplementedError        

        self.device = args.gpu

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
        model = CoMatch_Net(model, proj_size=self.args.proj_size)
        diffusion_model = Diffusion_model(proj_size=64, num_class=8, diffusion_feature_dim=1024, num_timesteps=1000)
        return model, diffusion_model

    def set_ema_model(self):
        ema_model = self.net_builder(num_classes=self.num_classes)
        ema_model = CoMatch_Net(ema_model, proj_size=self.args.proj_size)
        ema_model.load_state_dict(self.check_prefix_state_dict(self.model.state_dict()))
        diffusion_ema_model = Diffusion_model(proj_size=64, num_class=8, diffusion_feature_dim=1024, num_timesteps=1000)
        diffusion_ema_model.load_state_dict(self.check_prefix_state_dict(self.diffusion_model.state_dict()))
        return ema_model, diffusion_ema_model
    
    def train_step(self, x_lb, y_lb, x_ulb_w, x_ulb_s):
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

            # calculate mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=logits_x_ulb_w, relation_matrix=self.relation_matrix)


            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook", 
                                          logits=logits_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T)
            
            with torch.no_grad():
                features_lb = feats_x_lb.detach()
                y_feature_batch = feats_x_ulb_s.detach()

                lb_probs = torch.softmax(logits_x_lb, dim=1)
                _, lb_guess = torch.max(lb_probs, dim=1)

                for cur_class in range(self.num_classes):
                    class_mask = (y_lb == cur_class) & (lb_guess == y_lb)
                    if class_mask.sum() == 0:
                        continue
                    cur_class_feature = features_lb[class_mask].sum(0) / class_mask.sum()
                    self.prototype[cur_class] = 0.9 * self.prototype[cur_class] + 0.1 * cur_class_feature
                if self.update_m == 'L2':
                    L2_dis = torch.norm(self.prototype[:, None] - self.prototype, dim=2, p=2) / 0.5
                    new_relation_matrix = torch.exp(-L2_dis)
                elif self.update_m == 'L1':
                    L1_dis = torch.sum(self.prototype[:, None] - self.prototype, dim=2) / 0.5
                    new_relation_matrix = torch.exp(-L1_dis)
                elif self.update_m == 'cos':
                    prototype_tmp = torch.norm(self.prototype, p=2, dim=1, keepdim=True).expand_as(self.prototype) + 1e-12
                    prototype_tmp = self.prototype / prototype_tmp
                    cos_sim = torch.mm(prototype_tmp, prototype_tmp.T) / 0.5
                    new_relation_matrix = torch.exp(cos_sim)

                new_relation_matrix = new_relation_matrix / torch.sum(new_relation_matrix, dim=1, keepdim=True).expand_as(new_relation_matrix)
                self.relation_matrix = 0.9 * self.relation_matrix + 0.1 * new_relation_matrix

                # calculate information entropy
                #use label/unlabel data4
                    #variance_by_data = nn.functional.one_hot(y_lb, num_classes=self.num_classes).float()
                if self.use_diffusion:
                    variance_by_data = pseudo_label
                    _, max_idx = torch.max(variance_by_data, dim=-1,keepdim=True)
                    if self.noise_guidence:
                        pseudo_label_entropy = -torch.sum(variance_by_data * torch.log2(variance_by_data+1e-8), dim=1)
                        pseudo_label_entropy = pseudo_label_entropy.clamp(0.1, 20.0)
                        distributions = [Normal(0, pseudo_label_entropy[i]) for i in range(len(pseudo_label_entropy))]

                        sorted_indices = torch.argsort(self.relation_matrix, dim=1, descending=True)
                        sorted_ranks = torch.argsort(sorted_indices, dim=1)

                        dis = sorted_ranks[max_idx]
                        variance = [torch.exp(dist.log_prob(dis[i]))*self.k for i, dist in enumerate(distributions)]
                        variance= torch.stack(variance).squeeze()
                        variance = variance.clamp(0.1, 20.0)
                    else:
                        variance = torch.ones_like(pseudo_label)
                    variance = torch.cat([variance, variance], dim=1)
                    #diffusion re
                    start = time.time()
                    if self.args.ddim:
                        label_t_0, feature_t_0 = generalized_steps(self.seq, self.diffusion_model, self.betas, self.prototype, self.args.eta, variance[:,:variance.shape[1]//2])
                    else:
                        label_t_0, feature_t_0 = p_sample_loop(self.diffusion_model, self.num_timesteps, self.alphas, self.one_minus_alphas_bar_sqrt, self.prototype, self.relation_matrix, self.k, variance[:,:variance.shape[1]//2], only_last_sample=True)
                    diffusion_label_t_0 = label_t_0.detach().to(self.device)
                    end = time.time()
            if self.use_diffusion:
                diffusion_time = end - start
                _, diffusion_max_id = torch.max(diffusion_label_t_0, dim=-1,keepdim=True)
                #mask = torch.ones_like(variance_by_data)
                diffusion_true_ratios = (((diffusion_max_id.view(-1) == max_idx.view(-1)) * mask).sum() / (mask.sum()+1e-7))
                print(diffusion_true_ratios, 'time:', diffusion_time)
                #label_t_0_predict_logits = self.model(feature_t_0, diffusion_feat=True, t=None, predict_nosie=False)['diffusion_logits']
                #diffusion_feature2label_loss = self.ce_loss(label_t_0_predict_logits, label_t_0, reduction='mean')

                diffusion_label_loss = self.consistency_loss(logits_x_ulb_s,
                                            diffusion_label_t_0,
                                            'ce',
                                            mask=mask,
                                            similarity_matrix=None)

                # diffusion forward loss
                self.diffusion_model.train()
                
                variance_by_data = torch.softmax(logits_x_lb, dim=-1)
                    #print(variance_by_data)
                
                n = variance_by_data.size(0)
                y_batch = variance_by_data
                t = torch.randint(
                    low=0, high=self.num_timesteps, size=(n // 2 + 1,)
                ).to(self.device)
                t = torch.cat([t, self.num_timesteps - 1 - t], dim=0)[:n]
                e = torch.normal(0., std = variance).to(y_batch.device)
                
                y_t_batch, y_feature_t_batch = q_sample(y_batch, y_feature_batch, variance, self.alphas_bar_sqrt, t, e, self.prototype)
                diffusion_output = self.diffusion_model(y_feature_t_batch, t)['noise']
                diffusion_noise_loss = (e - diffusion_output).square().mean()
                # cross-entropy for y_0 reparameterization
                diffusion_ce_loss = torch.tensor([0])
                criterion = nn.CrossEntropyLoss()
                if self.use_diffusion_ce_loss:
                    y_0_reparam_batch, _ = y_0_reparam(self.model, y_feature_t_batch, y_t_batch, self.prototype, t,
                                                    self.one_minus_alphas_bar_sqrt)
                    raw_prob_batch = -(y_0_reparam_batch - 1) ** 2
                    diffusion_ce_loss = criterion(raw_prob_batch, y_batch.to(self.device))
                    diffusion_noise_loss += self.diffusion_lambda_ce * diffusion_ce_loss
                diffusion_noise_loss = diffusion_noise_loss*mask.mean()
            
            #calculate unlabeled loss
            # unsup_loss = self.consistency_loss(logits_x_ulb_s,
            #                               pseudo_label,
            #                               'soft_cos',
            #                               mask=mask,
            #                               similarity_matrix=self.relation_matrix)
            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                pseudo_label,
                                'ce',
                                mask=mask)
            relation_file = open('relation2.txt', mode='a')
            relation_file.write(str(self.relation_matrix)+'\n')
            
            # print(self.relation_matrix)
            # calculate entropy loss
            if mask.sum() > 0:
               ent_loss, _ = entropy_loss(mask, logits_x_ulb_s, self.p_model, self.label_hist)
            else:
               ent_loss = 0.0
            ent_loss = 0.0 
            total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_e * ent_loss 
            print('total_loss:', total_loss)
            #total_loss = sup_loss + self.lambda_u * unsup_loss
            #total_loss = sup_loss + self.lambda_u * unsup_loss + self.lambda_e * ent_loss 


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
        save_dict['mod_relation'] = self.hooks_dict['MaskingHook'].mod_relation.cpu()
        return save_dict


    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['MaskingHook'].p_model = checkpoint['p_model'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].time_p = checkpoint['time_p'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].label_hist = checkpoint['label_hist'].cuda(self.args.gpu)
        self.hooks_dict['MaskingHook'].mod_relation = checkpoint['mod_relation'].cuda(self.args.gpu)
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
            SSL_Argument('--beta_schedule', str, 'linear'),
            SSL_Argument('--beta_start', float, 0.0001),
            SSL_Argument('--beta_end', float, 0.02),
            SSL_Argument('--timesteps', int, 1000),
            SSL_Argument('--diffusion_ce_loss', str2bool, False),
            SSL_Argument('--diffusion_lambda_ce', float, 0.01),
            SSL_Argument('--diffusion_feature2label_loss_coff', float, 0.1),
            SSL_Argument('--diffusion_feature_dim', int, 64),
            SSL_Argument('--use_diffusion', str2bool, False),
            #ddim
            SSL_Argument('--ddim', str2bool, True),
            SSL_Argument('--skip', int, 100),
            SSL_Argument('--skip-type', str, 'uniform'),
            SSL_Argument('--eta', float, 0),
        ]