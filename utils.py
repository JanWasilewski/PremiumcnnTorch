import torch
import torch.nn as nn
import torch.nn.functional as F

def nll_gaussian(y_gt, y_pred_mean, y_pred_sd, loss_fn=nn.CrossEntropyLoss()):
    mu = (-y_pred_mean.log() * torch.nn.functional.one_hot(y_gt, num_classes=10)).sum(dim=1).mean()
    mu_2 = mu ** 2
    s = 1. / (y_pred_sd + 1e-4)
    loss1 = torch.mean(torch.sum(mu_2 * s, dim=-1))
    loss2 = torch.mean(torch.sum(torch.log(y_pred_sd), dim=-1))
    loss = loss1 + loss2
    return loss

class MinMaxConstraint:
    def __init__(self, min_value=-11.5, max_value=0.542):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return torch.clamp(w, self.min_value, self.max_value)

def x_Sigma_w_x_T(x, W_Sigma):
    batch_sz = x.shape[0]
    dim = x.shape[-1]
    x = x / dim
    xx_t = torch.sum(x * x, dim=-1, keepdim=True)  # [50, 17, 64] -> [50, 17, 1] or [50, 64] -> [50, 1]
    return xx_t * W_Sigma  # [50,17,64] or [50, 64] or [50, 10]

def w_t_Sigma_i_w(w_mu, in_Sigma):
    dim = torch.sqrt(torch.tensor(w_mu.shape[0], dtype=torch.float32))
    w_mu = w_mu / dim
    Sigma_1 = torch.matmul(in_Sigma, w_mu * w_mu)  # [50, 17, 64] or [50, 10]
    return Sigma_1

def tr_Sigma_w_Sigma_in(in_Sigma, W_Sigma):
    dim = W_Sigma.shape[-1]
    Sigma = torch.sum(in_Sigma, dim=-1, keepdim=True)  # [50,17, 1]
    return Sigma * W_Sigma / dim  # [50,17, 64]

def activation_Sigma(gradi, Sigma_in):
    grad1 = gradi * gradi  
    dim = grad1.shape[1] #Channel num, works properly with other values too
    return Sigma_in * grad1 / dim 

def kl_regularizer(mu, logvar):
    n = mu.shape[0]
    prior_var = 1.0
    kl = -torch.mean(
        (1 + logvar - torch.log(1 + torch.exp(logvar)) / prior_var)
        - (torch.sum(mu ** 2, dim=0) / (n * prior_var))
    )
    kl = torch.where(torch.isnan(kl), torch.tensor(1.0e-5), kl)
    kl = torch.where(torch.isinf(kl), torch.tensor(1.0e-5), kl)
    return kl

def kl_regularizer_conv(mu, logvar):
    k = mu.shape[0]
    mu = mu.view(-1, k)  # Reshape the tensor
    n = mu.shape[0]
    prior_var = 1.0
    kl = -torch.mean((1 + logvar - (torch.log(1 + torch.exp(logvar)) / prior_var)) - 
                     (torch.sum(mu**2, dim=0) / (n * prior_var)))
    
    # Replace NaN and Inf values with a small constant
    kl = torch.where(torch.isnan(kl), torch.tensor(1.0e-5, device=kl.device), kl)
    kl = torch.where(torch.isinf(kl), torch.tensor(1.0e-5, device=kl.device), kl)
    
    return kl