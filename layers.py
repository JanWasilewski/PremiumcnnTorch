import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

from utils import *

class VDPFirstConv(nn.Module):
    def __init__(self, kernel_size=5, kernel_num=16, kernel_stride=1, padding='valid'):
        super(VDPFirstConv, self).__init__()
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.kernel_stride = kernel_stride
        self.padding = padding
        #self.w_mu = nn.Parameter(torch.ones(kernel_num, 1, kernel_size, kernel_size) *0.02)
        init_from_numpy = torch.tensor(np.random.normal(scale=0.05, size=(kernel_num, 1, kernel_size, kernel_size)), dtype=torch.float32)
        self.w_mu = nn.Parameter(init_from_numpy)
        self.w_sigma = nn.Parameter(torch.full((kernel_num,), -4.6))

    def forward(self, mu_in):
        batch_size = mu_in.size(0)
        num_channel = mu_in.size(1)

        kl_conv = kl_regularizer_conv(self.w_mu, self.w_sigma) 
        w_sigma_2 = torch.log1p(torch.exp(self.w_sigma))
        mu_out = F.conv2d(mu_in, self.w_mu, stride=self.kernel_stride)

        # Extract patches
        x_train_patches = F.unfold(mu_in, kernel_size=self.kernel_size, stride=self.kernel_stride)
        x_train_matrix = x_train_patches.view(batch_size, num_channel * self.kernel_size * self.kernel_size, -1)
        x_dim = x_train_matrix.size(1)
        x_train_matrix = torch.sum(x_train_matrix ** 2, dim=1) / x_dim
        X_XTranspose = x_train_matrix.unsqueeze(-1).repeat(1, 1, self.kernel_num)
        Sigma_out = (w_sigma_2 * X_XTranspose).view(batch_size, mu_out.shape[-1], mu_out.shape[-1], self.kernel_num).permute(0,3,1,2)
        return mu_out, Sigma_out, kl_conv  # mu_out: [batch_size, out_channels, H_out, W_out], Sigma_out: [batch_size, L, kernel_num]

class LinearNotFirst(nn.Module):
    """y = w.x + b"""
    def __init__(self, units, feature_dim=4608):
        super(LinearNotFirst, self).__init__()
        #self.w_mu = nn.Parameter(torch.ones(feature_dim, units)*0.05)
        self.w_mu = nn.Parameter(torch.tensor(np.random.normal(scale=0.05, size=(feature_dim, units)), dtype=torch.float32))
        self.w_sigma = nn.Parameter(torch.full((units,), -4.6))

    def forward(self, mu_in, Sigma_in):
        mu_out = torch.matmul(mu_in, self.w_mu)
        kl_fc = kl_regularizer(self.w_mu, self.w_sigma)
        W_Sigma = torch.log(1 + torch.exp(self.w_sigma))

        # Compute covariances
        Sigma_1 = w_t_Sigma_i_w(self.w_mu, Sigma_in)
        Sigma_2 = x_Sigma_w_x_T(mu_in, W_Sigma)
        Sigma_3 = tr_Sigma_w_Sigma_in(Sigma_in, W_Sigma)
        Sigma_out = Sigma_1 + Sigma_2 + Sigma_3
        Sigma_out = torch.where(torch.isnan(Sigma_out), torch.tensor(1.0e-5, device=Sigma_out.device), Sigma_out)
        Sigma_out = torch.where(torch.isinf(Sigma_out), torch.tensor(1.0, device=Sigma_out.device), Sigma_out)

        return mu_out, Sigma_out, kl_fc

class VDPMaxPooling(nn.Module):
    """VDP_MaxPooling"""
    def __init__(self, pooling_size=2, pooling_stride=2, pooling_pad='same'):
        super(VDPMaxPooling, self).__init__()
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.pooling_pad = pooling_pad # TODO - NOT USED

    def forward(self, mu_in, Sigma_in):
        batch_size, num_channel, hw_in, _ = mu_in.shape
        mu_out, indices = F.max_pool2d(mu_in, kernel_size=self.pooling_size, stride=self.pooling_stride, padding=(0,0), return_indices=True)
        
        # Get output dimensions
        hw_out = mu_out.shape[2]
        indices_flat = indices.view(batch_size, num_channel, -1)  # [batch_size, num_channel, new_size * new_size]
        
        # Compute x_index and y_index
        x_index = indices_flat % hw_in
        y_index = indices_flat // hw_in

        # Combine x_index and y_index to create a final index
        index = y_index * hw_in + x_index  # [batch_size, num_channel, new_size * new_size]
        Sigma_in_reshaped = Sigma_in.view(batch_size, num_channel, -1)  # [batch_size, num_channel, im_size * im_size]
        Sigma_out = Sigma_in_reshaped.gather(2, index)  # [batch_size, num_channel, new_size * new_size]
        Sigma_out = Sigma_out.view(batch_size, num_channel, hw_out, hw_out)  # [batch_size, num_channel, new_size, new_size]

        return mu_out, Sigma_out

class MySoftmax(nn.Module):
    def __init__(self):
        super(MySoftmax, self).__init__()

    def forward(self, mu_in, Sigma_in):
        mu_dim = mu_in.size(-1)
        mu_out = F.softmax(mu_in, dim=-1)
        
        grad = (mu_out - mu_out**2)**2
        Sigma_out = grad * Sigma_in / mu_dim

        # Handle NaN and Inf in Sigma_out
        Sigma_out = torch.where(torch.isnan(Sigma_out), torch.tensor(1.0e-5, device=Sigma_out.device), Sigma_out)
        Sigma_out = torch.where(torch.isinf(Sigma_out), torch.tensor(1.0, device=Sigma_out.device), Sigma_out)

        return mu_out, Sigma_out

class VDP_ReLU(nn.Module):
    """ReLU"""
    def __init__(self):
        super(VDP_ReLU, self).__init__()

    def forward(self, mu_in, Sigma_in):
        mu_out = F.relu(mu_in)

        # Compute gradients
        mu_in.requires_grad_()
        out = F.relu(mu_in)
        gradi = torch.autograd.grad(out, mu_in, torch.ones_like(out), retain_graph=True)[0]

        # Compute new Sigma
        Sigma_out = activation_Sigma(gradi, Sigma_in) 
        Sigma_out = torch.where(torch.isnan(Sigma_out), torch.tensor(1.0e-5, device=Sigma_out.device), Sigma_out)
        Sigma_out = torch.where(torch.isinf(Sigma_out), torch.tensor(1.0, device=Sigma_out.device), Sigma_out)

        return mu_out, Sigma_out
    
class DensityPropCNN(nn.Module):
    def __init__(self, kernel_size, num_kernel, pooling_size, pooling_stride, pooling_pad, units):
        super(DensityPropCNN, self).__init__()
        self.kernel_size = kernel_size
        self.num_kernel = num_kernel
        self.pooling_size = pooling_size
        self.pooling_stride = pooling_stride
        self.pooling_pad = pooling_pad
        self.units = units
        self.conv_1 = VDPFirstConv(kernel_size=self.kernel_size, kernel_num=self.num_kernel, kernel_stride=1, padding='valid')  
        self.relu_1 = VDP_ReLU()
        self.maxpooling_1 = VDPMaxPooling(pooling_size=self.pooling_size[0], pooling_stride=self.pooling_stride[0], pooling_pad=self.pooling_pad)  
        self.fc_1 = LinearNotFirst(self.units)
        self.mysoftmax = MySoftmax()

    def forward(self, inputs, training=True):
        batch_size = inputs.size(0)
        
        mu1, sigma1, kl1 = self.conv_1(inputs) 
        mu2, sigma2 = self.relu_1(mu1, sigma1) 
        mu3, sigma3 = self.maxpooling_1(mu2, sigma2)
        mu4 = mu3.view(batch_size, -1) # Reshape for fully connected layer
        sigma4 = sigma3.view(batch_size, -1)

        mu5, sigma5, kl2 = self.fc_1(mu4, sigma4)
        kl = kl1 + kl2
        
        mu_out, Sigma_out = self.mysoftmax(mu5, sigma5)

        # Handle NaN and Inf in Sigma_out
        Sigma_out = torch.where(torch.isnan(Sigma_out), torch.zeros_like(Sigma_out), Sigma_out)
        Sigma_out = torch.where(torch.isinf(Sigma_out), torch.zeros_like(Sigma_out), Sigma_out)
        
        return mu_out, Sigma_out, kl

# ------------------ NOT USED -------------------
class LinearFirst(nn.Module):
    """y = w.x + b"""
    def __init__(self, units):
        super(LinearFirst, self).__init__()
        self.units = units
        self.ini_sigma = -4.6

    def reset_parameters(self):
        """Initialize the weights and biases"""
        nn.init.normal_(self.w_mu, mean=0.0, std=0.05)
        nn.init.constant_(self.w_sigma, self.ini_sigma)

    def forward(self, inputs):  # inputs shape: [batch_size, seq_len, input_dim]
        # inputs: [50, 17, 64]
        batch_size, seq_len, input_dim = inputs.size()

        # Mean weights
        if not hasattr(self, 'w_mu'):
            self.w_mu = nn.Parameter(torch.empty(input_dim, self.units))
            self.w_sigma = nn.Parameter(torch.empty(self.units))
            self.reset_parameters()

        # KL Divergence regularization (you will need to define this function)
        kl_fc = kl_regularizer(self.w_mu, self.w_sigma)

        # Mean output
        mu_out = torch.matmul(inputs, self.w_mu)  # Shape: [50, 17, 64]

        # Variance
        W_Sigma = torch.log(1 + torch.exp(self.w_sigma))  # Shape: [64]
        Sigma_out = x_Sigma_w_x_T(inputs, W_Sigma)  # Define this function similarly to how it was done in Keras

        # Handle NaN and Inf in Sigma_out
        Sigma_out = torch.where(torch.isnan(Sigma_out), torch.tensor(1.0e-5, device=Sigma_out.device), Sigma_out)
        Sigma_out = torch.where(torch.isinf(Sigma_out), torch.tensor(1.0, device=Sigma_out.device), Sigma_out)

        return mu_out, Sigma_out, kl_fc
    
class LayerNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)  # Compute mean
        std = x.std(dim=-1, keepdim=True)  # Compute standard deviation
        return (x - mean) / (std + self.eps)  # Normalize

class Bayesian_LayerNorm(nn.Module):
    def __init__(self, eps=1e-6):
        super(Bayesian_LayerNorm, self).__init__()
        self.eps = eps

    def forward(self, mu_x, sigma_x):
        mean = mu_x.mean(dim=-1, keepdim=True)
        std = mu_x.std(dim=-1, keepdim=True)
        out_mu = (mu_x - mean) / (std + self.eps)

        mean_sigma = sigma_x.mean(dim=-1, keepdim=True)
        std_sigma = sigma_x.std(dim=-1, keepdim=True)
        Sigma_out = (sigma_x - mean_sigma) / (std_sigma + self.eps)

        # Handle NaN and Inf in Sigma_out
        Sigma_out = torch.where(torch.isnan(Sigma_out), torch.tensor(1.0e-5, device=Sigma_out.device), Sigma_out)
        Sigma_out = torch.where(torch.isinf(Sigma_out), torch.tensor(1.0, device=Sigma_out.device), Sigma_out)

        return out_mu, Sigma_out

class VDP_GeLU(nn.Module):
    def __init__(self):
        super(VDP_GeLU, self).__init__()

    def forward(self, mu_in, Sigma_in):  # mu_in = [50, 17, 64], Sigma_in = [50, 17, 64]
        mu_out = F.gelu(mu_in)  # Apply GeLU

        # Compute gradients
        mu_in.requires_grad_()  # Enable gradient computation
        out = F.gelu(mu_in)
        gradi = torch.autograd.grad(out, mu_in, torch.ones_like(out), retain_graph=True)[0]  # [50, 17, 64]

        # Compute new Sigma
        Sigma_out = activation_Sigma(gradi, Sigma_in)  # Define this function according to your requirements
        return mu_out, Sigma_out  # Shapes: [50, 17, 64], [50, 17, 64, 64]

class VDP_MLP(nn.Module):
    def __init__(self, hidden_features, out_features, dropout_rate=0.1):
        super(VDP_MLP, self).__init__()
        self.dense1 = LinearNotFirst(hidden_features)  # Assuming this class is defined
        self.dense2 = LinearNotFirst(out_features)  # Assuming this class is defined
        self.dropout1 = VDP_Dropout(dropout_rate)
        self.gelu_1 = VDP_GeLU()  # Assuming this class is defined

    def forward(self, mu_in, sigma_in):
        mu_out, sigma_out, kl1 = self.dense1(mu_in, sigma_in)
        mu_out, sigma_out = self.gelu_1(mu_out, sigma_out)
        mu_out, sigma_out = self.dropout1(mu_out, sigma_out)
        mu_out, sigma_out, kl2 = self.dense2(mu_out, sigma_out)
        mu_out, sigma_out = self.dropout1(mu_out, sigma_out)

        kl = kl1 + kl2
        
        # Handle NaN and Inf in Sigma_out
        Sigma_out = torch.where(torch.isnan(sigma_out), torch.tensor(1.0e-5, device=sigma_out.device), sigma_out)
        Sigma_out = torch.where(torch.isinf(sigma_out), torch.tensor(1.0, device=sigma_out.device), sigma_out)

        return mu_out, Sigma_out, kl
    
class VDP_Dropout(nn.Module):
    def __init__(self, drop_prop):
        super(VDP_Dropout, self).__init__()
        self.drop_prop = drop_prop

    def forward(self, mu_in, Sigma_in, training=True):
        scale_sigma = 1.0 / (1 - self.drop_prop)
        if training:
            mu_out = F.dropout(mu_in, p=self.drop_prop, training=True)
            non_zero = mu_out != 0  # [50, 17, 64]
            non_zero_sigma_mask = Sigma_in[non_zero]
            idx_sigma = torch.nonzero(non_zero, as_tuple=False)  # Indices of non-zero entries

            # Initialize Sigma_out with zeros
            Sigma_out = torch.zeros_like(mu_out)
            Sigma_out[idx_sigma[:, 0], idx_sigma[:, 1], idx_sigma[:, 2]] = non_zero_sigma_mask
            Sigma_out /= mu_out.size(-1)
        else:
            mu_out = mu_in
            Sigma_out = Sigma_in

        # Handle NaN and Inf in Sigma_out
        Sigma_out = torch.where(torch.isnan(Sigma_out), torch.tensor(1.0e-5, device=Sigma_out.device), Sigma_out)
        Sigma_out = torch.where(torch.isinf(Sigma_out), torch.tensor(1.0, device=Sigma_out.device), Sigma_out)
        
        return mu_out, Sigma_out
    
class D_Dropout(nn.Module):
    def __init__(self, drop_prop):
        super(D_Dropout, self).__init__()
        self.drop_prop = drop_prop

    def forward(self, mu_in, training=True):
        # mu_in shape: [batch_size, seq_length, embedding_dim]
        if training:
            mu_out = F.dropout(mu_in, p=self.drop_prop, training=True)  # Apply dropout
        else:
            mu_out = mu_in  # During evaluation, return input as is
        return mu_out


