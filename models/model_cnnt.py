"""
Convolutional Neural Net Transformer

A novel structure that combines the ideas behind CNNs and Transformers
CNNT is able to utilize the temporal correlation while keeping the computations efficient

At runtime a config dictionary is required to create the model
"""

import os
import sys
import math
import numpy as np
from math import exp
from pathlib import Path

Project_DIR = Path(__file__).parents[0].resolve()
sys.path.insert(1, str(Project_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F

# -------------------------------------------------------------------------------------------------
# Helpers

def compute_conv_output_shape(h_w, kernel_size, stride, pad, dilation):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    
    h = torch.div( (h_w[0] + (2*pad[0]) - (dilation*(kernel_size[0] - 1)) - 1), stride[0], rounding_mode='floor') + 1
    w = torch.div( (h_w[1] + (2*pad[1]) - (dilation*(kernel_size[1] - 1)) - 1), stride[1], rounding_mode='floor') + 1
    
    return h, w

class Conv2DExt(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.conv2d = nn.Conv2d(*args,**kwargs)
    def forward(self, input):
        # requried input to have 5 dimensions
        B, T, C, H, W = input.shape
        #input = input.view(-1, C, H, W)
        y = self.conv2d(input.reshape((B*T, C, H, W)))
        #y = self.conv2d(input.view(-1, C, H, W))
        return torch.reshape(y, [B, T, y.shape[1], y.shape[2], y.shape[3]])
    
class Conv3DExt(nn.Module):
    def __init__(self,*args,**kwargs):
        super().__init__()
        self.conv3d = nn.Conv3d(*args,**kwargs)
    def forward(self, input):
        # requried input to have 5 dimensions
        y = self.conv3d(torch.permute(input, (0, 2, 1, 3, 4)))
        return torch.permute(y, (0, 2, 1, 3, 4))
    
# -------------------------------------------------------------------------------------------------
# The CNN transformer to process the [B, T, C, H, W], a series of images

class CnnSelfAttention(nn.Module):
    """
    Multi-head cnn attention model    
    """
    
    def __init__(self, H, W, C=1, T=32, output_channels=16, is_causal=False, n_head=8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dropout_p=0.1):
        """Define the layers for a cnn self-attention

            Input to the attention layer has the size [B, T, C, H, W]
            Output has the size [B, T, output_channels, H', W']
            
        Args:
            C (int, optional): input dimension [B, T, C]. Defaults to 4.
            T (int, optional): number of time points for attention layer. Defaults to 32.
            output_channels (int, optional): number of output channels. Defaults to 1.
            is_causal (bool, optional): whether applying the masking to make the layer causal. Defaults to False.
            n_head (int, optional): number of heads. Defaults to 4.
            kernel_size, stride, padding: convolution parameters            
        """
        super().__init__()            
        
        self.C = C
        self.T = T
        self.output_channels = output_channels
        self.is_causal = is_causal
        self.n_head = n_head
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
                        
        H_prime, W_prime = compute_conv_output_shape((H, W), kernel_size=self.kernel_size, stride=self.stride, pad=self.padding, dilation=1)
       
        # key, query, value projections convolution
        # Wk, Wq, Wv
        self.key = Conv2DExt(C, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.query = Conv2DExt(C, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.value = Conv2DExt(C, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
                
        self.output_proj = Conv2DExt(output_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)
        self.attn_drop = nn.Dropout(dropout_p)
        self.resid_drop = nn.Dropout(dropout_p)
    
        self.register_buffer("mask", torch.tril(torch.ones(1000, 1000)).view(1, 1, 1000, 1000))
                
    def forward(self, x):
        """forward pass for the 

        Args:
            x ([B, T, C, H, W]): Input of a batch of time series

        Returns:
            y: logits in the shape of [B, T, output_channels, H', W']
        """

        assert self.output_channels % self.n_head == 0
        
        B, T, C, H, W = x.size()

        H_prime = torch.div( (H + (2*self.padding[0]) - (self.kernel_size[0] - 1) - 1), self.stride[0], rounding_mode='floor') + 1
        W_prime = torch.div( (W + (2*self.padding[1]) - (self.kernel_size[1] - 1) - 1), self.stride[1], rounding_mode='floor') + 1

        # apply the key, query and value matrix
        k = self.key(x).view(B, T, self.n_head, torch.div(self.output_channels, self.n_head, rounding_mode='floor'), H_prime, W_prime).transpose(1, 2)
        q = self.query(x).view(B, T, self.n_head, torch.div(self.output_channels, self.n_head, rounding_mode='floor'), H_prime, W_prime).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, torch.div(self.output_channels, self.n_head, rounding_mode='floor'), H_prime, W_prime).transpose(1, 2)

        # k, q, v are [B, nh, T, hc, H', W']

        B, nh, T, hc, H_prime, W_prime = k.shape

        # Compute attention matrix, use the matrix broadcasing 
        # https://pytorch.org/docs/stable/notes/broadcasting.html
        # (B, nh, T, hc, H', W') x (B, nh, hc, H', W', T) -> (B, nh, T, T)
        att = (q.view(B, nh, T, hc*H_prime*W_prime) @ k.view(B, nh, T, hc*H_prime*W_prime).transpose(-2, -1)) * torch.tensor(1.0 / math.sqrt(hc*H_prime*W_prime))

        #att = (q.view(B, nh, T, hc*H_prime*W_prime) @ k.view(B, nh, T, hc*H_prime*W_prime).transpose(-2, -1))
        
        # if causality is needed, apply the mask
        if(self.is_causal):
            att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        
        # (B, nh, T, T) * (B, nh, T, hc, H', W')
        y = att @ v.view(B, nh, T, hc*H_prime*W_prime)
        y = y.transpose(1, 2).contiguous().view(B, T, self.output_channels, H_prime, W_prime)
        y = self.output_proj(y)
        return y

class CnnTransformer(nn.Module):
    """ 
    CNN Transformer module
    
    The Pre-LayerNorm implementation is used here:
    
    x-> LayerNorm -> attention -> + -> LayerNorm -> CNN mixer -> + -> logits
    |-----------------------------| |-------------------------------|  
    """

    def __init__(self, H, W, C=1, T=32, 
                 output_channels=16,                  
                 is_causal=False, 
                 n_head=8, 
                 norm_mode='layer',
                 kernel_size=(3, 3), 
                 stride=(1, 1), 
                 padding=(1, 1), 
                 dropout_p=0.1,
                 with_mixer=True):
        """set up cnn transformer
        
           norm_mode: layer - norm along C, H, W; batch - norm along B*T; or instance
        """
        
        super().__init__()
        
        self.norm_mode = norm_mode
        assert self.norm_mode=="layer" or self.norm_mode=="batch" or self.norm_mode=="instance"        
        if(self.norm_mode=="layer"):
            self.ln1 = nn.LayerNorm([output_channels, H, W])
            self.ln2 = nn.LayerNorm([output_channels, H, W])
        elif(self.norm_mode=="batch"):
            self.bn1 = nn.BatchNorm2d(output_channels)
            self.bn2 = nn.BatchNorm2d(output_channels)
        elif(self.norm_mode=="instance"):
            self.in1 = nn.InstanceNorm2d(output_channels)
            self.in2 = nn.InstanceNorm2d(output_channels)
            
        self.attn = CnnSelfAttention(H, W, C=output_channels, T=T, output_channels=output_channels, is_causal=is_causal, n_head=n_head, kernel_size=kernel_size, stride=stride, padding=padding, dropout_p=dropout_p)

        self.with_mixer = with_mixer
        if(self.with_mixer):
            self.mlp = nn.Sequential(
                Conv2DExt(output_channels, 4*output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                nn.GELU(),
                Conv2DExt(4*output_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
                nn.Dropout(dropout_p),
            )

    def forward(self, x):
              
        if(self.norm_mode=="layer"):
            x = x + self.attn(self.ln1(x))
            if(self.with_mixer):
                x = x + self.mlp(self.ln2(x))

        elif(self.norm_mode=="batch"):
            B, T, C, H, W = x.shape
            x1 = torch.reshape(self.bn1(torch.reshape(x, (B*T, C, H, W))), x.shape)
            x = x + self.attn(x1)

            if(self.with_mixer):
                x2 = torch.reshape(self.bn2(torch.reshape(x, (B*T, C, H, W))), x.shape)
                x = x + self.mlp(x2)

        elif(self.norm_mode=="instance"):
            B, T, C, H, W = x.shape
            x1 = torch.reshape(self.in1(torch.reshape(x, (B*T, C, H, W))), x.shape)
            x = x + self.attn(x1)

            if(self.with_mixer):
                x2 = torch.reshape(self.in2(torch.reshape(x, (B*T, C, H, W))), x.shape)
                x = x + self.mlp(x2)

        return x

class BlockSet(nn.Module):
    """
    A set of CNNT blocks
    """

    def __init__(self, blocks_per_set, H, W, C=1, T=30, interpolate="none", output_channels=16, is_causal=False, n_head=8, norm_mode='layer', kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dropout_p=0.1, with_mixer=True):
        super().__init__()

        H_prime, W_prime = compute_conv_output_shape((H, W), kernel_size=kernel_size, stride=stride, pad=padding, dilation=1)

        self.input_proj = Conv2DExt(C, output_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)

        self.blocks = nn.Sequential(*[CnnTransformer(H=H_prime, 
                                                     W=W_prime, 
                                                     C=output_channels, 
                                                     T=T, 
                                                     output_channels=output_channels, 
                                                     is_causal=is_causal, 
                                                     n_head=n_head, 
                                                     norm_mode=norm_mode, 
                                                     kernel_size=kernel_size, 
                                                     stride=stride, 
                                                     padding=padding, 
                                                     dropout_p=dropout_p, 
                                                     with_mixer=with_mixer) for _ in range(blocks_per_set)])

        self.interpolate = interpolate

    def forward(self, x):

        x = self.input_proj(x)
        x = self.blocks(x)

        B, T, C, H, W = x.shape

        interp = x

        if self.interpolate=="down":
            interp = F.interpolate(x, scale_factor=(1.0, 0.5, 0.5), mode="trilinear", align_corners=False, recompute_scale_factor=False)
            interp = interp.view(B, T, C, torch.div(H, 2, rounding_mode='floor'), torch.div(W, 2, rounding_mode='floor'))

        if self.interpolate=="up":
            interp = F.interpolate(x, scale_factor=(1.0, 2.0, 2.0), mode="trilinear", align_corners=False, recompute_scale_factor=False)
            interp = interp.view(B, T, C, H*2, W*2)

        # self.interpolate=="none"

        return x, interp

class CNNTUnet(nn.Module):
    """
    The full CNN_Transformer model for Unet architecture
    """

    def __init__(self, blocks, blocks_per_set, H, W, C_in, T, C_out, n_head=8, norm_mode='layer', kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), dropout_p=0.1, with_mixer=True):
        """CNNT Unet configuration

        Args:
            blocks (list): number of output channels for every resolution layers; e.g. [16, 32, 64] means two downsample layers with 16 and 32 channels and the bridge layers have 64 channel outputs
            blocks_per_set (int): number of CNN transformer module each block has
            H, W, C_in, T (int): tensor size of inputs [B, T, C_in, H, W]
            C_out : number of output channels
            n_head (int, optional): number of heads in CNNT transformer. Defaults to 8.
            norm_mode (str, optional) : norm mode, "layer", "batch" or "instance" or "mixed"; if mixed, the top layers use instance norm and deeper layers uses batch norm
            kernel_size (tuple, optional): conv kernel size in cnnt transformer. Defaults to (3, 3).
            stride (tuple, optional): conv stride. Defaults to (1, 1).
            padding (tuple, optional): conv padding. Defaults to (1, 1).
            dropout_p (float, optional): drop out prob in cnnt transformer. Defaults to 0.1.
            with_mixer (bool, optional): with to add mixer in the CNNT module
        """
        super().__init__()

        N = blocks_per_set
        K = kernel_size
        S = stride
        P = padding
        D = dropout_p

        assert len(blocks)>1 and len(blocks)<5

        self.blocks = blocks

        if(norm_mode != "mixed"):
            norm_modes = [norm_mode for i in range(2*len(self.blocks)-1)]
        else:
            norm_modes = ["batch" for i in range(2*len(self.blocks)-1)]
            norm_modes[0] = "instance"
            norm_modes[-1] = "instance"

        if(len(self.blocks)==2): # one downsample layers
           
            self.down1 = BlockSet(N, H, W, C=C_in, T=T, output_channels=blocks[0], interpolate="down", n_head=n_head, norm_mode=norm_modes[0], kernel_size=K, stride=S, padding=P, dropout_p=D, with_mixer=with_mixer)

            self.up1 = BlockSet(N, torch.div(H, 2, rounding_mode='floor'), torch.div(W, 2, rounding_mode='floor'), C=blocks[0], T=T, output_channels=blocks[1], interpolate="up", n_head=n_head, norm_mode=norm_modes[1], kernel_size=K, stride=S, padding=P, dropout_p=D, with_mixer=with_mixer)

            self.final = BlockSet(N, H, W, C=blocks[0]+blocks[1], T=T, output_channels=blocks[1], interpolate="none", n_head=n_head, norm_mode=norm_modes[2], kernel_size=K, stride=S, padding=P, dropout_p=D, with_mixer=with_mixer)

        if(len(self.blocks)==3): # two downsample layers
                
            self.down1 = BlockSet(N, H, W, C=C_in, T=T, output_channels=blocks[0], interpolate="down", n_head=n_head, norm_mode=norm_modes[0], kernel_size=K, stride=S, padding=P, dropout_p=D, with_mixer=with_mixer)
            self.down2 = BlockSet(N, torch.div(H, 2, rounding_mode='floor'), torch.div(W, 2, rounding_mode='floor'), C=blocks[0], T=T, output_channels=blocks[1], interpolate="down", n_head=n_head, norm_mode=norm_modes[1], kernel_size=K, stride=S, padding=P, dropout_p=D, with_mixer=with_mixer)

            self.up1 = BlockSet(N, torch.div(H, 4, rounding_mode='floor'), torch.div(W, 4, rounding_mode='floor'), C=blocks[1], T=T, output_channels=blocks[2], interpolate="up", n_head=n_head, norm_mode=norm_modes[2], kernel_size=K, stride=S, padding=P, dropout_p=D, with_mixer=with_mixer)
            self.up2 = BlockSet(N, torch.div(H, 2, rounding_mode='floor'), torch.div(W, 2, rounding_mode='floor'), C=blocks[1]+blocks[2], T=T, output_channels=blocks[2], interpolate="up", n_head=n_head, norm_mode=norm_modes[3], kernel_size=K, stride=S, padding=P, dropout_p=D, with_mixer=with_mixer)

            self.final = BlockSet(N, H, W, C=blocks[0]+blocks[2], T=T, output_channels=blocks[1], interpolate="none", n_head=n_head, norm_mode=norm_modes[4], kernel_size=K, stride=S, padding=P, dropout_p=D, with_mixer=with_mixer)

        if(len(self.blocks)==4): # three downsample layers
            self.down1 = BlockSet(N, H, W, C=C_in, T=T, output_channels=blocks[0], interpolate="down", n_head=n_head, norm_mode=norm_modes[0], kernel_size=K, stride=S, padding=P, dropout_p=D, with_mixer=with_mixer)
            self.down2 = BlockSet(N, torch.div(H, 2, rounding_mode='floor'), torch.div(W, 2, rounding_mode='floor'), C=blocks[0], T=T, output_channels=blocks[1], interpolate="down", n_head=n_head, norm_mode=norm_modes[1], kernel_size=K, stride=S, padding=P, dropout_p=D, with_mixer=with_mixer)
            self.down3 = BlockSet(N, torch.div(H, 4, rounding_mode='floor'), torch.div(W, 4, rounding_mode='floor'), C=blocks[1], T=T, output_channels=blocks[2], interpolate="down", n_head=n_head, norm_mode=norm_modes[2], kernel_size=K, stride=S, padding=P, dropout_p=D, with_mixer=with_mixer)

            self.up1 = BlockSet(N, torch.div(H, 8, rounding_mode='floor'), torch.div(W, 8, rounding_mode='floor'), C=blocks[2], T=T, output_channels=blocks[3], interpolate="up", n_head=n_head, norm_mode=norm_modes[3], kernel_size=K, stride=S, padding=P, dropout_p=D, with_mixer=with_mixer)
            self.up2 = BlockSet(N, torch.div(H, 4, rounding_mode='floor'), torch.div(W, 4, rounding_mode='floor'), C=blocks[2]+blocks[3], T=T, output_channels=blocks[3], interpolate="up", n_head=n_head, norm_mode=norm_modes[4], kernel_size=K, stride=S, padding=P, dropout_p=D, with_mixer=with_mixer)
            self.up3 = BlockSet(N, torch.div(H, 2, rounding_mode='floor'), torch.div(W, 2, rounding_mode='floor'), C=blocks[1]+blocks[3], T=T, output_channels=blocks[2], interpolate="up", n_head=n_head, norm_mode=norm_modes[5], kernel_size=K, stride=S, padding=P, dropout_p=D, with_mixer=with_mixer)

            self.final = BlockSet(N, H, W, C=blocks[0]+blocks[2], T=T, output_channels=blocks[1], interpolate="none", n_head=n_head, norm_mode=norm_modes[6], kernel_size=K, stride=S, padding=P, dropout_p=D, with_mixer=with_mixer)

        self.output_proj = Conv2DExt(blocks[1], C_out, K, padding=P)

    def forward(self, x):

        if(len(self.blocks)==2): # blocks [16, 32]
                                                     # x :[B, T,  C, 64, 64]
            x1, x1_interp = self.down1(x)            # x1:[B, T, 16, 64, 64], x1_interp:[B, T, 16, 32, 32]

            y1, y1_interp = self.up1(x1_interp)      # y1:[B, T, 32, 16, 16], y1_interp:[B, T, 32, 32, 32]
            c1 = torch.cat((y1_interp, x1), dim=2)   # c1:[B, T, 48, 32, 32]

            z1, z1_interp = self.final(c1)           # z1:[B, T, 32, 64, 64], z1_interp:[B, T, 32, 64, 64]

            output = self.output_proj(z1)

        if(len(self.blocks)==3): # blocks [16, 32, 64]
                                                     # x :[B, T,  C, 64, 64]
            x1, x1_interp = self.down1(x)            # x1:[B, T, 16, 64, 64], x1_interp:[B, T, 16, 32, 32]
            x2, x2_interp = self.down2(x1_interp)    # x2:[B, T, 32, 32, 32], x2_interp:[B, T, 32, 16, 16]

            y1, y1_interp = self.up1(x2_interp)      # y1:[B, T, 64, 16, 16], y1_interp:[B, T, 64, 32, 32]
            c1 = torch.cat((y1_interp, x2), dim=2)   # c1:[B, T, 96, 32, 32]
            y2, y2_interp = self.up2(c1)             # y2:[B, T, 64, 32, 32], y2_interp:[B, T, 64, 64, 64] 
            c2 = torch.cat((y2_interp, x1), dim=2)   # c2:[B, T, 80, 64, 64] 
    
            z1, z1_interp = self.final(c2)           # z1:[B, T, 32, 64, 64], z1_interp:[B, T, 32, 64, 64]

            output = self.output_proj(z1)

        if(len(self.blocks)==4): # blocks [16, 32, 64, 128]
                                                     # x :[B, T,  C, 64, 64]
            x1, x1_interp = self.down1(x)            # x1:[B, T, 16, 64, 64], x1_interp:[B, T, 16, 32, 32]
            x2, x2_interp = self.down2(x1_interp)    # x2:[B, T, 32, 32, 32], x2_interp:[B, T, 32, 16, 16]
            x3, x3_interp = self.down3(x2_interp)    # x3:[B, T, 64, 16, 16], x3_interp:[B, T, 64, 8, 8]

            y1, y1_interp = self.up1(x3_interp)      # y1:[B, T, 128,    8, 8],   y1_interp:[B, T, 128, 16, 16]
            c1 = torch.cat((y1_interp, x3), dim=2)   # c1:[B, T, 128+64, 16, 16]
            y2, y2_interp = self.up2(c1)             # y2:[B, T, 128,    16, 16], y2_interp:[B, T, 128, 32, 32] 
            c2 = torch.cat((y2_interp, x2), dim=2)   # c2:[B, T, 128+32, 32, 32] 
            y3, y3_interp = self.up3(c2)             # y3:[B, T, 64,     32, 32], y3_interp:[B, T, 64, 64, 64] 
            c3 = torch.cat((y3_interp, x1), dim=2)   # c3:[B, T, 64+16,  64, 64]

            z1, z1_interp = self.final(c3)           # z1:[B, T, 32, 64, 64], z1_interp:[B, T, 32, 64, 64]

            output = self.output_proj(z1)

        return output

# -------------------------------------------------------------------------------------------------

class CNNT_base_model_runtime(nn.Module):
    """CNNT base model for image enhancement
    """

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.height = config.height[0]
        self.width = config.width[0]

    def set_up_scheduling(self):
        if self.config.optim is not None:

            if (self.config.optim  == "adamw"):
                if self.config.no_w_decay:
                    self.optim = self.configure_optimizers(self.config)
                else:
                    self.optim = optim.AdamW(self.parameters(), lr=self.config.global_lr, betas=(self.config.beta1, self.config.beta2), eps=1e-08,
                                            weight_decay=self.config.weight_decay, amsgrad=False)

            if (self.config.optim  == "sgd"):
                self.optim = optim.SGD(self.parameters(), lr=self.config.global_lr, momentum=0.9, weight_decay=self.config.weight_decay,
                                    nesterov=False)

            if (self.config.optim  == "nadam"):
                self.optim = optim.NAdam(self.parameters(), lr=self.config.global_lr, betas=(self.config.beta1, self.config.beta2), eps=1e-08,
                                        weight_decay=self.config.weight_decay, momentum_decay=0.004)

            # set up the scheduler
            self.scheduler = None

            if (self.config.scheduler == "ReduceLROnPlateau"):
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optim, 'min',
                                                                    patience=5,
                                                                    min_lr=[2e-6, self.config.global_lr] if self.config.no_w_decay else 2e-6,
                                                                    cooldown=2,
                                                                    factor=0.8,)
                self.scheduler_on_batch = False

            if (self.config.scheduler == "StepLR"):
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optim, 5, gamma=0.8, last_epoch=-1,)
                self.scheduler_on_batch = False

            # if (self.config.scheduler == "OneCycleLR"):
            #     self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optim, max_lr=self.config.global_lr,
            #                                                     total_steps=train_total_len, 
            #                                                     pct_start=0.3,
            #                                                     anneal_strategy='cos', cycle_momentum=True,
            #                                                     base_momentum=0.85, max_momentum=0.95,
            #                                                     div_factor=25,
            #                                                     final_div_factor=10000,
            #                                                     three_phase=False,
            #                                                     last_epoch=-1)

            #     self.scheduler_on_batch = True
        else:
            # Used for testing/plotting
            pass

    def configure_optimizers(self, config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Conv3d, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.config.global_lr, betas=(self.config.beta1, self.config.beta2), eps=1e-08, amsgrad=False)

        return optimizer

    def set_up_loss(self, loss, loss_weights, loss_args=[0,0,0,0,0], device='cpu'):

        loss_f = Image_Enhancement_Combined_Loss()
        for ind, l in enumerate(loss):
            if(l == "mse"):
                loss_f.add_loss(Weighted_MSE_Complex_Loss(), w=loss_weights[ind], condition=loss_args[ind])

            elif(l == "l1"):
                loss_f.add_loss(Weighted_L1_Complex_Loss(), w=loss_weights[ind], condition=loss_args[ind])

            elif(l == "ssim"):
                loss_f.add_loss(Weighted_SSIM_Complex_Loss(reduction='mean', window_size=7, device=device), w=loss_weights[ind], condition=loss_args[ind])

            elif(l == "ssim3D"):
                loss_f.add_loss(Weighted_SSIM3D_Complex_Loss(reduction='mean', window_size=7, device=device), w=loss_weights[ind], condition=loss_args[ind])

            elif(l == "sobel"):
                loss_f.add_loss(Weighted_Sobel_Complex_Loss(device=device), w=loss_weights[ind], condition=loss_args[ind])

            else:
                raise f"loss type not supported:{l}"

        return loss_f

    def save(self, epoch):
        """
        Save the current model weights on the given epoch
        """
        save_file_name = f"{self.config.model_file_name}_epoch-{epoch}.pt"
        save_file_path = os.path.join(self.config.check_path, save_file_name)
        torch.save(self.state_dict(), save_file_path)

    def load(self, load_path, device=None):
        """
        Load a checkpoint is the load path is given in config
        """
        # logging.info(f"Loading model from {load_path}")
        if(device is None):
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        try:
            self.load_state_dict(torch.load(load_path, map_location=device))
        except:
            self.load_state_dict(torch.load(load_path, map_location=device)["model_state"])

# -------------------------------------------------------------------------------------------------

class Batch_perm(torch.nn.Module):
    def forward(self, x):
        return torch.permute(x, (0, 2, 1, 3, 4))

class CNNT_enhanced_denoising_runtime(CNNT_base_model_runtime):
    """
    The full CNN_Transformer model for Microscopy Denoising
    """

    def __init__(self, config):
        super().__init__(config=config)

        K = config.kernel_size
        S = config.stride
        P = config.padding
        D = config.dropout_p
        with_mixer = (config.with_mixer>0)

        self.no_residual = config.no_residual

        C_in = 1
        C_out = 1

        self.pre_cnnt = nn.Sequential(
            Conv2DExt(C_in, 16, kernel_size=K, stride=S, padding=P, bias=True),
            Conv2DExt(  16, 32, kernel_size=K, stride=S, padding=P, bias=True)
        )

        self.cnnt = CNNTUnet(blocks=config.blocks,
                         blocks_per_set=config.blocks_per_set,
                         H=self.height, W=self.width,
                         C_in=32,
                         T=config.time,
                         C_out=32,
                         n_head=config.n_head,
                         norm_mode=config.norm_mode,
                         kernel_size=(K,K), stride=(S,S), padding=(P,P),
                         dropout_p=D, with_mixer=with_mixer)

        self.pos_cnnt = nn.Sequential(
            Conv2DExt(32,    16, kernel_size=K, stride=S, padding=P, bias=True),
            Conv2DExt(16, C_out, kernel_size=K, stride=S, padding=P, bias=True)
        )

        device = config.device

        # setup loss function and optimizer
        self.loss_f = self.set_up_loss(config.loss, config.loss_weights, device=device)

        self.loss_f_test = self.set_up_loss(config.loss, config.loss_weights, device='cpu')

        self.set_up_scheduling()

        # if a load checkpoint is given, load it
        if config.load_path != None:
            self.load(config.load_path)

    def forward(self, x):
        # Pass the input to CNNT and work with the output

        pre = self.pre_cnnt(x)

        noise = self.cnnt(pre)
        output = noise if self.no_residual else pre - noise

        pos = self.pos_cnnt(output)

        return pos

    def compute_loss(self, output, targets, weights, inputs=None, epoch=None):
        # compute loss
        loss = self.loss_f(output, targets, weights, inputs, epoch)
        return loss

    def compute_loss_test(self, output, targets, weights, inputs=None, epoch=None):
        # separate loss func to compute loss on cpu on test set
        loss = self.loss_f_test(output, targets, weights, inputs, epoch)
        return loss

# -------------------------------------------------------------------------------------------------
# All different types of losses

class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        x = torch.mul(x, x)
        x = torch.sum(x, dim=1, keepdim=True)
        x = torch.sqrt(x)
        return x

class Weighted_Sobel_Complex_Loss:
    """
    Weighted loss for complex with Sobel operator
    """
    def __init__(self, device='cpu'):
        self.sobel = Sobel()
        self.sobel.to(device=device)

    def __call__(self, outputs, targets, weights):

        B, T, C, H, W = targets.shape

        if(C==2):
            outputs_im = torch.sqrt(outputs[:,:,0,:,:]*outputs[:,:,0,:,:] + outputs[:,:,1,:,:]*outputs[:,:,1,:,:])
            targets_im = torch.sqrt(targets[:,:,0,:,:]*targets[:,:,0,:,:] + targets[:,:,1,:,:]*targets[:,:,1,:,:])
        else:
            outputs_im = outputs
            targets_im = targets

        outputs_im = torch.reshape(outputs_im, (B*T, 1, H, W))
        targets_im = torch.reshape(targets_im, (B*T, 1, H, W))
        diff_sobel_square = torch.square(self.sobel(outputs_im)-self.sobel(targets_im))

        if(weights is not None):
            if(weights.ndim==2):
                weights = torch.reshape(weights, (outputs.shape[0], T, 1, 1))
                weights_used = weights.reshape(B*T)
            elif(weights.ndim==5):
                weights_used = weights.reshape(B*T, 1, H, W)
            else:
                weights = torch.reshape(weights, (outputs.shape[0], 1, 1, 1))
                weights_used = weights.repeat(1, T, 1, 1).reshape(B*T)
            v_sobel = torch.sum(weights_used[:, None, None, None] * diff_sobel_square) / torch.sum(weights_used)
        else:
            v_sobel = torch.sum(diff_sobel_square)

        if(torch.any(torch.isnan(v_sobel))):
            v_sobel = 0.0

        return torch.sqrt(v_sobel) / (B*T) / (H*W)

class Weighted_SSIM_Complex_Loss:
    """
    Weighted loss for complex with SSIM
    """
    def __init__(self, reduction='mean', window_size=7, device='cpu'):
        self.reduction=reduction
        self.ssim_loss = SSIM(size_average=False, window_size=window_size, device=device)

    def __call__(self, outputs, targets, weights):

        B, T, C, H, W = targets.shape
        if(C==2):
            outputs_im = torch.sqrt(outputs[:,:,0,:,:]*outputs[:,:,0,:,:] + outputs[:,:,1,:,:]*outputs[:,:,1,:,:])
            targets_im = torch.sqrt(targets[:,:,0,:,:]*targets[:,:,0,:,:] + targets[:,:,1,:,:]*targets[:,:,1,:,:])
        else:
            outputs_im = outputs
            targets_im = targets

        outputs_im = torch.reshape(outputs_im, (B*T, 1, H, W))
        targets_im = torch.reshape(targets_im, (B*T, 1, H, W))
        if(weights is not None):
            if(weights.ndim==2):
                weights = torch.reshape(weights, (outputs.shape[0], T, 1, 1))
                weights_used = weights.reshape(B*T)
            elif(weights.ndim==5):
                weights_used = weights.reshape(B*T, 1, H, W)
            else:
                weights = torch.reshape(weights, (outputs.shape[0], 1, 1, 1))
                weights_used = weights.repeat(1, T, 1, 1).reshape(B*T)
            if weights.ndim==5:
                v_ssim = torch.mean(self.ssim_loss(weights_used*outputs_im, weights_used*targets_im)) #/ torch.sum(weights_used)
            else:
                v_ssim = torch.sum(weights_used*self.ssim_loss(outputs_im, targets_im)) / torch.sum(weights_used)
        else:
            v_ssim = torch.mean(self.ssim_loss(outputs_im, targets_im))

        if(torch.any(torch.isnan(v_ssim))):
            v_ssim = 1.0

        return (1.0-v_ssim)

class Weighted_SSIM3D_Complex_Loss:
    """
    Weighted loss for complex with SSIM3D
    """
    def __init__(self, reduction='mean', window_size=7, device='cpu'):
        self.reduction=reduction
        self.ssim_loss = SSIM3D(size_average=False, window_size=window_size, device=device)

    def __call__(self, outputs, targets, weights):

        B, T, C, H, W = targets.shape
        if(C==2):
            outputs_im = torch.sqrt(outputs[:,:,0,:,:]*outputs[:,:,0,:,:] + outputs[:,:,1,:,:]*outputs[:,:,1,:,:])
            targets_im = torch.sqrt(targets[:,:,0,:,:]*targets[:,:,0,:,:] + targets[:,:,1,:,:]*targets[:,:,1,:,:])
        else:
            outputs_im = outputs
            targets_im = targets

        outputs_im = torch.permute(outputs_im, (0, 2, 1, 3, 4))
        targets_im = torch.permute(targets_im, (0, 2, 1, 3, 4))
        if(weights is not None):
            if(weights.ndim==2):
                weights = torch.reshape(weights, (outputs.shape[0], T, 1, 1))
                weights_used = weights.reshape(B*T)
            elif(weights.ndim==5):
                weights_used = weights.reshape(B*T, 1, H, W)
            else:
                weights = torch.reshape(weights, (outputs.shape[0], 1, 1, 1))
                weights_used = weights.repeat(1, T, 1, 1).reshape(B*T)
            if weights.ndim==5:
                v_ssim = torch.mean(self.ssim_loss(weights_used*outputs_im, weights_used*targets_im)) #/ torch.sum(weights_used)
            else:
                v_ssim = torch.sum(weights_used*self.ssim_loss(outputs_im, targets_im)) / torch.sum(weights_used)
        else:
            v_ssim = torch.mean(self.ssim_loss(outputs_im, targets_im))

        if(torch.any(torch.isnan(v_ssim))):
            v_ssim = 1.0

        return (1.0-v_ssim)

class Weighted_L1_Complex_Loss:
    """
    Weighted L1 loss for complex
    """
    def __init__(self, reduction='mean'):
        self.reduction=reduction

    def __call__(self, outputs, targets, weights):

        B, T, C, H, W = targets.shape

        if(C==2):
            diff_L1 = torch.abs(outputs[:,:,0,:,:]-targets[:,:,0,:,:]) + torch.abs(outputs[:,:,1,:,:]-targets[:,:,1,:,:])
        else:
            diff_L1 = torch.abs(outputs-targets)

        if(weights is not None):
            if(weights.ndim==2):
                weights = torch.reshape(weights, (outputs.shape[0], T, 1, 1))
                v_l1 = torch.sum(weights.reshape(B,T,1,1,1) * diff_L1.reshape(B,T,1,H,W)) / torch.sum(weights)
            else:
                weights = torch.reshape(weights, (outputs.shape[0], 1, 1, 1))
                v_l1 = torch.sum(weights.reshape(B,1,1,1,1) * diff_L1.reshape(B,T,1,H,W)) / torch.sum(weights)
        else:
            v_l1 = torch.sum(diff_L1.reshape(B,T,1,H,W))

        return v_l1 / (B*T) / (H*W)

class Weighted_MSE_Complex_Loss:
    """
    Weighted MSE loss for complex
    """
    def __init__(self, reduction='mean'):
        self.reduction=reduction

    def __call__(self, outputs, targets, weights):

        B, T, C, H, W = targets.shape
        if(C==2):
            diff_mag_square = torch.square(outputs[:,:,0,:,:]-targets[:,:,0,:,:]) + torch.square(outputs[:,:,1,:,:]-targets[:,:,1,:,:])
        else:
            diff_mag_square = torch.square(outputs-targets)

        if(weights is not None):
            if(weights.ndim==2):
                weights = torch.reshape(weights, (outputs.shape[0], T, 1, 1))
                v_l2 = torch.sum(weights.reshape(B,T,1,1,1) * diff_mag_square.reshape(B,T,1,H,W)) / torch.sum(weights)
            else:
                weights = torch.reshape(weights, (outputs.shape[0], 1, 1, 1))
                v_l2 = torch.sum(weights.reshape(B,1,1,1,1) * diff_mag_square.reshape(B,T,1,H,W)) / torch.sum(weights)
        else:
            v_l2 = torch.sum(diff_mag_square.reshape(B,T,1,H,W))

        return torch.sqrt(v_l2) / (B*T) / (H*W)

class PSNR:
    """
    PSNR for metric comparison
    """
    def __init__(self, reduction='mean'):
        self.reduction=reduction

    def __call__(self, outputs, targets):

        return -4.342944819 * torch.log(torch.mean(torch.square(targets - outputs)))

# -------------------------------------------------------------------------------------------------

class Image_Enhancement_Combined_Loss:
    """Combined loss for image enhancement
    """

    def __init__(self):
        self.losses = []

    def add_loss(self, a_loss, w=1.0, condition=0):
        self.losses.append((a_loss, w, condition))

    def __call__(self, outputs, targets, weights=None, inputs=None, epoch=None): # adding epoch here to keep code consistent

        assert len(self.losses) > 0

        if self.losses[0][2]: # loss [0][2] is an int but only 0 1 for now so can work as a condition
            combined_loss = self.losses[0][1] * self.losses[0][0](outputs=outputs, targets=inputs, weights=weights)
        else:
            combined_loss = self.losses[0][1] * self.losses[0][0](outputs=outputs, targets=targets, weights=None)

        for k in range(1,len(self.losses)):
            if self.losses[k][2]:
                combined_loss += self.losses[k][1] * self.losses[k][0](outputs=outputs, targets=inputs, weights=weights)
            else:
                combined_loss += self.losses[k][1] * self.losses[k][0](outputs=outputs, targets=targets, weights=None)

        return combined_loss

    def __str__(self):
        content = f"Image_Enhancement_Combined_Loss, {len(self.losses)} losses\n"
        for l in self.losses:
            content += f"loss - {type(l[0])}, weights {l[1]}\n"

        return content

# -------------------------------------------------------------------------------------------------
# A pytorch SSIM and SSIM 3D implementation

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True, device='cpu'):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.window = self.window.to(device=device, dtype=torch.float32)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

class SSIM3D(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True, device="cpu"):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)
        self.window = self.window.to(device=device, dtype=torch.float32)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel

        return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)

def ssim3D(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)
