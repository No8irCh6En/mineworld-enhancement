import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np
import cv2
import argparse
import os
import glob
import json
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import random
from collections import defaultdict


class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride=stride),
                nn.BatchNorm2d(out_c)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class AttentionTokenPredictor(nn.Module):
    def __init__(self, input_channels=4, action_dim=11, num_tokens=8192, feature_dim=512): 
        super().__init__()
        self.feature_dim = feature_dim
        
        # 1. Visual Encoder (ResNet)
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, padding=1, stride=2), 
            nn.BatchNorm2d(32), nn.ReLU()
        )
        self.layer1 = ResBlock(32, 64, stride=2)   
        self.layer2 = ResBlock(64, 128, stride=2)  
        self.layer3 = ResBlock(128, 256, stride=2) 
        self.layer4 = ResBlock(256, feature_dim, stride=1) # Output: (B, 512, 14, 24)
        
        # 2. Action Encoder (MLP)
        self.action_mlp = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, feature_dim),
            nn.ReLU()
        )
        
        # 3. Attention Mechanism
        self.scale = feature_dim ** -0.5
        
        # 4. Output Heads
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, 3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU()
        )
        
        self.head_cls = nn.Conv2d(feature_dim, 1, 1)
        self.head_token = nn.Conv2d(feature_dim, num_tokens, 1) 
        
        nn.init.normal_(self.head_token.weight, std=0.01)
        nn.init.constant_(self.head_token.bias, 0)

    def forward(self, img_depth, action_vec, return_features=False): # [修改] 添加 return_features 参数
        # 1. Extract Visual Features (f)
        x = self.initial(img_depth)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        f_curr = self.layer4(x) # (B, 512, 14, 24)
        
        B, C, H, W = f_curr.shape
        
        # 2. Extract Action Features (a)
        a = self.action_mlp(action_vec) # (B, 512)
        a_expanded = a.view(B, C, 1, 1) # (B, 512, 1, 1)
        
        # 3. Compute Predicted Next Frame Features (f + a)
        f_pred = f_curr + a_expanded # (B, 512, 14, 24)
        
        # 4. Cross Attention
        Q = f_pred.view(B, C, -1).permute(0, 2, 1) # (B, N, C)
        K = f_curr.view(B, C, -1)                  # (B, C, N)
        V = f_curr.view(B, C, -1).permute(0, 2, 1) # (B, N, C)
        
        attn_scores = torch.bmm(Q, K) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1) # (B, N, N)
        
        attn_out = torch.bmm(attn_weights, V) 
        attn_out = attn_out.permute(0, 2, 1).view(B, C, H, W)
        
        # 5. Fusion & Prediction
        combined = torch.cat([f_pred, attn_out], dim=1) # (B, 1024, 14, 24)
        combined = self.fusion_conv(combined)           # (B, 512, 14, 24)
        
        logits_cls = self.head_cls(combined)
        logits_token = self.head_token(combined)

        if return_features:
            # [新增] 返回中间特征字典
            return logits_cls, logits_token, {
                "f_curr": f_curr,   # 当前帧视觉特征
                "f_pred": f_pred,   # 预测的下一帧特征 (Before Attention)
                "combined": combined # 最终融合特征 (After Attention)
            }
        
        return logits_cls, logits_token

