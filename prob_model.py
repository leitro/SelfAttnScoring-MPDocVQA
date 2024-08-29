import glob
import cv2
from PIL import Image
import os
import numpy as np
import time
import random
from transformers import DistilBertModel
import torch

FACIL = False

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class ProbModule(torch.nn.Module):
    def __init__(self):
        super(ProbModule, self).__init__()
        self.avg = torch.nn.AdaptiveAvgPool1d(1)
        self.fc0 = torch.nn.Linear(16*16*3, 256)
        self.fc1 = torch.nn.Linear(256, 64)
        self.fc2 = torch.nn.Linear(64, 1)
        self.drop = torch.nn.Dropout(p=0.2)
        self.relu = torch.nn.LeakyReLU()
        enc_layer = torch.nn.TransformerEncoderLayer(d_model=768, nhead=8, dim_feedforward=768, dropout=0.5, batch_first=True)
        self.enc = torch.nn.TransformerEncoder(enc_layer, num_layers=1).to(DEVICE)

    # feat: batch, channel, length, mask: batch, length
    def forward(self, feat, mask):
        if FACIL: # too simple, model cannot learn enough features
            out = self.relu(feat)
            out = self.drop(out)
            out = self.avg(out)
            oout = out.squeeze(-1)
        else:
            bin_mask = (mask < 0.5) # put PAD(0.) to True and feature(1.) to False
            out = self.enc(torch.permute(feat, (0, 2, 1)), src_key_padding_mask=bin_mask) # batch, 2048, 768
            oout = out[:, 0, :] # batch, 768

        res = self.relu(oout)
        res = self.drop(res)
        res = self.fc0(res)
        res = self.fc1(res)
        res = self.fc2(res)
        return res.squeeze(-1)

