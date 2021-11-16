from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class RESLoss(nn.Module):
    def __init__(self):
        super(RESLoss, self).__init__()
        self.loss_his = {}
        self.loss_his['All'] = 0

    def forward(self, sr, hr):
        loss_fun = torch.nn.MSELoss(reduce=True, size_average=True)
        O_Bs, lr = sr

        loss = sum([loss_fun(O_B, hr) for O_B in O_Bs])
        self.loss_his['All'] += loss

        return loss

    def show_loss(self):
        print('All loss' + str(self.loss_his['All'].cpu().detach().numpy()))
