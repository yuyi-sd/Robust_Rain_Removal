from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class RCDLoss(nn.Module):
    def __init__(self, step):
        super(RCDLoss, self).__init__()
        self.loss_his = {}
        self.loss_his['Bs'] = 0
        self.loss_his['B0'] = 0
        self.loss_his['Rs'] = 0
        self.loss_his['B'] = 0
        self.loss_his['R'] = 0
        self.loss_his['All'] = 0
        self.S  = step

    def forward(self, sr, hr):
        loss_fun = torch.nn.MSELoss(reduce=True, size_average=True)
        B0, ListB, ListR, lr = sr

        loss_Bs = 0
        loss_Rs = 0

        for j in range(self.S):
            loss_Bs = float(loss_Bs) + 0.1* loss_fun(ListB[j], hr)
            loss_Rs = float(loss_Rs) + 0.1* loss_fun(ListR[j], lr-hr)

        loss_B = loss_fun(ListB[-1], hr)
        loss_R = 0.9 * loss_fun(ListR[-1], lr-hr)
        loss_B0 = 0.1* loss_fun(B0, hr)

        self.loss_his['Bs'] += loss_Bs
        self.loss_his['B0'] += loss_B0
        self.loss_his['Rs'] += loss_Rs
        self.loss_his['B'] += loss_B
        self.loss_his['R'] += loss_R

        loss = loss_B0 + loss_Bs  + loss_Rs + loss_B + loss_R
        self.loss_his['All'] += loss

        return loss

    def show_loss(self):
        print('Bs loss: '+ str(self.loss_his['Bs'].cpu().detach().numpy()))
        print('B0 loss: '+ str(self.loss_his['B0'].cpu().detach().numpy()))
        print('Rs loss: '+ str(self.loss_his['Rs'].cpu().detach().numpy()))
        print('B loss: ' + str(self.loss_his['B'].cpu().detach().numpy()))
        print('R loss:'  + str(self.loss_his['R'].cpu().detach().numpy()))
        print('All loss' + str(self.loss_his['All'].cpu().detach().numpy()))
