from model import common

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class JORDERLoss(nn.Module):
    def __init__(self):
        super(JORDERLoss, self).__init__()
        self.loss_his = {}

        self.loss_his['All'] = 0
        self.loss_his['Mask'] = 0
        self.loss_his['Level'] = 0
        self.loss_his['Rect'] = 0

    def forward(self, sr, hr):
        loss_fun = torch.nn.MSELoss(reduce=True, size_average=True)
        mask_loss = torch.nn.CrossEntropyLoss(reduce=False, size_average=True)

        sr1, sr2, mask, level, lr = sr

        w1 = 10e-4
        w2 = 10e-3

        per_pixel_detection_loss = mask_loss(mask, ((hr-lr)[:,0,:,:]>0).type(torch.cuda.LongTensor))
        per_pixel_detection_loss = per_pixel_detection_loss.sum()

        rect_loss = loss_fun(sr1, hr) + loss_fun(sr2, hr)
        detect_loss = w1*per_pixel_detection_loss
        rain_loss = w2 * loss_fun(level, hr-lr)

        loss = rect_loss + detect_loss + rain_loss 

        self.loss_his['All'] += loss
        self.loss_his['Rect'] += rect_loss
        self.loss_his['Mask'] += detect_loss
        self.loss_his['Level'] += rain_loss

        return loss

    def show_loss(self):
        print('All loss' + str(self.loss_his['All'].cpu().detach().numpy()))
        print('Mask loss' + str(self.loss_his['Mask'].cpu().detach().numpy()))
        print('Level loss' + str(self.loss_his['Level'].cpu().detach().numpy()))
        print('Rect loss' + str(self.loss_his['Rect'].cpu().detach().numpy()))
