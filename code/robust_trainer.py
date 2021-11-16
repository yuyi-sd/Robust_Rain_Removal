import os
import math
from decimal import Decimal
import utility
import IPython
import torch
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm
import scipy.io as sio 
import matplotlib
import matplotlib.pyplot as plt
import pylab
import numpy as np
import lpips

upper_limit, lower_limit = 255,0

def clamp(X, lower_limit, upper_limit):
    return torch.max(torch.min(X, upper_limit), lower_limit)

def attack_pgd(model, X, X_sr, epsilon, alpha, attack_iters, restarts, scale, attack_loss, with_mask=False, rain_threshold = 10, mask_norain = False):
    torch.cuda.empty_cache()
    if attack_loss == 'l_2':
        criterion = nn.MSELoss()
    elif attack_loss == 'l_1':
        criterion = nn.L1Loss()
    elif attack_loss == 'lpips':
        criterion = lpips.LPIPS(net='alex').cuda()
    max_loss = torch.zeros(X.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    output = X_sr
    if with_mask:
        mask = (X_sr - X).abs() > rain_threshold
        mask = mask.sum(dim = 1)
        mask = mask.repeat([1,3,1,1])
        mask = mask > 0 if not mask_norain else mask == 0
        mask = mask.float()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        delta.uniform_(-epsilon, epsilon)
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            if with_mask:
                delta = delta * mask
            robust_output = model([X + delta, 'Test'], scale)
            if attack_loss != 'lpips':
                loss = criterion(robust_output, output)
            else:
                loss = criterion(2 * robust_output/255 -1 ,2 * output/255 - 1)
            grad = torch.autograd.grad(loss, [delta])[0].detach()
            d = delta
            g = grad
            x = X
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data = d
        if with_mask:
            delta = delta * mask
        robust_output = model([X + delta, 'Test'], scale)
        if attack_loss != 'lpips':
                all_loss = criterion(robust_output, output)
        else:
            all_loss = criterion(2 * robust_output/255 -1 ,2 * output/255 - 1)
        if attack_loss == 'lpips':
            all_loss = all_loss.squeeze().squeeze()
        max_delta[all_loss.detach() >= max_loss] = delta.detach()[all_loss.detach() >= max_loss]
        max_loss = torch.max(max_loss, all_loss.detach())
        del delta, robust_output, loss, grad, d, g, x, all_loss
    if with_mask:
        return max_delta, mask
    else:
        return max_delta

def attack_pgd2(model, X, X_sr, epsilon, alpha, attack_iters, restarts, scale, attack_loss, with_mask=False, rain_threshold = 10, mask_norain = False):
    torch.cuda.empty_cache()
    if attack_loss == 'l_2':
        criterion = nn.MSELoss()
    elif attack_loss == 'l_1':
        criterion = nn.L1Loss()
    elif attack_loss == 'lpips':
        criterion = lpips.LPIPS(net='alex').cuda()
    max_loss = torch.zeros(X.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    output = X_sr
    if with_mask:
        mask = (X_sr - X).abs() > rain_threshold
        mask = mask.sum(dim = 1)
        mask = mask.repeat([1,3,1,1])
        mask = mask > 0 if not mask_norain else mask == 0
        mask = mask.float()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        delta.uniform_(-epsilon, epsilon)
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            if with_mask:
                delta = delta * mask
            robust_output = model([X + delta, 'Test'], scale)
            if attack_loss != 'lpips':
                loss = -1 * criterion(robust_output, output)
            else:
                loss = -1 * criterion(2 * robust_output/255 -1 ,2 * output/255 - 1)
            grad = torch.autograd.grad(loss, [delta])[0].detach()
            d = delta
            g = grad
            x = X
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data = d
        if with_mask:
                delta = delta * mask
        robust_output = model([X + delta, 'Test'], scale)
        if attack_loss != 'lpips':
                all_loss = criterion(robust_output, output)
        else:
            all_loss = criterion(2 * robust_output/255 -1 ,2 * output/255 - 1)
        if attack_loss == 'lpips':
            all_loss = all_loss.squeeze().squeeze()
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    if with_mask:
        return max_delta, mask
    else:
        return max_delta

def attack_pgd3(model, X, X_sr, epsilon, alpha, attack_iters, restarts, scale, mse_weight, with_mask=False, rain_threshold = 10, mask_norain = False):
    torch.cuda.empty_cache()
    criterion_l2 = nn.MSELoss()
    criterion_lpips = lpips.LPIPS(net='vgg').cuda()
    max_loss = -1e6 * torch.ones(X.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    output = X_sr
    if with_mask:
        mask = (X_sr - X).abs() > rain_threshold
        mask = mask.sum(dim = 1)
        mask = mask.repeat([1,3,1,1])
        mask = mask > 0 if not mask_norain else mask == 0
        mask = mask.float()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        delta.uniform_(-epsilon, epsilon)
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            if with_mask:
                delta = delta * mask
            robust_output = model([X + delta, 'Test'], scale)
            loss = criterion_lpips(2 * robust_output/255 -1 ,2 * output/255 - 1).squeeze().mean() - mse_weight * criterion_l2(robust_output/255, output/255)
            grad = torch.autograd.grad(loss, [delta])[0].detach()
            d = delta
            g = grad
            x = X
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data = d
        if with_mask:
            delta = delta * mask
        robust_output = model([X + delta, 'Test'], scale)
        all_loss = criterion_lpips(2 * robust_output/255 -1 ,2 * output/255 - 1).squeeze().mean() - mse_weight * criterion_l2(robust_output/255, output/255)
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    if with_mask:
        return max_delta, mask
    else:
        return max_delta

def attack_pgd4(model, X, X_sr, epsilon, alpha, attack_iters, restarts, scale, mse_weight, with_mask=False, rain_threshold = 10, mask_norain = False):
    torch.cuda.empty_cache()
    criterion_l2 = nn.MSELoss()
    criterion_lpips = lpips.LPIPS(net='vgg').cuda()
    max_loss = -1e6 * torch.ones(X.shape[0]).cuda()
    max_delta = torch.zeros_like(X).cuda()
    output = X_sr
    if with_mask:
        mask = (X_sr - X).abs() > rain_threshold
        mask = mask.sum(dim = 1)
        mask = mask.repeat([1,3,1,1])
        mask = mask > 0 if not mask_norain else mask == 0
        mask = mask.float()
    for _ in range(restarts):
        delta = torch.zeros_like(X).cuda()
        delta.uniform_(-epsilon, epsilon)
        delta = clamp(delta, lower_limit-X, upper_limit-X)
        delta.requires_grad = True
        for _ in range(attack_iters):
            if with_mask:
                delta = delta * mask
            robust_output = model([X + delta, 'Test'], scale)
            loss = -1 * criterion_lpips(2 * robust_output/255 -1 ,2 * X/255 - 1).squeeze().mean() - mse_weight * criterion_l2(robust_output/255, output/255)
            grad = torch.autograd.grad(loss, [delta])[0].detach()
            d = delta
            g = grad
            x = X
            d = torch.clamp(d + alpha * torch.sign(g), min=-epsilon, max=epsilon)
            d = clamp(d, lower_limit - x, upper_limit - x)
            delta.data = d
        if with_mask:
            delta = delta * mask
        robust_output = model([X + delta, 'Test'], scale)
        all_loss = -1 * criterion_lpips(2 * robust_output/255 -1 ,2 * X/255 - 1).squeeze().mean() - mse_weight * criterion_l2(robust_output/255, output/255)
        max_delta[all_loss >= max_loss] = delta.detach()[all_loss >= max_loss]
        max_loss = torch.max(max_loss, all_loss)
    if with_mask:
        return max_delta, mask
    else:
        return max_delta

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.scheduler = utility.make_scheduler(args, self.optimizer)
        if self.args.load != '.':
            self.optimizer.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer.pt'))
            )
            for _ in range(len(ckp.log)): self.scheduler.step()

        self.error_last = 1e8

    def test(self):
        epoch = self.scheduler.last_epoch #+ 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale),3))
        self.model.eval()

        epsilon = (self.args.robust_epsilon * (self.args.rgb_range / 255.))
        alpha = (self.args.robust_alpha * (self.args.rgb_range / 255.))

        timer_test = utility.timer()
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                eval_acc = 0
                self.loader_test.dataset.set_scale(idx_scale)
                tqdm_test = tqdm(self.loader_test, ncols=80)
                for idx_img, (lr, hr, filename) in enumerate(tqdm_test):
                    torch.cuda.empty_cache()
                    filename = filename[0]
                    no_eval = (hr.nelement() == 1)
                    if not no_eval:
                        lr, hr = self.prepare(lr, hr)
                    else:
                        lr, = self.prepare(lr)

                    lr = torch.autograd.Variable(lr)
                    if self.args.attack_gt:
                        sr = hr
                    else:
                        sr = self.model([lr, 'Test'], idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)    # restored background at the last stage

                    with torch.enable_grad():
                        if self.args.target == 'output':
                            if self.args.attack == 'pgd':
                                delta = attack_pgd(self.model, lr, sr, epsilon, alpha, self.args.attack_iters, self.args.restarts, idx_scale, self.args.attack_loss, self.args.with_mask, self.args.rain_threshold, self.args.mask_norain)
                            else:
                                delta = attack_pgd(self.model, lr, sr, epsilon, epsilon / self.args.attack_iters, self.args.attack_iters, self.args.restarts, idx_scale, self.args.attack_loss, self.args.with_mask, self.args.rain_threshold, self.args.mask_norain)
                        elif self.args.target == 'down_stream':
                            if self.args.attack == 'pgd':
                                delta = attack_pgd3(self.model, lr, sr, epsilon, alpha, self.args.attack_iters, self.args.restarts, idx_scale, self.args.mse_weight, self.args.with_mask, self.args.rain_threshold, self.args.mask_norain)
                            else:
                                delta = attack_pgd3(self.model, lr, sr, epsilon, epsilon / self.args.attack_iters, self.args.attack_iters, self.args.restarts, idx_scale, self.args.mse_weight, self.args.with_mask, self.args.rain_threshold, self.args.mask_norain)
                        elif self.args.target == 'down_stream_v2':
                            if self.args.attack == 'pgd':
                                delta = attack_pgd4(self.model, lr, sr, epsilon, alpha, self.args.attack_iters, self.args.restarts, idx_scale, self.args.mse_weight, self.args.with_mask, self.args.rain_threshold, self.args.mask_norain)
                            else:
                                delta = attack_pgd4(self.model, lr, sr, epsilon, epsilon / self.args.attack_iters, self.args.attack_iters, self.args.restarts, idx_scale, self.args.mse_weight, self.args.with_mask, self.args.rain_threshold, self.args.mask_norain)
                        else:
                            if self.args.attack == 'pgd':
                                delta = attack_pgd2(self.model, lr, lr, epsilon, alpha, self.args.attack_iters, self.args.restarts, idx_scale, self.args.attack_loss, self.args.with_mask, self.args.rain_threshold, self.args.mask_norain)
                            else:
                                delta = attack_pgd2(self.model, lr, lr, epsilon, epsilon / self.args.attack_iters, self.args.attack_iters, self.args.restarts, idx_scale, self.args.attack_loss, self.args.with_mask, self.args.rain_threshold, self.args.mask_norain)
                        if self.args.with_mask:
                            mask = delta[1]
                            delta = delta[0]
                    lr_attack = lr + delta
                    sr_attack = self.model([lr_attack, 'Test'], idx_scale)
                    sr_attack = utility.quantize(sr_attack, self.args.rgb_range)

                    if self.args.attack_gt:
                        sr = self.model([lr, 'Test'], idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range) 

                    save_list = [sr.detach()]
                    if not no_eval:
                        self.ckp.log[-1, idx_scale,0] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, benchmark=self.loader_test.dataset.benchmark
                        )
                        self.ckp.log[-1, idx_scale,1] += utility.calc_psnr(
                            sr, sr_attack, scale, self.args.rgb_range, benchmark=self.loader_test.dataset.benchmark
                        )
                        self.ckp.log[-1, idx_scale,2] += utility.calc_psnr(
                            lr, lr_attack, 1, self.args.rgb_range, benchmark=self.loader_test.dataset.benchmark
                        )
                        if self.args.save_gt:
                            save_list.extend([lr.detach(), hr.detach()])

                        if self.args.save_attack:
                            save_list.extend([sr_attack.detach(), lr_attack.detach()])

                        if self.args.with_mask:
                            save_list.extend([mask.detach().mul(self.args.rgb_range)])

                    if self.args.save_results:
                        self.ckp.save_results(filename, save_list, scale)

                    del lr, hr, sr, delta, lr_attack, sr_attack

                self.ckp.log[-1, idx_scale] /= len(self.loader_test)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale,0],
                        best[0][idx_scale,0],
                        best[1][idx_scale,0] + 1
                    )
                )
                self.ckp.write_log(
                    '[{} x{}]\tsr PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale,1],
                        best[0][idx_scale,1],
                        best[1][idx_scale,1] + 1
                    )
                )
                self.ckp.write_log(
                    '[{} x{}]\tlr PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        self.args.data_test,
                        scale,
                        self.ckp.log[-1, idx_scale,2],
                        best[0][idx_scale,2],
                        best[1][idx_scale,2] + 1
                    )
                )

        self.ckp.write_log(
            'Total time: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0] + 1 == epoch))

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda:0')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)
           
        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch #+ 1
            return epoch >= self.args.epochs