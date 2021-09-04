import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .hourglass import HourGlass
from utils.dct import DCT_Lowfrequency
from utils.filters_tensor import bgr2gray

from collections import OrderedDict
import numpy as np


class Quantize(Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        y = x.round()
        return y

    @staticmethod
    def backward(ctx, grad_output):
        inputX = ctx.saved_tensors
        return grad_output


class ResHalf(nn.Module):
    def __init__(self, train=True, warm_stage=False):
        super(ResHalf, self).__init__()
        self.encoder = HourGlass(inChannel=4, outChannel=1, resNum=4, convNum=4)
        self.decoder = HourGlass(inChannel=1, outChannel=3, resNum=4, convNum=4)
        self.dcter = DCT_Lowfrequency(size=256, fLimit=50)
        # quantize [-1,1] data to be {-1,1}
        self.quantizer = lambda x: Quantize.apply(0.5 * (x + 1.)) * 2. - 1.
        self.isTrain = train
        if warm_stage:
            for name, param in self.decoder.named_parameters():
                param.requires_grad = False

    def add_impluse_noise(self, input_halfs, p=0.0):
        N,C,H,W = input_halfs.shape
        SNR = 1-p
        np_input_halfs = input_halfs.detach().to("cpu").numpy()
        np_input_halfs = np.transpose(np_input_halfs, (0, 2, 3, 1))
        for i in range(N):
            mask = np.random.choice((0, 1, 2), size=(H, W, 1), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
            np_input_halfs[i, mask==1] = 1.0
            np_input_halfs[i, mask==2] = -1.0
        return torch.from_numpy(np_input_halfs.transpose((0, 3, 1, 2))).to(input_halfs.device)

    def forward(self, *x):
        # x[0]: color_image
        # x[1]: ref_halftone
        noise = torch.randn_like(x[1]) * 0.3
        halfRes = self.encoder(torch.cat((x[0], noise), dim=1))
        #halfRes = self.encoder(torch.cat((input_tensor+noise_map, input_tensor-noise_map), dim=1))
        halfResQ = self.quantizer(halfRes)
        #! for testing only
        #halfResQ = self.add_impluse_noise(halfResQ, p=0.20)
        restored = self.decoder(halfResQ)
        if self.isTrain:
            halfDCT = self.dcter(halfRes / 2. + 0.5)
            refDCT = self.dcter(bgr2gray(x[0] / 2. + 0.5))
            return halfRes, halfDCT, refDCT, restored
        else:
            return halfRes, restored