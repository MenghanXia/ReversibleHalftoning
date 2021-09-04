import os
import torch.nn.functional as F
import torch
from utils.filters_tensor import GaussianSmoothing, bgr2gray
from utils import pytorch_ssim
from torch import nn
from .hourglass import HourGlass
from torchvision.models.vgg import vgg19


def l2_loss(y_input, y_target):
    return F.mse_loss(y_input, y_target)


def l1_loss(y_input, y_target):
    return F.l1_loss(y_input, y_target)


def gaussianL2(yInput, yTarget):
    # data range [-1,1]
    smoother = GaussianSmoothing(channels=1, kernel_size=11, sigma=2.0)
    gaussianInput = smoother(yInput)
    gaussianTarget = smoother(bgr2gray(yTarget))
    return F.mse_loss(gaussianInput, gaussianTarget)


def binL1(yInput):
    # data range is [-1,1]
    return (yInput.abs() - 1.0).abs().mean()


def ssimLoss(yInput, yTarget):
    # data range is [-1,1]
    ssim = pytorch_ssim.ssim(yInput / 2. + 0.5, bgr2gray(yTarget / 2. + 0.5), window_size=11)
    return 1. - ssim


class InverseHalf(nn.Module):
    def __init__(self):
        super(InverseHalf, self).__init__()
        self.net = HourGlass(inChannel=1, outChannel=1)

    def forward(self, x):
        grayscale = self.net(x)
        return grayscale


class FeatureLoss:
    def __init__(self, pretrainedPath, requireGrad=False, multiGpu=True):
        self.featureExactor = InverseHalf()
        if multiGpu:
            self.featureExactor = torch.nn.DataParallel(self.featureExactor).cuda()
        print("-loading feature extractor: {} ...".format(pretrainedPath))
        checkpoint = torch.load(pretrainedPath)
        self.featureExactor.load_state_dict(checkpoint['state_dict'])
        print("-feature network loaded")
        if not requireGrad:
            for param in self.featureExactor.parameters():
                param.requires_grad = False

    def __call__(self, yInput, yTarget):
        inFeature = self.featureExactor(yInput)
        return l2_loss(inFeature, yTarget)


class Vgg19Loss:
    def __init__(self, multiGpu=True):
        os.environ['TORCH_HOME']='~/bigdata/0ProgramS/checkpoints'
        # data in BGR format, [0,1] range
        self.mean = [0.485, 0.456, 0.406]
        self.mean.reverse()
        self.std = [0.229, 0.224, 0.225]
        self.std.reverse()
        vgg = vgg19(pretrained=True)
        # maxpoll after conv4_4
        self.featureExactor = nn.Sequential(*list(vgg.features)[:28]).eval()
        for param in self.featureExactor.parameters():
            param.requires_grad = False
        if multiGpu:
            self.featureExactor = torch.nn.DataParallel(self.featureExactor).cuda()
        print('[*] Vgg19Loss init!')

    def normalize(self, tensor):
        tensor = tensor.clone()
        mean = torch.as_tensor(self.mean, dtype=torch.float32, device=tensor.device)
        std = torch.as_tensor(self.std, dtype=torch.float32, device=tensor.device)
        tensor.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
        return tensor

    def __call__(self, yInput, yTarget):
        inFeature = self.featureExactor(self.normalize(yInput).flip(1))
        targetFeature = self.featureExactor(self.normalize(yTarget).flip(1))
        return l2_loss(inFeature, targetFeature)
