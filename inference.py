import numpy as np
import cv2
import os, argparse, json
from os.path import join
from glob import glob

import torch
import torch.nn.functional as F

from model.model import ResHalf
from model.model import Quantize
from model.loss import l1_loss
from utils import util
from utils.dct import DCT_Lowfrequency
from utils.filters_tensor import bgr2gray
from collections import OrderedDict

class Inferencer:
    def __init__(self, checkpoint_path, model, use_cuda=True, multi_gpu=True):
        self.checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.use_cuda = use_cuda
        self.model = model.eval()
        if multi_gpu:
            self.model = torch.nn.DataParallel(self.model)
            state_dict = self.checkpoint['state_dict']
        else:
            ## remove keyword "module" in the state_dict
            state_dict = OrderedDict()
            for k, v in self.checkpoint['state_dict'].items():
                name = k[7:]
                state_dict[name] = v 
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model.load_state_dict(state_dict)

    def __call__(self, input_img, decoding_only=False):
        with torch.no_grad():
            scale = 8
            _, _, H, W = input_img.shape
            if H % scale != 0 or W % scale != 0:
                input_img = F.pad(input_img, [0, scale - W % scale, 0, scale - H % scale], mode='reflect')
            if self.use_cuda:
                input_img = input_img.cuda()
            if decoding_only:
                resColor = self.model(input_img, decoding_only)
                if H % scale != 0 or W % scale != 0:
                    resColor = resColor[:, :, :H, :W]
                return resColor
            else:
                resHalftone, resColor = self.model(input_img, decoding_only)
                resHalftone = Quantize.apply((resHalftone + 1.0) * 0.5) * 2.0 - 1.
                if H % scale != 0 or W % scale != 0:
                    resHalftone = resHalftone[:, :, :H, :W]
                    resColor = resColor[:, :, :H, :W]
                return resHalftone, resColor


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='invHalf')
    parser.add_argument('--model', default=None, type=str,
                        help='model weight file path')
    parser.add_argument('--decoding', action='store_true', default=False, help='restoration from halftone input')
    parser.add_argument('--data_dir', default=None, type=str, 
                        help='where to load input data (RGB images)')
    parser.add_argument('--save_dir', default=None, type=str, 
                        help='where to save the result')
    args = parser.parse_args()

    invhalfer = Inferencer(
        checkpoint_path=args.model,
        model=ResHalf(train=False)        
    )
    save_dir = os.path.join(args.save_dir)
    util.ensure_dir(save_dir)
    test_imgs = glob(join(args.data_dir, '*.*g'))
    print('------loaded %d images.' % len(test_imgs) )
    for img in test_imgs:
        print('[*] processing %s ...' % img)
        if args.decoding:
            input_img = cv2.imread(img, flags=cv2.IMREAD_GRAYSCALE) / 127.5 - 1.
            c = invhalfer(util.img2tensor(input_img), decoding_only=True)
            c = util.tensor2img(c / 2. + 0.5) * 255.
            cv2.imwrite(join(save_dir, 'restored_' + img.split('/')[-1].split('.')[0] + '.png'), c)
        else:
            input_img = cv2.imread(img, flags=cv2.IMREAD_COLOR) / 127.5 - 1.
            h, c = invhalfer(util.img2tensor(input_img), decoding_only=False)
            h = util.tensor2img(h / 2. + 0.5) * 255.
            c = util.tensor2img(c / 2. + 0.5) * 255.
            cv2.imwrite(join(save_dir, 'halftone_' + img.split('/')[-1].split('.')[0] + '.png'), h)
            cv2.imwrite(join(save_dir, 'restored_' + img.split('/')[-1].split('.')[0] + '.png'), c)
