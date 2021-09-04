import numpy as np
import mmcv
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


class Inferencer:
    def __init__(self, config, checkpoint_path, model):
        self.checkpoint = torch.load(checkpoint_path)
        self.use_cuda = config['cuda']
        self.model = model.eval()
        if config['multi-gpus']:
            self.model = torch.nn.DataParallel(self.model)
        if self.use_cuda:
            self.model = self.model.cuda()
        self.model.load_state_dict(self.checkpoint['state_dict'])

    def __call__(self, colorImage, refHalftone):
        with torch.no_grad():
            scale = 8
            _, _, H, W = colorImage.shape
            if H % scale != 0 or W % scale != 0:
                colorImage = F.pad(colorImage, [0, scale - W % scale, 0, scale - H % scale], mode='reflect')
                refHalftone = F.pad(refHalftone, [0, scale - W % scale, 0, scale - H % scale], mode='reflect')
            if self.use_cuda:
                colorImage = colorImage.cuda()
                refHalftone = refHalftone.cuda()
            res = self.model(colorImage, refHalftone)
            resHalftone = Quantize.apply((res[0] + 1.0) * 0.5) * 2.0 - 1.
            resColor = res[1]
            if H % scale != 0 or W % scale != 0:
                resHalftone = resHalftone[:, :, :H, :W]
                resColor = resColor[:, :, :H, :W]
            return resHalftone, resColor


def inference_testset(config, inferencer, testset, save_dir):
    save_path = save_dir
    print('============', save_path)
    util.ensure_dir(join(save_path, testset))
    test_imgs = glob(join(config['data_dir'], testset, 'target_c/*g'))
    print('------loaded %d images.' % len(test_imgs) )
    for img in test_imgs:
        print('[*] processing %s ...' % img)
        color = mmcv.imread(img, flag='color') / 127.5 - 1.
        # refHalf = mmcv.imread(img.replace('target_c', 'raw_ov'), flag='grayscale') / 127.5 - 1.
        refHalf = mmcv.imread(img, flag='grayscale') / 127.5 - 1.
        h, c = inferencer(util.img2tensor(color), util.img2tensor(refHalf))
        h = util.tensor2img(h / 2. + 0.5) * 255.
        c = util.tensor2img(c / 2. + 0.5) * 255.
        mmcv.imwrite(h, join(save_path, testset, 'halftone', img.split('/')[-1].split('.')[0] + '.png'))
        mmcv.imwrite(c, join(save_path, testset, 'restored', img.split('/')[-1].split('.')[0] + '.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='invHalf')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('--last', action='store_true', 
                        help='whether load the last model (default: best)')
    args = parser.parse_args()
    config_dict = json.load(open(args.config))
    workspace = os.path.join(config_dict['save_dir'], config_dict['name'])
    checkpoint = os.path.join(workspace, 'model_best.pth.tar')
    if args.last:
        checkpoint = os.path.join(workspace, 'model_last.pth.tar')

    # testsets = ['HalftoneVOC2012/test']
    # testsets = ['example', 'HalftoneVOC2012/val']
    testsets = ['example']   # new_sample
    Name = 'result'
    invhalfer = Inferencer(
        config=config_dict,
        checkpoint_path=checkpoint,
        model=ResHalf(train=False)
    )
    save_dir = os.path.join(workspace, Name)
    for test_set in testsets:
        inference_testset(config_dict, invhalfer, test_set, save_dir)
