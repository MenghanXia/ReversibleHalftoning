import os, glob, datetime, time
import argparse, json

import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torch.utils.data import DataLoader
from torch.backends import cudnn

from model.base_module import tensor2array
from model.model import ResHalf
from model.loss import *
from utils.dataset import HalftoneVOC2012 as Dataset
from utils.util import ensure_dir, save_list, save_images_from_batch


class Trainer():
    def __init__(self, config, resume):
        self.config = config
        self.name = config['name']
        self.resume_path = resume
        self.n_epochs = config['trainer']['epochs']
        self.with_cuda = config['cuda'] and torch.cuda.is_available()
        self.seed = config['seed']
        self.start_epoch = 0
        self.save_freq = config['trainer']['save_epochs']
        self.checkpoint_dir = os.path.join(config['save_dir'], self.name)
        ensure_dir(self.checkpoint_dir)
        json.dump(config, open(os.path.join(self.checkpoint_dir, 'config.json'), 'w'),
                  indent=4, sort_keys=False)
        print("@Workspace: %s *************"%self.checkpoint_dir)
        self.cache = os.path.join(self.checkpoint_dir, 'train_cache')
        self.val_halftone = os.path.join(self.cache, 'halftone')
        self.val_restored = os.path.join(self.cache, 'restored')
        ensure_dir(self.val_halftone)
        ensure_dir(self.val_restored)

        ## model        
        self.model = eval(config['model'])(train=True, warm_stage=True)
        if self.config['multi-gpus']:
            self.model = torch.nn.DataParallel(self.model).cuda()
        elif self.with_cuda:
            self.model = self.model.cuda()

        ## optimizer
        self.optimizer = getattr(optim, config['optimizer_type'])(self.model.parameters(), **config['optimizer'])
        self.lr_sheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, **config['lr_sheduler'])
        
        ## dataset loader
        with open(os.path.join(config['data_dir'], config['data_loader']['dataset'])) as f:
            dataset = json.load(f)
        train_set = Dataset(dataset['train'])
        self.train_data_loader = DataLoader(train_set, batch_size=config['data_loader']['batch_size'],
                                   shuffle=config['data_loader']['shuffle'],
                                   num_workers=config['data_loader']['num_workers'])
        val_set = Dataset(dataset['val'])
        self.valid_data_loader = DataLoader(val_set, batch_size=config['data_loader']['batch_size'],
                                   shuffle=False,
                                   num_workers=config['data_loader']['num_workers'])
        # special dataloader: constant color images
        with open(os.path.join(config['data_dir'], config['data_loader']['special_set'])) as f:
            dataset = json.load(f)
        specialSet = Dataset(dataset['train'])
        self.specialDataloader = DataLoader(specialSet, batch_size=config['data_loader']['batch_size'],
                                    shuffle=config['data_loader']['shuffle'],
                                    num_workers=config['data_loader']['num_workers'])

        ## loss function
        self.quantizeLoss = eval(config['quantizeLoss'])
        self.quantizeLossWeight = config['quantizeLossWeight']
        self.toneLoss = eval(config['toneLoss'])
        self.toneLossWeight = config['toneLossWeight']
        self.structureLoss = eval(config['structureLoss'])
        self.structureLossWeight = config['structureLossWeight']
        # quantize [-1,1] data to be {-1,1}
        self.quantizer = lambda x: Quantize.apply(0.5 * (x + 1.)) * 2. - 1.
        self.blueNoiseLossWeight = config['blueNoiseLossWeight']
        self.featureLoss = FeatureLoss(
             requireGrad=False, pretrainedPath='checkpoints/invhalftone_checkpt/model_best.pth.tar')
        self.featureLossWeight = config['featureLossWeight']

        # resume checkpoint
        if self.resume_path:
            assert os.path.exists(resume_path), 'Invalid checkpoint Path: %s' % resume_path
            self.load_checkpoint(self.resume_path)
        
    
    def _train(self):
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        cudnn.benchmark = True
        
        start_time = time.time()
        self.monitor_best = 999.
        for epoch in range(self.start_epoch, self.n_epochs + 1):
            ep_st = time.time()
            epoch_loss = self._train_epoch(epoch)
            # perform lr_sheduler
            self.lr_sheduler.step(epoch_loss['total_loss'])
            epoch_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            epoch_metric = self._valid_epoch(epoch)
            print("[*] --- epoch: %d/%d | loss: %4.4f | metric: %4.4f | time-consumed: %4.2f ---" % \
                (epoch+1, self.n_epochs, epoch_loss['total_loss'], epoch_metric, (time.time()-ep_st)))

            # save losses and learning rate
            epoch_loss['metric'] = epoch_metric
            epoch_loss['lr'] = epoch_lr
            self.save_loss(epoch_loss, epoch)
            if ((epoch+1) % self.save_freq == 0 or epoch == (self.n_epochs-1)):
                print('---------- saving model ...')
                self.save_checkpoint(epoch)
            if self.monitor_best > epoch_metric:
                self.monitor_best = epoch_metric
                self.save_checkpoint(epoch, save_best=True)

        print("Training finished! consumed %f sec" % (time.time() - start_time))


    def _to_variable(self, data, target):
        data, target = Variable(data), Variable(target)
        if self.with_cuda:
            data, target = data.cuda(), target.cuda()
        return data, target


    def _train_epoch(self, epoch):
        self.model.train()
        total_loss, quantize_loss, feature_loss = 0, 0, 0
        tone_loss, structure_loss, blue_noise_loss = 0, 0, 0

        specialIter = iter(self.specialDataloader)
        time_stamp = time.time()
        for batch_idx, (color, halftone) in enumerate(self.train_data_loader):
            color, halftone = self._to_variable(color, halftone)
            # special data
            try:
                specialColor, specialHalftone = next(specialIter)
            except StopIteration:
                # reinitialize data loader
                specialIter = iter(self.specialDataloader)
                specialColor, specialHalftone = next(specialIter)
            specialColor, specialHalftone = self._to_variable(specialColor, specialHalftone)
            self.optimizer.zero_grad()
            output = self.model(color, halftone)
            quantizeLoss = self.quantizeLoss(output[0])
            toneLoss = self.toneLoss(output[0], color)
            structureLoss = self.structureLoss(output[0], color)
            featureLoss = self.featureLoss(output[0], bgr2gray(color))

            # special data
            output = self.model(specialColor, specialHalftone)
            toneLossSpecial = self.toneLoss(output[0], specialColor)
            blueNoiseLoss = l1_loss(output[1], output[2])
            quantizeLossSpecial = self.quantizeLoss(output[0])
            loss = (self.toneLossWeight * toneLoss + self.blueNoiseLossWeight*toneLossSpecial) \
                   + self.quantizeLossWeight * (0.5*quantizeLoss + 0.5*quantizeLossSpecial) \
                   + self.structureLossWeight * structureLoss \
                   + self.blueNoiseLossWeight * blueNoiseLoss \
                   + self.featureLossWeight * featureLoss
                   
            loss.backward()
            # apply grad clip to make training roboust
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.0001)
            self.optimizer.step()

            total_loss += loss.item()
            quantize_loss += quantizeLoss.item()
            feature_loss += featureLoss.item()
            tone_loss += toneLoss.item()
            structure_loss += structureLoss.item()
            blue_noise_loss += blueNoiseLoss.item()
            if batch_idx % 100 == 0:
                tm = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
                print("%s >> [%d/%d] iter:%d loss:%4.4f "%(tm, epoch+1, self.n_epochs, batch_idx+1, loss.item()))

        epoch_loss = dict()
        epoch_loss['total_loss'] = total_loss / (batch_idx+1)
        epoch_loss['quantize_loss'] = quantize_loss / (batch_idx+1)
        epoch_loss['tone_loss'] = tone_loss / (batch_idx+1)
        epoch_loss['structure_loss'] = structure_loss / (batch_idx+1)
        epoch_loss['bluenoise_loss'] = blue_noise_loss / (batch_idx+1)
        epoch_loss['feature_loss'] = feature_loss / (batch_idx+1)

        return epoch_loss


    def _valid_epoch(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch_idx, (color, halftone) in enumerate(self.valid_data_loader):
                color, halftone = self._to_variable(color, halftone)
                output = self.model(color, halftone)
                quantizeLoss = self.quantizeLoss(output[0])
                toneLoss = self.toneLoss(output[0], color)
                structureLoss = self.structureLoss(output[0], color)
                featureLoss = self.featureLoss(output[0], bgr2gray(color))

                loss = self.toneLossWeight * toneLoss \
                       + self.quantizeLossWeight * quantizeLoss \
                       + self.structureLossWeight * structureLoss \
                       + self.featureLossWeight * featureLoss

                total_loss += loss.item()
                #! save intermediate images
                gray_imgs = tensor2array(output[0])
                color_imgs = tensor2array(output[-1])
                save_images_from_batch(gray_imgs, self.val_halftone, None, batch_idx)
                save_images_from_batch(color_imgs, self.val_restored, None, batch_idx)
            
            return total_loss


    def save_loss(self, epoch_loss, epoch):
        if epoch == 0:
            for key in epoch_loss:
                save_list(os.path.join(self.cache, key), [epoch_loss[key]], append_mode=False)
        else:
            for key in epoch_loss:
                save_list(os.path.join(self.cache, key), [epoch_loss[key]], append_mode=True)


    def load_checkpoint(self, checkpt_path):
        print("-loading checkpoint from: {} ...".format(checkpt_path))
        checkpoint = torch.load(checkpt_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.monitor_best = checkpoint['monitor_best']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print("-pretrained checkpoint loaded.")

    
    def save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'monitor_best': self.monitor_best
        }
        save_path = os.path.join(self.checkpoint_dir, 'model_last.pth.tar')
        if save_best:
            save_path = os.path.join(self.checkpoint_dir, 'model_best.pth.tar')
        torch.save(state, save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='InvHalf')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')
    args = parser.parse_args()
    config_dict = json.load(open(args.config))
    node = Trainer(config_dict, resume=args.resume)
    node._train()