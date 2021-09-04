import torch
import torch.utils.data as data
import mmcv
import numpy as np
from os.path import join


class HalftoneVOC2012(data.Dataset):
    # data range is [-1,1], color image is in BGR format
    def __init__(self, data_list):
        super(HalftoneVOC2012, self).__init__()
        self.inputs = [join('Data', x) for x in data_list['inputs']]
        self.labels = [join('Data', x) for x in data_list['labels']]

    @staticmethod
    def load_input(name):
        img = mmcv.imread(name, 'color')
        # transpose data
        img = img.transpose((2, 0, 1))
        # to Tensor
        img = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0)
        return img

    @staticmethod
    def load_label(name):
        img = mmcv.imread(name, 'grayscale')
        # transpose data
        img = img[np.newaxis, :, :]
        # to Tensor
        img = torch.from_numpy(img.astype(np.float32) / 127.5 - 1.0)
        return img

    def __getitem__(self, index):
        input_data = self.load_input(self.inputs[index])
        label_data = self.load_label(self.labels[index])
        return input_data, label_data

    def __len__(self):
        return len(self.inputs)