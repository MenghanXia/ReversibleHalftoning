import os
import numpy as np
import mmcv
import torch

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_filelist(data_dir):
    file_list = glob.glob(os.path.join(data_dir, '*.*'))
    file_list.sort()
    return file_list
    

def collect_filenames(data_dir):
    file_list = get_filelist(data_dir)
    name_list = []
    for file_path in file_list:
        _, file_name = os.path.split(file_path)
        name_list.append(file_name)
    name_list.sort()
    return name_list


def save_list(save_path, data_list, append_mode=False):
    n = len(data_list)
    if append_mode:
        with open(save_path, 'a') as f:
            f.writelines([str(data_list[i]) + '\n' for i in range(n-1,n)])
    else:
        with open(save_path, 'w') as f:
            f.writelines([str(data_list[i]) + '\n' for i in range(n)])
    return None


def save_images_from_batch(img_batch, save_dir, filename_list, batch_no=-1):
    N,H,W,C = img_batch.shape
    if C == 3:
        #! rgb color image
        for i in range(N):
            # [-1,1] >>> [0,255]
            img_batch_i = np.clip(img_batch[i,:,:,:]*0.5+0.5, 0, 1)
            image = (255.0*img_batch_i).astype(np.uint8)
            save_name = filename_list[i] if batch_no==-1 else '%05d.png' % (batch_no*N+i)
            mmcv.imwrite(image, os.path.join(save_dir, save_name))
    elif C == 1:
        #! single-channel gray image
        for i in range(N):
            # [-1,1] >>> [0,255]
            img_batch_i = np.clip(img_batch[i,:,:,0]*0.5+0.5, 0, 1)
            image = (255.0*img_batch_i).astype(np.uint8)
            save_name = filename_list[i] if batch_no==-1 else '%05d.png' % (batch_no*img_batch.shape[0]+i)
            mmcv.imwrite(image, os.path.join(save_dir, save_name))
    return None


def imagesc(nd_array):
    plt.imshow(nd_array)
    plt.colorbar()
    plt.show()


def img2tensor(img):
    if len(img.shape) == 2:
        img = img[..., np.newaxis]
    img_t = np.expand_dims(img.transpose(2, 0, 1), axis=0)
    img_t = torch.from_numpy(img_t.astype(np.float32))
    return img_t


def tensor2img(img_t):
    img = img_t[0].detach().to("cpu").numpy()
    img = np.transpose(img, (1, 2, 0))
    if img.shape[-1] == 1:
        img = img[..., 0]
    return img
