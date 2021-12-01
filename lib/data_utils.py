import numpy as np
from PIL import Image, ImageFilter
from skimage.measure import compare_ssim
import cv2
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import ConcatDataset
import math
import random
from torchvision.transforms import ToPILImage
from glob import *
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TrainsetLoader(Dataset):
    def __init__(self, trainset_dir_hr, trainset_dir_lr, upscale_factor, patch_size=32):
        super(TrainsetLoader).__init__()
        # print(trainset_dir_hr)
        self.trainset_dir_hr = trainset_dir_hr
        self.hr_dir = glob(self.trainset_dir_hr + "/*")
        self.trainset_dir_lr = trainset_dir_lr
        self.upscale_factor = upscale_factor
        self.patch_size = patch_size

    def __getitem__(self, idx):
        hr = self.hr_dir[idx]
        fname = hr.split("/")[-1]
        hrdirnum = hr.split("/")[-2]
        fnum = int(hr.split("/")[-1].split(".")[0])
        HR = Image.open(hr).convert('RGB')
        HR = np.array(HR, dtype=np.float32) / 255.0
        if fnum == 0:
            lrname0 = self.trainset_dir_lr + "/" + hrdirnum + "/" + fname
        elif fnum == 1:
            lrname0 = self.trainset_dir_lr + "/" + hrdirnum + "/" + str(fnum - 1).zfill(8) + ".png"
        else:
            lrname0 = self.trainset_dir_lr + "/" + hrdirnum + "/" + str(fnum - 2).zfill(8) + ".png"
        LR0 = Image.open(lrname0).convert('RGB')
        if fnum == 0:
            LR0 = LR0.filter(ImageFilter.GaussianBlur(radius=0.2))
        elif fnum == 1:
            LR0 = LR0.filter(ImageFilter.GaussianBlur(radius=0.1))
        LR0 = np.array(LR0, dtype=np.float32) / 255.0

        if fnum == 0:
            lrname1 = self.trainset_dir_lr + "/" + hrdirnum + "/" + fname
        else:
            lrname1 = self.trainset_dir_lr + "/" + hrdirnum + "/" + str(fnum - 1).zfill(8) + ".png"
        LR1 = Image.open(lrname1).convert('RGB')
        if fnum == 0:
            LR1 = LR1.filter(ImageFilter.GaussianBlur(radius=0.1))
        LR1 = np.array(LR1, dtype=np.float32) / 255.0

        lrname2 = self.trainset_dir_lr + "/" + hrdirnum + "/" + fname
        LR2 = Image.open(lrname2).convert('RGB')
        LR2 = np.array(LR2, dtype=np.float32) / 255.0

        if fnum == self.__len__()-1:
            lrname3 = self.trainset_dir_lr + "/" + hrdirnum + "/" + fname
        else:
            lrname3 = self.trainset_dir_lr + "/" + hrdirnum + "/" + \
                      str(fnum + 1).zfill(8) + ".png"
        LR3 = Image.open(lrname3).convert('RGB')
        if fnum == self.__len__() - 1:
            LR3 = LR3.filter(ImageFilter.GaussianBlur(radius=0.1))
        LR3 = np.array(LR3, dtype=np.float32) / 255.0

        if fnum == self.__len__()-1:
            lrname4 = self.trainset_dir_lr + "/" + hrdirnum + "/" + fname
        elif fnum == self.__len__()-2:
            lrname4 = self.trainset_dir_lr + "/" + hrdirnum + "/" + \
                      str(fnum + 1).zfill(8) + ".png"
        else:
            lrname4 = self.trainset_dir_lr + "/" + hrdirnum + "/" + \
                      str(fnum + 2).zfill(8) + ".png"
        LR4 = Image.open(lrname4).convert('RGB')
        if fnum == self.__len__() - 1:
            LR4 = LR4.filter(ImageFilter.GaussianBlur(radius=0.2))
        elif fnum == self.__len__()-2:
            LR4 = LR4.filter(ImageFilter.GaussianBlur(radius=0.1))
        LR4 = np.array(LR4, dtype=np.float32) / 255.0

        # crop patchs randomly
        HR, LR0, LR1, LR2, LR3, LR4 = random_crop(HR, LR0, LR1, LR2, LR3, LR4, self.patch_size, self.upscale_factor)

        LR0 = LR0[:, :, :, np.newaxis]
        LR1 = LR1[:, :, :, np.newaxis]
        LR2 = LR2[:, :, :, np.newaxis]
        LR3 = LR3[:, :, :, np.newaxis]
        LR4 = LR4[:, :, :, np.newaxis]
        LR = np.concatenate((LR0, LR1, LR2, LR3, LR4), axis=3)
        # data augmentation
        LR, HR = augumentation()(LR, HR)
        return toTensorLR(LR), toTensorHR(HR)
    def __len__(self):
        return len(self.hr_dir)

class VimeoTrainSetLoader(Dataset):
    def __init__(self, trainset_dir_hr, trainset_dir_lr, upscale_factor, patch_size=32):
        super(VimeoTrainSetLoader).__init__()
        self.trainset_dir_lr = trainset_dir_lr
        self.lr_dir = sorted(glob(self.trainset_dir_lr + "/*/*"))
        self.trainset_dir_hr = trainset_dir_hr
        self.upscale_factor = upscale_factor
        self.patch_size = patch_size

    def __getitem__(self, idx):
        lrdir = self.lr_dir[idx]
        hrdirnum = lrdir.split("/")[-2::]

        lrname0 = lrdir + "/" + "im2.png"
        LR0 = Image.open(lrname0).convert('RGB')
        LR0 = np.array(LR0, dtype=np.float32) / 255.0

        lrname1 = lrdir + "/" + "im3.png"
        LR1 = Image.open(lrname1).convert('RGB')
        LR1 = np.array(LR1, dtype=np.float32) / 255.0

        lrname2 = lrdir + "/" + "im4.png"
        LR2 = Image.open(lrname2).convert('RGB')
        LR2 = np.array(LR2, dtype=np.float32) / 255.0

        lrname3 = lrdir + "/" + "im5.png"
        LR3 = Image.open(lrname3).convert('RGB')
        LR3 = np.array(LR3, dtype=np.float32) / 255.0

        lrname4 = lrdir + "/" + "im6.png"
        LR4 = Image.open(lrname4).convert('RGB')
        LR4 = np.array(LR4, dtype=np.float32) / 255.0

        hrname = self.trainset_dir_hr + "/" + hrdirnum[0] + "/" + hrdirnum[1] + "/" + "im4.png"
        HR = Image.open(hrname).convert('RGB')
        HR = np.array(HR, dtype=np.float32) / 255.0

        # crop patchs randomly
        HR, LR0, LR1, LR2, LR3, LR4 = random_crop(HR, LR0, LR1, LR2, LR3, LR4, self.patch_size, self.upscale_factor)

        LR0 = LR0[:, :, :, np.newaxis]
        LR1 = LR1[:, :, :, np.newaxis]
        LR2 = LR2[:, :, :, np.newaxis]
        LR3 = LR3[:, :, :, np.newaxis]
        LR4 = LR4[:, :, :, np.newaxis]
        LR = np.concatenate((LR0, LR1, LR2, LR3, LR4), axis=3)
        # data augmentation
        LR, HR = augumentation()(LR, HR)
        return toTensorLR(LR), toTensorHR(HR)
    def __len__(self):
        return len(self.lr_dir)

class VimeoValSetLoader(Dataset):
    def __init__(self, valset_dir_hr, valset_dir_lr, upscale_factor, patch_size=32):
        super(VimeoValSetLoader).__init__()
        self.valset_dir_lr = valset_dir_lr
        self.lr_dir = sorted(glob(self.valset_dir_lr + "/*/*"))
        self.valset_dir_hr = valset_dir_hr
        self.upscale_factor = upscale_factor
        self.patch_size = patch_size

    def __getitem__(self, idx):
        lrdir = self.lr_dir[idx]
        hrdirnum = lrdir.split("/")[-2::]

        lrname0 = lrdir + "/" + "im2.png"
        LR0 = Image.open(lrname0).convert('RGB')
        LR0 = np.array(LR0, dtype=np.float32) / 255.0

        lrname1 = lrdir + "/" + "im3.png"
        LR1 = Image.open(lrname1).convert('RGB')
        LR1 = np.array(LR1, dtype=np.float32) / 255.0

        lrname2 = lrdir + "/" + "im4.png"
        LR2 = Image.open(lrname2).convert('RGB')
        LR2 = np.array(LR2, dtype=np.float32) / 255.0

        lrname3 = lrdir + "/" + "im5.png"
        LR3 = Image.open(lrname3).convert('RGB')
        LR3 = np.array(LR3, dtype=np.float32) / 255.0

        lrname4 = lrdir + "/" + "im6.png"
        LR4 = Image.open(lrname4).convert('RGB')
        LR4 = np.array(LR4, dtype=np.float32) / 255.0

        hrname = self.valset_dir_hr + "/" + hrdirnum[0] + "/" + hrdirnum[1] + "/" + "im4.png"
        HR = Image.open(hrname).convert('RGB')
        HR = np.array(HR, dtype=np.float32) / 255.0

        # crop patchs randomly
        HR, LR0, LR1, LR2, LR3, LR4 = random_crop(HR, LR0, LR1, LR2, LR3, LR4, self.patch_size, self.upscale_factor)

        LR0 = LR0[:, :, :, np.newaxis]
        LR1 = LR1[:, :, :, np.newaxis]
        LR2 = LR2[:, :, :, np.newaxis]
        LR3 = LR3[:, :, :, np.newaxis]
        LR4 = LR4[:, :, :, np.newaxis]
        LR = np.concatenate((LR0, LR1, LR2, LR3, LR4), axis=3)
        # data augmentation
        LR, HR = augumentation()(LR, HR)
        return toTensorLR(LR), toTensorHR(HR)
    def __len__(self):
        return len(self.lr_dir)

class ValidationsetLoader(Dataset):
    def __init__(self, val_dir_hr, val_dir_lr, upscale_factor, patch_size=48, random_cro=True):
        super(TrainsetLoader).__init__()
        self.val_dir_hr = val_dir_hr
        self.val_dir_lr = val_dir_lr
        self.hr_dir = sorted(glob(self.val_dir_hr + "/*"))
        self.hr_dir.pop(0)
        self.hr_dir.pop(-1)
        self.hr_dir.pop(0)
        self.hr_dir.pop(-1)
        self.upscale_factor = upscale_factor
        self.patch_size = patch_size
        self.random_cro = random_cro

    def __getitem__(self, idx):
        hr = self.hr_dir[idx]
        fname = hr.split("/")[-1]
        hrdirnum = hr.split("/")[-2]
        fnum = int(hr.split("/")[-1].split(".")[0])
        HR = Image.open(hr).convert('RGB')
        HR = np.array(HR, dtype=np.float32) / 255.0
        if fnum == 0:
            lrname0 = self.val_dir_lr + "/" + hrdirnum + "/" + fname
        elif fnum == 1:
            lrname0 = self.val_dir_lr + "/" + hrdirnum + "/" + str(fnum - 1).zfill(8) + ".png"
        else:
            lrname0 = self.val_dir_lr + "/" + hrdirnum + "/" + str(fnum - 2).zfill(8) + ".png"
        LR0 = Image.open(lrname0).convert('RGB')
        if fnum == 0:
            LR0 = LR0.filter(ImageFilter.GaussianBlur(radius=0.2))
        elif fnum == 1:
            LR0 = LR0.filter(ImageFilter.GaussianBlur(radius=0.1))
        LR0 = np.array(LR0, dtype=np.float32) / 255.0
        
        if fnum == 0:
            lrname1 = self.val_dir_lr + "/" + hrdirnum + "/" + fname
        else:
            lrname1 = self.val_dir_lr + "/" + hrdirnum + "/" + str(fnum - 1).zfill(8) + ".png"
        LR1 = Image.open(lrname1).convert('RGB')
        if fnum == 0:
            LR1 = LR1.filter(ImageFilter.GaussianBlur(radius=0.1))
        LR1 = np.array(LR1, dtype=np.float32) / 255.0

        lrname2 = self.val_dir_lr + "/" + hrdirnum + "/" + fname
        LR2 = Image.open(lrname2).convert('RGB')
        LR2 = np.array(LR2, dtype=np.float32) / 255.0

        if fnum == self.__len__() - 1:
            lrname3 = self.val_dir_lr + "/" + hrdirnum + "/" + fname
        else:
            lrname3 = self.val_dir_lr + "/" + hrdirnum + "/" + \
                      str(fnum + 1).zfill(8) + ".png"
        LR3 = Image.open(lrname3).convert('RGB')

        if fnum == self.__len__() - 1:
            LR3 = LR3.filter(ImageFilter.GaussianBlur(radius=0.1))
        LR3 = np.array(LR3, dtype=np.float32) / 255.0

        if fnum == self.__len__() - 1:
            lrname4 = self.val_dir_lr + "/" + hrdirnum + "/" + fname
        elif fnum == self.__len__() - 2:
            lrname4 = self.val_dir_lr + "/" + hrdirnum + "/" + \
                      str(fnum + 1).zfill(8) + ".png"
        else:

            lrname4 = self.val_dir_lr + "/" + hrdirnum + "/" + \
                      str(fnum + 2).zfill(8) + ".png"
        LR4 = Image.open(lrname4).convert('RGB')
        if fnum == self.__len__() - 1:
            LR4 = LR4.filter(ImageFilter.GaussianBlur(radius=0.2))
        elif fnum == self.__len__() - 2:
            LR4 = LR4.filter(ImageFilter.GaussianBlur(radius=0.1))
        LR4 = np.array(LR4, dtype=np.float32) / 255.0

        # crop patchs randomly
        if self.random_cro == True:
            HR, LR0, LR1, LR2, LR3, LR4 = random_crop(HR, LR0, LR1, LR2, LR3, LR4, self.patch_size, self.upscale_factor)

        LR0 = LR0[:, :, :, np.newaxis]
        LR1 = LR1[:, :, :, np.newaxis]
        LR2 = LR2[:, :, :, np.newaxis]
        LR3 = LR3[:, :, :, np.newaxis]
        LR4 = LR4[:, :, :, np.newaxis]
        LR = np.concatenate((LR0, LR1, LR2, LR3, LR4), axis=3)
        # data augmentation
        # LR, HR = augumentation()(LR, HR)
        return toTensorLR(LR), toTensorHR(HR)

    def __len__(self):
        return len(self.hr_dir)

class TestsetLoader(Dataset):
    def __init__(self, dataset_dir, gt_dir, upscale_factor, patch_size=48, random_cro=True):
        super(TrainsetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.gt_dir = gt_dir
        self.frames = sorted(glob(self.dataset_dir + "/*"))
        self.hr_dir = sorted(glob(self.gt_dir + "/*"))
        self.upscale_factor = upscale_factor
        self.patch_size = patch_size
        self.random_cro = random_cro

    def __getitem__(self, idx):
        path = self.frames[idx]
        fnum = int(path.split("/")[-1].split(" ")[-1].split(".")[0])
        GTpath = self.gt_dir + "/Frame " + str(fnum).zfill(2) + ".png"
        GT = Image.open(GTpath).convert('RGB')
        GT = np.array(GT, dtype=np.float32) / 255.0

        if fnum == 1:
            lrname0 = path
        elif fnum == 2:
            lrname0 = self.dataset_dir + "/Frame " + str(fnum - 1).zfill(2) + ".png"
        else:
            lrname0 = self.dataset_dir + "/Frame " + str(fnum - 2).zfill(2) + ".png"
        LR0 = Image.open(lrname0).convert('RGB')
        if fnum == 1:
            LR0 = LR0.filter(ImageFilter.GaussianBlur(radius=0.2))
        elif fnum == 2:
            LR0 = LR0.filter(ImageFilter.GaussianBlur(radius=0.1))
        LR0 = np.array(LR0, dtype=np.float32) / 255.0

        if fnum == 1:
            lrname1 = path
        else:
            lrname1 = self.dataset_dir + "/Frame " + str(fnum - 1).zfill(2) + ".png"
        LR1 = Image.open(lrname1).convert('RGB')
        if fnum == 1:
            LR1 = LR1.filter(ImageFilter.GaussianBlur(radius=0.1))
        LR1 = np.array(LR1, dtype=np.float32) / 255.0

        lrname2 = path
        LR2 = Image.open(lrname2).convert('RGB')
        LR2 = np.array(LR2, dtype=np.float32) / 255.0

        if fnum == self.__len__():
            lrname3 = path
        else:
            lrname3 = self.dataset_dir + "/Frame " + str(fnum + 1).zfill(2) + ".png"
        LR3 = Image.open(lrname3).convert('RGB')
        if fnum == self.__len__():
            LR3 = LR3.filter(ImageFilter.GaussianBlur(radius=0.1))
        LR3 = np.array(LR3, dtype=np.float32) / 255.0

        if fnum == self.__len__():
            lrname4 = path
        elif fnum == self.__len__() - 1:
            lrname4 = self.dataset_dir + "/Frame " + str(fnum + 1).zfill(2) + ".png"
        else:
            lrname4 = self.dataset_dir + "/Frame " + str(fnum + 2).zfill(2) + ".png"
        LR4 = Image.open(lrname4).convert('RGB')
        if fnum == self.__len__():
            LR4 = LR4.filter(ImageFilter.GaussianBlur(radius=0.2))
        elif fnum == self.__len__() - 1:
            LR4 = LR4.filter(ImageFilter.GaussianBlur(radius=0.1))
        LR4 = np.array(LR4, dtype=np.float32) / 255.0

        # LR0, LR1, LR2, LR3, LR4 = random_crop_lr(LR0, LR1, LR2, LR3, LR4, self.patch_size)
        if self.random_cro == True:
            GT, LR0, LR1, LR2, LR3, LR4 = random_crop(GT, LR0, LR1, LR2, LR3, LR4, self.patch_size, self.upscale_factor)

        LR0 = LR0[:, :, :, np.newaxis]
        LR1 = LR1[:, :, :, np.newaxis]
        LR2 = LR2[:, :, :, np.newaxis]
        LR3 = LR3[:, :, :, np.newaxis]
        LR4 = LR4[:, :, :, np.newaxis]
        LR = np.concatenate((LR0, LR1, LR2, LR3, LR4), axis=3)
        # data augmentation
        # LR, HR = augumentation()(LR, HR)
        return toTensorLR(LR), toTensorHR(GT)
    def __len__(self):
        return len(self.frames)

class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[:, ::-1, :, :]
            target = target[:, ::-1, :]
        if random.random()<0.5:
            input = input[::-1, :, :]
            target = target[::-1, :, :]
        if random.random()<0.5:
            input = input.transpose(1, 0, 2, 3)
            target = target.transpose(1, 0, 2)
        return np.ascontiguousarray(input), np.ascontiguousarray(target)

def random_crop(HR, LR0, LR1, LR2, LR3, LR4, patch_size_lr, upscale_factor):
    h_hr = HR.shape[1]    #64*112
    w_hr = HR.shape[0]
    h_lr = h_hr // upscale_factor   #256*448
    w_lr = w_hr // upscale_factor
    #print(w_lr)
    #print(patch_size_lr)
    idx_h = random.randint(10, h_lr - patch_size_lr - 10)
    #idx_w = random.randint(10, w_lr - patch_size_lr - 10)
    idx_w = random.randint(1, w_lr - patch_size_lr - 1)
    #idx_h = 30
    #idx_w = 30

    h_start_hr = (idx_h - 1) * upscale_factor
    h_end_hr = (idx_h - 1 + patch_size_lr) * upscale_factor
    w_start_hr = (idx_w - 1) * upscale_factor
    w_end_hr = (idx_w - 1 + patch_size_lr) * upscale_factor

    h_start_lr = idx_h - 1
    h_end_lr = idx_h - 1 + patch_size_lr
    w_start_lr = idx_w - 1
    w_end_lr = idx_w - 1 + patch_size_lr

    HR = HR[w_start_hr:w_end_hr, h_start_hr:h_end_hr,:]
    LR0 = LR0[w_start_lr:w_end_lr, h_start_lr:h_end_lr,:]
    LR1 = LR1[w_start_lr:w_end_lr, h_start_lr:h_end_lr,:]
    LR2 = LR2[w_start_lr:w_end_lr, h_start_lr:h_end_lr,:]
    LR3 = LR3[w_start_lr:w_end_lr, h_start_lr:h_end_lr, :]
    LR4 = LR4[w_start_lr:w_end_lr, h_start_lr:h_end_lr, :]
    return HR, LR0, LR1, LR2, LR3, LR4
def random_crop_lr(LR0, LR1, LR2, LR3, LR4, patch_size_lr):
    h_lr = LR0.shape[1]    #1280*720
    w_lr = LR1.shape[0]
    idx_h = random.randint(10, h_lr - patch_size_lr - 10) 
    idx_w = random.randint(10, w_lr - patch_size_lr - 10)

    h_start_lr = idx_h - 1
    h_end_lr = idx_h - 1 + patch_size_lr
    w_start_lr = idx_w - 1
    w_end_lr = idx_w - 1 + patch_size_lr

    LR0 = LR0[w_start_lr:w_end_lr, h_start_lr:h_end_lr,:]
    LR1 = LR1[w_start_lr:w_end_lr, h_start_lr:h_end_lr,:]
    LR2 = LR2[w_start_lr:w_end_lr, h_start_lr:h_end_lr,:]
    LR3 = LR3[w_start_lr:w_end_lr, h_start_lr:h_end_lr, :]
    LR4 = LR4[w_start_lr:w_end_lr, h_start_lr:h_end_lr, :]
    return LR0, LR1, LR2, LR3, LR4

def toTensorLR(img):
    #img = img[::-1,:,:,:].copy()
    img = torch.from_numpy(img.transpose((2, 3, 0, 1)))
    # img.float().div(255)
    return img
def toTensorHR(img):
    #img = img[::-1,:,:].copy()
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    # img.float().div(255)
    return img

def calcmeas(target, ref):
    #target_data = target[:, :, scale:-scale, scale:-scale]
    target_data = target
    ref_data = ref
    #ref_data = ref[:, :, scale:-scale, scale:-scale]
    psn = psnr(target_data,ref_data)
    ssim = SSIM(ref_data,target_data)
    return psn, ssim

def psnr(target, ref):
    # assume RGB image
    target_data = target       
    ref_data = ref
                  
    diff = ref_data - target_data
    diff = diff.flatten('C')
    rmse = math.sqrt( np.mean(diff ** 2.) )
    return 20*math.log10(1.0/rmse)

def SSIMnp(y_true , y_pred):
    u_true = np.mean(y_true)
    u_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    std_true = np.sqrt(var_true)
    std_pred = np.sqrt(var_pred)
    c1 = np.square(0.01*7)
    c2 = np.square(0.03*7)
    ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
    denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
    return ssim / denom
def SSIM(im1,im2):
    ssim_avg = 0.0
    for i in range(im1.shape[0]):
        img1 = im1[i,:,:,:]
        img1 = img1.transpose(1,2,0)
        img2 = im2[i,:,:,:]
        img2 = img2.transpose(1,2,0)
        ssim = compare_ssim(img1, img2, multichannel=True)
        ssim_avg += ssim
    ssim_avg /= im1.shape[0]
    return ssim_avg
