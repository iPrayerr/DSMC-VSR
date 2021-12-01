# x4 test - reds4

import warnings
warnings.filterwarnings('ignore')

import os
from lib.SRnet import *
from lib.data_utils import calcmeas, ValidationsetLoader
from glob import *
from torch.autograd import Variable
from torchvision import transforms as trans
from torch.utils.data import ConcatDataset
from tqdm import tqdm

if __name__ == '__main__':
    frame_path = "/data/REDS/TEST/L"
    gt_path = "/data/REDS/TEST/H"
    weights_path = "./pretrained_weights/x4.pkl"
    batch_size = 8

    testset = ""
    #testdir = sorted(glob(gt_path+"/020"))
    testdir = sorted([gt_path+"/000", gt_path+"/011", gt_path+"/015", gt_path+"/020"])
    for tes in testdir:
        tset = ValidationsetLoader(val_dir_hr=tes, val_dir_lr=frame_path,
                                    upscale_factor=4, random_cro=False, patch_size=32)
        if testdir.index(tes)==0:
            testset = tset
        else:
            testset = ConcatDataset([testset,tset])
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=2)

    # load net
    net = SRNet(srsiz=4, block_config=(2,6,6,3))
    net.load_state_dict(torch.load(weights_path))
    net = net.to(device)
    
    resdir = "./results/"
    if not os.path.exists(resdir):
        os.makedirs(resdir)
    f = open(resdir+"metrics.log",'w+')
    resdir_0 = "./results/LR/"
    if not os.path.exists(resdir_0):
        os.makedirs(resdir_0)
    resdir_1 = "./results/bicubic/"
    if not os.path.exists(resdir_1):
        os.makedirs(resdir_1)
    resdir_2 = "./results/ours/"
    if not os.path.exists(resdir_2):
        os.makedirs(resdir_2)
    resdir_3 = "./results/GT/"
    if not os.path.exists(resdir_3):
        os.makedirs(resdir_3)
    
    # test
    PSNR_Avg = 0.0
    Pbic_Avg = 0.0
    SSIM_Avg = 0.0
    Sbic_Avg = 0.0
    for i, data in enumerate(tqdm(testloader,desc="Testing"), 0):
        l, gt = data
        l, gt = Variable(l.to(device)), Variable(gt.to(device))
        with torch.no_grad():
            h, _ = net(l)
            bich = F.interpolate(l[:, :, 2, :, :], scale_factor=4, mode='bicubic')
            one = torch.ones_like(h)
            zero = torch.zeros_like(h)
            h = torch.where(h > 1.0, one, h)
            h = torch.where(h < 0.0, zero, h)
            bich = torch.where(bich > 1.0, one, bich)
            bich = torch.where(bich < 0.0, zero, bich)
            PSNR, SSIM = calcmeas(h.cpu().detach().numpy(), gt.cpu().detach().numpy())
            Pbic, Sbic = calcmeas(bich.cpu().detach().numpy(), gt.cpu().detach().numpy())
            PSNR_Avg += PSNR
            SSIM_Avg += SSIM
            Pbic_Avg += Pbic
            Sbic_Avg += Sbic
        for j in range(h.shape[0]):
            L = l[j,:,2,:,:]
            L = trans.ToPILImage()(L.cpu()).convert('RGB')
            L = L.save(resdir_0+str(j+i*batch_size).zfill(8)+".png",quality=95)
            H = h[j,:,:,:]
            H = trans.ToPILImage()(H.cpu()).convert('RGB')
            H = H.save(resdir_2+str(j+i*batch_size).zfill(8)+".png",quality=95)
            H_bic = bich[j,:,:,:]
            H_bic = trans.ToPILImage()(H_bic.cpu()).convert('RGB')
            H_bic = H_bic.save(resdir_1+str(j+i*batch_size).zfill(8)+".png",quality=95)
            GT = gt[j,:,:,:]
            GT = trans.ToPILImage()(GT.cpu()).convert('RGB')
            GT = GT.save(resdir_3+str(j+i*batch_size).zfill(8)+".png",quality=95)
    PSNR_Avg /= len(testloader)
    SSIM_Avg /= len(testloader)
    Pbic_Avg /= len(testloader)
    Sbic_Avg /= len(testloader)
    
    print("Avg PSNR: %.2f" % PSNR_Avg)
    f.write("Avg PSNR: %.2f\n" % PSNR_Avg)
    print("Avg SSIM: %.4f" % SSIM_Avg)
    f.write("Avg SSIM: %.3f\n" % SSIM_Avg)
    print("Avg PSNR_bicubic: %.2f" % Pbic_Avg)
    f.write("Avg PSNR_bicubic: %.2f\n" % Pbic_Avg)
    print("Avg SSIM_bicubic: %.4f" % Sbic_Avg)
    f.write("Avg SSIM_bicubic: %.3f\n" % Sbic_Avg)
    f.close()
