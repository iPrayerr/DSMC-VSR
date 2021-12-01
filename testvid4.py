# x4 test - vid4
import os
import warnings
warnings.filterwarnings('ignore')

from lib.SRnet import *
from lib.data_utils import calcmeas, ValidationsetLoader
from glob import *
from torch.autograd import Variable
from torch.utils.data import ConcatDataset
from tqdm import tqdm

if __name__ == '__main__':
    frame_path = "/data1/datasets/Vid4/Vid4/BIx4"
    gt_path = "/data1/datasets/Vid4/Vid4/GT"
    weights_path = "./weights_vimeo/params_25.pkl"
    batch_size = 1
    
    testset = ""
    testdir = sorted(glob(gt_path+"/*"))
    #testdir = sorted([gt_path+"/01", gt_path+"/02", gt_path+"/03", gt_path+"/04"])
    for tes in testdir:
        tset = ValidationsetLoader(val_dir_hr=tes, val_dir_lr=frame_path,
                                    upscale_factor=4, random_cro=False)
        if testdir.index(tes)==0:
            testset = tset
        else:
            testset = ConcatDataset([testset,tset])
    
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=0)

    # load net
    net = SRNet(srsiz=4, block_config=(2,6,6,3))
    net.load_state_dict(torch.load(weights_path))
    net = net.to(device)

    resdir = "./results_vid4/"
    if not os.path.exists(resdir):
        os.makedirs(resdir)
    f = open(resdir+"metrics.log",'w+')
    resdir_0 = "./results_vimeo/LR/"
    if not os.path.exists(resdir_0):
        os.makedirs(resdir_0)
    resdir_1 = "./results_vimeo/bicubic/"
    if not os.path.exists(resdir_1):
        os.makedirs(resdir_1)
    resdir_2 = "./results_vimeo/ours/"
    if not os.path.exists(resdir_2):
        os.makedirs(resdir_2)
    resdir_3 = "./results_vimeo/GT/"
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
            one = torch.ones_like(bich)
            zero = torch.zeros_like(bich)
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
        
    PSNR_Avg /= len(testloader)
    SSIM_Avg /= len(testloader)
    Pbic_Avg /= len(testloader)
    Sbic_Avg /= len(testloader)
    print("Avg PSNR: %.6f" % PSNR_Avg)
    f.write("Avg PSNR: %.2f\n" % PSNR_Avg)
    print("Avg SSIM: %.6f" % SSIM_Avg)
    f.write("Avg SSIM: %.3f\n" % SSIM_Avg)
    print("Avg PSNR_bicubic: %.6f" % Pbic_Avg)
    f.write("Avg PSNR_bicubic: %.2f\n" % Pbic_Avg)
    print("Avg SSIM_bicubic: %.6f" % Sbic_Avg)
    f.write("Avg SSIM_bicubic: %.3f\n" % Sbic_Avg)
    f.close()