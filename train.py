# x4 SR
import warnings
warnings.filterwarnings('ignore')

import datetime
import argparse
from tqdm import tqdm
from lib.SRnet import *
from lib.data_utils import *
from lib.loss import SRLoss

from torch.autograd import Variable
import torch.optim as optim

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25, help="number of epochs")
    parser.add_argument("--patch_size", type=int, default=64, help="size of each image patch")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--lr", type=float, default=0.0002, help="learning rate")
    parser.add_argument("--parallel", type=bool, default=False, help="whether to apply data parallel")
    parser.add_argument("--trainlr", type=str, default="/data/REDS/train/train_sharp_bicubic",
                        help="path of training set LR")
    parser.add_argument("--trainhr", type=str, default="/data/REDS/train/train_sharp",
                        help="path of training set HR")
    parser.add_argument("--vallr", type=str, default="/data/REDS/val/val_sharp_bicubic",
                        help="path of validation set LR")
    parser.add_argument("--valhr", type=str, default="/data/REDS/val/val_sharp",
                        help="path of validation set HR")
    parser.add_argument("--pretrained_weights", type=str, default=None,
                        help="pretrained weights")
    opt = parser.parse_args()
    print(opt)

    hrdir = glob(opt.trainhr+"/*")
    trainset = ""
    for hr_dir in hrdir:
        tset = TrainsetLoader(trainset_dir_hr=hr_dir, trainset_dir_lr=opt.trainlr,
                                    upscale_factor=4,patch_size=opt.patch_size)
        if hrdir.index(hr_dir)==0:
            trainset = tset
        else:
            trainset = ConcatDataset([trainset,tset])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                              shuffle=True, num_workers=2)

    valdir = glob(opt.valhr+"/*")
    valset = ""
    for val in valdir:
        vset = ValidationsetLoader(val_dir_hr=val, val_dir_lr=opt.vallr,
                                    upscale_factor=4,patch_size=opt.patch_size)
        if valdir.index(val)==0:
            valset = vset
        else:
            valset = ConcatDataset([valset,vset])
    validloader = torch.utils.data.DataLoader(valset, batch_size=8,
                                              shuffle=True, num_workers=2)
    
    net = SRNet(block_config=(2, 6, 6, 3))
    if opt.pretrained_weights is not None:
        net.load_state_dict(torch.load(opt.pretrained_weights))
    
    if opt.parallel == True:
        net = nn.DataParallel(net)
    net = net.to(device)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=opt.lr)
    criterion = SRLoss().to(device)

    # Train
    logdir = "./logs_v24/"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    wdir = "./weights_v24/"
    if not os.path.exists(wdir):
        os.makedirs(wdir)
    f = open(logdir+datetime.date.today().strftime("%y%m%d")+".log",'w+')
    for epoch in range(opt.epochs):
        running_loss = 0.0
        PSNR_Avg = 0.0
        SSIM_Avg = 0.0
        for i,data in enumerate(tqdm(trainloader,desc="Epoch %d/%d"%(epoch+1, opt.epochs), dynamic_ncols=True),0):
            optimizer.zero_grad()
            l,gt = data
            l,gt = Variable(l.to(device)), Variable(gt.to(device))
            h,duall = net(l)
            loss = criterion(h.to(device), gt, duall.to(device), l[:,:,2,:,:])
            loss.backward()
            optimizer.step()
            running_loss += loss.data

            one = torch.ones_like(h)
            zero = torch.zeros_like(h)
            h = torch.where(h > 1.0, one, h)
            h = torch.where(h < 0.0, zero, h)
            PSNR, SSIM = calcmeas(h.cpu().detach().numpy(),gt.cpu().detach().numpy())
            PSNR_Avg += PSNR
            SSIM_Avg += SSIM
        print('Epoch %d/%d, running loss: %.6f, PSNR %.2f, SSIM %.3f\n' % (epoch + 1, opt.epochs,
                                                                           running_loss/len(trainloader),
                                                                           PSNR_Avg/len(trainloader),
                                                                           SSIM_Avg/len(trainloader)))
        if epoch%1 == 0:
            f.write('Epoch %d/%d, running loss: %.6f\n, PSNR %.2f, SSIM %.3f\n' % (epoch + 1, opt.epochs,
                                                                                   running_loss/len(trainloader),
                                                                                    PSNR_Avg/len(trainloader),
                                                                                   SSIM_Avg/len(trainloader)))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print("\nValidating...")
            PSNR_Avg = 0.0
            Pbic_Avg = 0.0
            SSIM_Avg = 0.0
            Sbic_Avg = 0.0
            for j, valdata in enumerate(tqdm(validloader,desc="Validation", dynamic_ncols=True), 0):
                vl, vgt = valdata
                vl, vgt = Variable(vl,requires_grad=False), Variable(vgt,requires_grad=False)
                vl = vl.to(device)
                with torch.no_grad():
                    vh, _ = net(vl)
                    bich = F.interpolate(vl[:,:,2,:,:],scale_factor=4,mode='bicubic')
                    one = torch.ones_like(vh)
                    zero = torch.zeros_like(vh)
                    vh = torch.where(vh > 1.0, one, vh)
                    vh = torch.where(vh < 0.0, zero, vh)
                    bich = torch.where(bich > 1.0, one, bich)
                    bich = torch.where(bich < 0.0, zero, bich)
                PSNR, SSIM = calcmeas(vh.cpu().detach().numpy(),vgt.cpu().detach().numpy())
                Pbic, Sbic = calcmeas(bich.cpu().detach().numpy(),vgt.cpu().detach().numpy())
                PSNR_Avg += PSNR
                Pbic_Avg += Pbic
                SSIM_Avg += SSIM
                Sbic_Avg += Sbic
            PSNR_Avg /= len(validloader)
            SSIM_Avg /= len(validloader)
            Pbic_Avg /= len(validloader)
            Sbic_Avg /= len(validloader)
            print("Validated.")
            f.write("Validation:\n")
            print("Avg PSNR: %.2f" % PSNR_Avg)
            f.write("Avg PSNR: %.2f\n" % PSNR_Avg)
            print("Avg SSIM: %.3f" % SSIM_Avg)
            f.write("Avg SSIM: %.3f\n" % SSIM_Avg)
            print("Avg PSNR_bicubic: %.2f" % Pbic_Avg)
            f.write("Avg PSNR_bicubic: %.2f\n" % Pbic_Avg)
            print("Avg SSIM_bicubic: %.3f" % Sbic_Avg)
            f.write("Avg SSIM_bicubic: %.3f\n" % Sbic_Avg)
            if opt.parallel == True:
                torch.save(net.module.state_dict(),wdir+"params_%d.pkl" % (epoch+1))
            else:
                torch.save(net.state_dict(),wdir+"params_%d.pkl" % (epoch+1)) 
            print("\nSaved weights: "+wdir+"params_%d.pkl\n"% (epoch+1))
            f.write("Saved weights: "+wdir+"params_%d.pkl\n\n"% (epoch+1))

    f.close()