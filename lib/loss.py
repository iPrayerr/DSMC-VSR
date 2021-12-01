import torch
from torch import nn
import torchvision.models as models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SRLoss(nn.Module):
    def __init__(self):
        super(SRLoss, self).__init__()
        
        vgg = models.vgg16(pretrained=True)
        loss_network = nn.Sequential(*list(vgg.children())[:-1]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network.to(device)
        self.cb_loss = CharbonnierLoss().to(device)

    def forward(self, out_images, target_images, lr, duallr):
        # Perception Loss
        perception_loss = self.cb_loss(self.loss_network(out_images), self.loss_network(target_images))
        # Image Loss
        image_loss = self.cb_loss(out_images, target_images)
        # Dual Loss
        dual_loss = self.cb_loss(duallr, lr)
        dual_perc = self.cb_loss(self.loss_network(duallr), self.loss_network(lr))

        return image_loss + 0.1 * dual_loss + 0.1 * perception_loss + 0.01 * dual_perc

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sum(torch.sqrt(diff * diff + self.eps))
        return loss
