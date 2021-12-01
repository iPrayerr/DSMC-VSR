# AAAI 2021

from collections import OrderedDict
from lib.non_local_embedded_gaussian import *
from thop import profile
from torchvision import transforms as trans

try:
    #from lib.dcn.deform_conv import ModulatedDeformConvPack as DCN
    from lib.dcn.deform_conv import deform_conv2d_naive as DCN
except ImportError:
    raise ImportError('Failed to import DCNv2 module.')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for m in net_l:
        # for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)
            m.weight.data *= scale  # for residual block
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            m.weight.data *= scale
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

class ConvDCN2d(nn.Module):
    def __init__(self, inchans, nf, kernel_size=3, stride=1, padding=1, bias=True, groups=8):
        super(ConvDCN2d, self).__init__()
        self.xconv = nn.Sequential(OrderedDict([
            ('mapping', nn.Conv2d(inchans, nf, kernel_size, stride=stride, padding=padding, bias=bias)),
        ]))
        nn.init.kaiming_normal_(self.xconv[0].weight.data)
        if inchans != nf or (inchans==nf and kernel_size==1):
            self.xconv.add_module('conv', nn.Conv2d(nf, nf, 3, 1, 1, bias=False))
            self.xconv.add_module('bn1', nn.BatchNorm2d(nf))
            nn.init.kaiming_normal_(self.xconv[1].weight.data)
        self.offconv = nn.Sequential(OrderedDict([
            ('mapping', nn.Conv2d(inchans, groups * 2 * kernel_size * kernel_size, 3, 1, 1, bias=False)),  # inchans, nf
            ('bn', nn.BatchNorm2d(groups * 2 * kernel_size * kernel_size)),
        ]))
        nn.init.kaiming_normal_(self.offconv[0].weight.data)
        #if inchans != nf or (inchans==nf and kernel_size == 1):
        #    self.offconv.add_module('conv', nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
        #    self.offconv.add_module('bn1', nn.BatchNorm2d(nf))
        #    nn.init.kaiming_normal_(self.offconv[2].weight.data)
        #self.ConvDCN = nn.Sequential(OrderedDict([
        #    ('dcn', DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)),
        #                            #extra_offset_mask=True)),
        #    ('bn', nn.BatchNorm2d(nf)),
        #]))
        self.ConvDCN = DCN(nf, nf, 3, stride=1, padding=1, dilation=1, deformable_groups=groups)
        self.bn = nn.BatchNorm2d(nf)

    def forward(self, x):
        x_fea = F.leaky_relu(self.xconv(x),negative_slope=0.1)
        offset = F.leaky_relu(self.offconv(x),negative_slope=0.1)
        x_fea = F.leaky_relu(self.bn(self.ConvDCN(x_fea,offset)),negative_slope=0.1)
        return x_fea

class ResidualBlock_noBN(nn.Sequential):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64, useDCN=True):
        super(ResidualBlock_noBN, self).__init__()
        self.useDCN = useDCN
        self.add_module('conv1', nn.Conv2d(nf, nf, 1, 1, bias=False))
        self.add_module('bn1', nn.BatchNorm2d(nf))
        self.add_module('relu1', nn.LeakyReLU(negative_slope=0.1, inplace=True))
        if self.useDCN == True:
            self.add_module('DCNconv', ConvDCN2d(nf, nf))
        else:
            self.add_module('conv2', nn.Conv2d(nf, nf, 3, 1, 1, bias=True))
            self.add_module('bn2', nn.BatchNorm2d(nf))

        # initialization
        # initialize_weights(self.modules(), 0.1)

    def forward(self, x):
        identity = x
        out = super(ResidualBlock_noBN, self).forward(x)
        if self.useDCN == True:
            return identity + out
        else:
            return F.leaky_relu(identity + out, negative_slope=0.1)

class Pyramid_Upsampling(nn.Module):
    def __init__(self, inchans):
        self.inchans = inchans
        super(Pyramid_Upsampling, self).__init__()
        self.upsampling = nn.PixelShuffle(2)
        self.reducing = nn.Sequential(OrderedDict([
            ('reduce', nn.Conv2d(inchans//4, 3, 1, 1, bias=True)),
            ('bn', nn.BatchNorm2d(3)),
        ]))
        nn.init.kaiming_normal_(self.reducing[0].weight.data)

        if inchans != 80:
            self.later = nn.Sequential(OrderedDict([
                ('pyramid_rb1', ResidualBlock_noBN(inchans//4,useDCN=False))
            ]))
            nn.init.kaiming_normal_(self.later[0][0].weight.data)
            nn.init.kaiming_normal_(self.later[0][3].weight.data)
            for i in range(2):  # ResNet3
                self.later.add_module('pyramid_rb%d' % (i + 1), ResidualBlock_noBN(inchans//4,useDCN=False))
                nn.init.kaiming_normal_(self.later[i][0].weight.data)
                nn.init.kaiming_normal_(self.later[i][3].weight.data)
            
    def forward(self, x):
        resimap = self.upsampling(x)
        outresi = F.leaky_relu(self.reducing(resimap), negative_slope=0.1)
        if self.inchans != 80:
            resimap = self.later(resimap)
            return outresi, resimap
        else:
            return outresi

class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm3d(num_input_features)),
        self.add_module('relu1', nn.LeakyReLU(negative_slope=0.1, inplace=True)),
        self.add_module('conv1', nn.Conv3d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', nn.BatchNorm3d(bn_size * growth_rate)),
        self.add_module('relu2', nn.LeakyReLU(negative_slope=0.1, inplace=True)),
        # FDB
        self.add_module('conv2-1', nn.Conv3d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)
            nn.init.kaiming_normal_(self[i][2].weight.data)
            nn.init.kaiming_normal_(self[i][5].weight.data)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.add_module('conv', nn.Conv3d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        nn.init.kaiming_normal_(self[2].weight.data)

class SRNet(nn.Module):
    def __init__(self, inchans=3, wnd=5, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, srsiz=4):

        super(SRNet, self).__init__()

        self.srsiz = srsiz
        self.wnd = wnd
        # First convolution - 64x64
        self.features_0 = ConvDCN2d(inchans,num_init_features)
        
        # 2D Residual Block for feature Extraction - front
        self.RB2d1 = nn.Sequential(OrderedDict([
            ('front_rb1', ResidualBlock_noBN(num_init_features))
        ]))
        nn.init.kaiming_normal_(self.RB2d1[0][0].weight.data)
        # nn.init.kaiming_normal_(self.RB2d1[0][3].weight.data)
        for i in range(4):
            self.RB2d1.add_module('front_rb%d' % (i+2), ResidualBlock_noBN(num_init_features))
            nn.init.kaiming_normal_(self.RB2d1[i][0].weight.data)
            # nn.init.kaiming_normal_(self.RB2d1[i][3].weight.data)
        
        self.down0 = nn.Conv2d(num_init_features, num_init_features, 3, 2, 1, bias=False)

        # Each denseblock
        self.features = nn.Sequential(OrderedDict([]))
        self.num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(num_layers=num_layers, num_input_features=self.num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
            self.features.add_module('denseblock%d' % (i + 1), block)
            self.num_features = self.num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=self.num_features, num_output_features=self.num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                self.num_features = self.num_features // 2
        self.features.add_module('3DNonLocal1', NONLocalBlock3D(num_init_features*srsiz, sub_sample=True, bn_layer=True))
        
        # 2D Residual Block for feature Extraction - 32x32
        self.RB2d2 = nn.Sequential(OrderedDict([
            ('later_rb1', ResidualBlock_noBN(num_init_features*wnd))
        ]))
        nn.init.kaiming_normal_(self.RB2d2[0][0].weight.data)
        # nn.init.kaiming_normal_(self.RB2d2[0][3].weight.data)
        for i in range(3):
            self.RB2d2.add_module('later_rb%d' % (i+2), ResidualBlock_noBN(num_init_features*wnd))
            nn.init.kaiming_normal_(self.RB2d2[i+1][0].weight.data)
            # nn.init.kaiming_normal_(self.RB2d2[i][3].weight.data)

        # Upsampling
        self.up1 = Pyramid_Upsampling(320)
        self.convout1 = nn.Conv2d(3,3,1,1,bias=False)
        nn.init.kaiming_normal_(self.convout1.weight.data)
        self.fusion1 = nn.Sequential(OrderedDict([
            ('fusion', nn.Conv2d(83,80,1,1, bias=False)),
            ('bn', nn.BatchNorm2d(80)),
        ]))
        nn.init.kaiming_normal_(self.fusion1[0].weight.data)
        self.up2 = Pyramid_Upsampling(80)

        # Dual Learning
        self.down1 = nn.Sequential(OrderedDict([
            ('blur', nn.Conv2d(3, 3, 3, 1, 1, bias=False)),
            ('relu', nn.LeakyReLU(negative_slope=0.1, inplace=True)),
            ('kernel', nn.Conv2d(3, 3, 3, 2, 1, bias=True)),
        ]))
        nn.init.kaiming_normal_(self.down1[0].weight.data)
        nn.init.kaiming_normal_(self.down1[2].weight.data)
        self.down2 = nn.Sequential(OrderedDict([
            ('blur', nn.Conv2d(3, 3, 3, 1, 1, bias=False)),
            ('relu', nn.LeakyReLU(negative_slope=0.1, inplace=True)),
            ('kernel', nn.Conv2d(3, 3, 3, 2, 1, bias=True)),
        ]))
        nn.init.kaiming_normal_(self.down2[0].weight.data)
        nn.init.kaiming_normal_(self.down2[2].weight.data)

    def forward(self, x):
        B, C, N, H, W = x.size()
        features = self.features_0(x.view(-1,C,H,W))
        features = self.RB2d1(features)
        pooled_f = self.down0(features).view(B,-1,N,H//2,W//2)
        pooled_f = self.features(pooled_f)
        f = F.pixel_shuffle(pooled_f[:,:,0,:,:],2)
        for i in range(1,self.wnd):
            f = torch.cat([f,F.pixel_shuffle(pooled_f[:,:,i,:,:],2)],2)
        del pooled_f
        f = f.view(B, -1, H, W)
        features = features.view(B, -1, H, W)
        features = features + f
        del f
        # ResBlock Before Upsampling
        features = self.RB2d2(features.view(B, -1, H, W))

        # Upsampling
        res1, res2 = self.up1(features)
        x_center = x[:,:,(self.wnd-1)//2,:,:].contiguous()
        x_center = F.interpolate(x_center, scale_factor=2, mode='bicubic')
        x_center += res1    # 128x128
        del res1
        res2 = torch.cat([res2, F.leaky_relu(self.convout1(x_center),negative_slope=0.1)], 1)
        res2 = F.leaky_relu(self.fusion1(res2),negative_slope=0.1)
        res2 = self.up2(res2)
        x_center = F.interpolate(x_center, scale_factor=2, mode='bicubic')
        x_center += res2    # 256x256
        del res2

        # Dual Learning
        dualx = self.down1(x_center)
        dualx = self.down2(dualx)
        return x_center, dualx