import torch
from torch import nn
from torch.nn import functional as F


class _NonLocalBlockND(nn.Module):
    def __init__(self, in_channels, inter_channels=None, dimension=3, sub_sample=True, bn_layer=True):
        super(_NonLocalBlockND, self).__init__()

        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        elif dimension == 1:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d
        else:
            print("Wrong dimension")
            return

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.g.weight.data)
        nn.init.constant_(self.g.bias.data, 0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight.data, 0)
            nn.init.kaiming_normal_(self.W[0].weight.data)
            nn.init.constant_(self.W[1].bias.data, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            # nn.init.constant_(self.W.weight, 0)
            nn.init.kaiming_normal_(self.W.weight.data)
            nn.init.constant_(self.W.bias.data, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.theta.weight.data)
        nn.init.constant_(self.theta.bias.data, 0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        nn.init.kaiming_normal_(self.phi.weight.data)
        nn.init.constant_(self.phi.bias.data, 0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x.size(0)
        # print(batch_size)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock3D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NONLocalBlock3D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=3, sub_sample=sub_sample,
                                              bn_layer=bn_layer)


if __name__ == '__main__':
    import torch
    import cv2

    img1 = cv2.imread("./06902.jpg")
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    inp1 = torch.from_numpy(img1).float().permute(2, 0, 1)
    img2 = cv2.imread("./06906.jpg")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    inp2 = torch.from_numpy(img2).float().permute(2, 0, 1)
    img3 = cv2.imread("./06909.jpg")
    img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
    inp3 = torch.from_numpy(img3).float().permute(2, 0, 1)

    inp = torch.stack((inp1,inp2,inp3),0).unsqueeze(0)
    # print(inp.size())
    net = NONLocalBlock3D(3, sub_sample=True, bn_layer=True)
    out = net(inp)
    # print(out.size())
    res0 = out[0][1]
    res0 = res0.squeeze(0).squeeze(0).permute(1, 2, 0)
    # print(res0.size())
    res = res0.detach().numpy()
    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    cv2.imwrite("./RES.jpg", res)

