import torch
import torch.nn as nn
import torch.nn.modules as nnm

import torch.nn.functional as F

import typing

""" typical 2D convolution, WxHxC => WxHxC """
def conv(in_channels, out_channels, filter, pad, dil, n_type):
    if (n_type == 'elu'):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, filter, stride=1, padding=(pad*dil), dilation=dil, bias=False),
            nn.ELU(inplace=False)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, filter, stride=1, padding=(pad*dil), dilation=dil, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False)
        )


class OmlDual(nn.Module):
    def __init__(self, 
        num_markers, 
        num_joints
    ):
        super(OmlDual, self).__init__()
        """
        Args:
            width: input width
            height: input height
            ndf: constant number from channels
            dil: dilation value - parameter for convolutional layers
            norma_type: normalization type (elu | batch norm)
        """
        self.h = 136
        self.w = 136
        self.dil = 1
        self.type = "batch_norm"
        self.markers_out = num_markers
        self.joints_out = num_joints
        self.out = self.markers_out + self.joints_out


        # ATTENTION: this is hardcoded due to the SoA model
        ndf = 64
        """ dmc_neural_network """
        self.conv1 = conv(1, ndf, 3, 0, dil=self.dil, n_type=self.type) 
        self.conv2 = conv(ndf, ndf, 3, 0, dil=self.dil, n_type=self.type) 
        self.pool3 = nn.MaxPool2d(2, 2, 0, self.dil, False, False)
        self.conv4 = conv(ndf, 2 * ndf, 3, 0, dil=self.dil, n_type=self.type) 
        self.conv5 = conv(2 * ndf, 2 * ndf, 3, 0, dil=self.dil, n_type=self.type)
        self.conv6 = conv(2 * ndf, 2 * ndf, 3, 0, dil=self.dil, n_type=self.type) 
        self.pool7 = nn.MaxPool2d(2, 2, 0, self.dil, False, False)
       

        f1d = int((((self.w-2 * 2 * self.dil) / 2 - 3 * 2 * self.dil) \
             / 2 * ((self.w-2 * 2 * self.dil) / 2 - 3 * 2 * self.dil) / 2 ) * ndf * 2)

        #f1d = 4608
        self.fc_1 = nn.Linear(f1d, 2048)        
        self.fc_2 = nn.Linear(2048, 3 * self.out)

    def forward(self, 
        data: torch.Tensor
    ) -> typing.Tuple[torch.Tensor, torch.Tensor]:
        out = self.conv1(data)
        out = self.conv2(out)
        out = self.pool3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out_viz = self.pool7(out)

        out = torch.reshape(out_viz, (out_viz.size()[0], out_viz.size()[1] * out_viz.size()[2] * out_viz.size()[3]))

        out = self.fc_1(out)
        out = nn.functional.relu(out)
        out_c = self.fc_2(out)
        out = out_c.view(-1, self.out, 3)
        return out[:, :self.markers_out], out[:, self.markers_out:, ...]
