import torch
import torch.nn as nn
class Shift_channel_mix(nn.Module):
    def __init__(self,shift_size):
        super(Shift_channel_mix, self).__init__()
        self.shift_size = shift_size

    def forward(self, x):

        x1, x2, x3, x4 = x.chunk(4, dim = 1)

        x1 = torch.roll(x1, self.shift_size, dims=2)#[:,:,1:,:]

        x2 = torch.roll(x2, -self.shift_size, dims=2)#[:,:,:-1,:]

        x3 = torch.roll(x3, self.shift_size, dims=3)#[:,:,:,1:]

        x4 = torch.roll(x4, -self.shift_size, dims=3)#[:,:,:,:-1]
         
        x = torch.cat([x1, x2, x3, x4], 1)

        return x

class CSUB(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1):
        super(CSUB,self).__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels
        self.up_dwc = nn.Sequential(
            nn.Upsample(scale_factor=2),
            Conv(self.in_channels, self.in_channels, kernel_size, g=self.in_channels, s=stride)
        )
        self.pwc = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.shift_channel_mix = Shift_channel_mix(1)

    def forward(self, x):
        x = self.up_dwc(x)
        x = self.channel_shuffle(x, self.in_channels)
        x = self.pwc(x)
        return x
    
    def channel_shuffle(self, x, groups):
        batchsize, num_channels, height, width = x.data.size()
        channels_per_group = num_channels // groups
        x = x.view(batchsize, groups, channels_per_group, height, width)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)
        x = self.shift_channel_mix(x)
        return x
