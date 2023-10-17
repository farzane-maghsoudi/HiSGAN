import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        res = x * y.expand_as(x)
        return res

class SeparableFourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, kernel_size=3):
        # bn_layer not used
        super(SeparableFourierUnit, self).__init__()
        self.groups = groups
        row_out_channels = out_channels // 2
        col_out_channels = out_channels - row_out_channels
        self.row_conv = torch.nn.Conv2d(in_channels=in_channels * 2,
                                        out_channels=row_out_channels * 2,
                                        kernel_size=(kernel_size, 1),
                                        # kernel size is always like this, but the data will be transposed
                                        stride=1, padding=(kernel_size // 2, 0),
                                        padding_mode='reflect',
                                        groups=self.groups, bias=False)
        self.col_conv = torch.nn.Conv2d(in_channels=in_channels * 2,
                                        out_channels=col_out_channels * 2,
                                        kernel_size=(kernel_size, 1),
                                        # kernel size is always like this, but the data will be transposed
                                        stride=1, padding=(kernel_size // 2, 0),
                                        padding_mode='reflect',
                                        groups=self.groups, bias=False)
        self.row_bn = torch.nn.BatchNorm2d(row_out_channels * 2)
        self.col_bn = torch.nn.BatchNorm2d(col_out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

    def process_branch(self, x, conv, bn):
        batch = x.shape[0]

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        ffted = torch.fft.rfft(x, norm="ortho")
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        ffted = self.relu(bn(conv(ffted)))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        output = torch.fft.irfft(ffted, s=x.shape[-1:], norm="ortho")
        return output

    def forward(self, x):
        rowwise = self.process_branch(x, self.row_conv, self.row_bn)
        colwise = self.process_branch(x.permute(0, 1, 3, 2), self.col_conv, self.col_bn).permute(0, 1, 3, 2)
        out = torch.cat((rowwise, colwise), dim=1)
        return out

class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode,
                              align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels=256, out_channels=256, stride=1, groups=1, enable_lfu=True, separable_fu=False, **fu_kwargs):
        # bn_layer not used
        super(FFC, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        conv1 = [nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels//2,
                                                  kernel_size=1, groups=groups, bias=False)),nn.GELU()]

        fu_class = SeparableFourierUnit if separable_fu else FourierUnit
        self.fu = fu_class(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = fu_class(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)
        self.conv1 = nn.Sequential(*conv1)

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)
        print(output.shape)

        return output
