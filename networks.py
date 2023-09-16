import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from utils import resize2d
from sgformer import *
from involution_pytorch import Inv2d

class adaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=False):
        super(adaILN, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.num_features = num_features
    
        if self.using_bn:
            self.rho = Parameter(torch.Tensor(1, num_features, 3))
            self.rho[:,:,0].data.fill_(3)
            self.rho[:,:,1].data.fill_(1)
            self.rho[:,:,2].data.fill_(1)
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1,1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1,1))
            self.running_mean.zero_()
            self.running_var.zero_()
        else:
            self.rho = Parameter(torch.Tensor(1, num_features, 2))
            self.rho[:,:,0].data.fill_(3.2)
            self.rho[:,:,1].data.fill_(1)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        softmax = nn.Softmax(2)
        rho = softmax(self.rho)
        
        
        if self.using_bn:
            if self.training:
                bn_mean, bn_var = torch.mean(input, dim=[0, 2, 3], keepdim=True), torch.var(input, dim=[0, 2, 3], keepdim=True)
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * bn_mean.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * bn_var.data)
                else:
                    self.running_mean.add_(bn_mean.data)
                    self.running_var.add_(bn_mean.data ** 2 + bn_var.data)
            else:
                bn_mean = torch.autograd.Variable(self.running_mean)
                bn_var = torch.autograd.Variable(self.running_var)
            out_bn = (input - bn_mean) / torch.sqrt(bn_var + self.eps)
            rho_0 = rho[:,:,0]
            rho_1 = rho[:,:,1]
            rho_2 = rho[:,:,2]

            rho_0 = rho_0.view(1, self.num_features, 1,1)
            rho_1 = rho_1.view(1, self.num_features, 1,1)
            rho_2 = rho_2.view(1, self.num_features, 1,1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            rho_2 = rho_2.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln + rho_2 * out_bn
        else:
            rho_0 = rho[:,:,0]
            rho_1 = rho[:,:,1]
            rho_0 = rho_0.view(1, self.num_features, 1,1)
            rho_1 = rho_1.view(1, self.num_features, 1,1)
            rho_0 = rho_0.expand(input.shape[0], -1, -1, -1)
            rho_1 = rho_1.expand(input.shape[0], -1, -1, -1)
            out = rho_0 * out_in + rho_1 * out_ln

        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)
        return out

class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=100, img_size=256, light=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        # Bottleneck
        self.inv1 = Inv2d(channels=256, kernel_size=3, stride=1) # channel+k+3 => channel=256, k=32, 291 => channel=128, k=64, 195
        self.inv2 = Inv2d(channels=256, kernel_size=3, stride=1) 
        self.inv3 = Inv2d(channels=256, kernel_size=3, stride=1) 
        self.inv4 = Inv2d(channels=256, kernel_size=3, stride=1) 
        self.inv5 = Inv2d(channels=256, kernel_size=3, stride=1) 
        #self.inv6 = Inv2d(channels=256, kernel_size=3, stride=1) 
        #self.inv7 = Inv2d(channels=256, kernel_size=3, stride=1) 

        self.SGfomer1 = Block(C, mask= False, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=8, linear=False)
        self.SGfomer2 = Block(C, mask= True, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=8, linear=False)
        self.SGfomer3 = Block(C, mask= False, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=2, linear=False)
        self.SGfomer4 = Block(C, mask= True, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=2, linear=False)
        self.SGfomer5 = Block(C, mask= False, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False)
        self.SGfomer6 = Block(C, mask= True, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False)
        #self.SGfomer7 = Block(C, mask= False, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False)
        #self.SGfomer8 = Block(C, mask= True, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False)  # sr_ratio=4 deleted
 
        # Up-Sampling
        self.dec0 = [nn.ReflectionPad2d(1),   
                         nn.Conv2d(ngf*4, ngf*4), kernel_size=3, stride=1, padding=0, bias=False),
                         adaILN(ndf*2),
                         nn.GELU]
        self.dec3 = [nn.ReflectionPad2d(1),   
                         nn.Conv2d(ngf*8, ngf*8), kernel_size=3, stride=1, padding=0, bias=False),
                         nn.PixelShuffle(2),
                         adaILN(ndf*2),
                         nn.GELU]
        self.dec2 = [nn.ReflectionPad2d(1),   
                         nn.Conv2d(ngf*4, ngf*4), kernel_size=3, stride=1, padding=0, bias=False),
                         nn.PixelShuffle(2),
                         adaILN(ndf),
                         nn.GELU]
        self.dec1 = [nn.ReflectionPad2d(1),   
                         nn.Conv2d(ngf*2, ngf), kernel_size=3, stride=1, padding=0, bias=False),
                         adaILN(ndf),
                         nn.GELU,
                         nn.ReflectionPad2d(3),
                         nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                         nn.Tanh()]

    def forward(self, input, z, x1, x2, x3):
        #x = z
        _, _, H, W= z.shape
        # calcuator SVD in input
        S = torch.linalg.svdvals(input)
        c1 = torch.matmul(torch.reshape(S[0,0:H,0], (H,1)), torch.reshape(S[0,0:H,1], (1,H)))
        c2 = torch.matmul(torch.reshape(S[0,0:H,1], (H,1)), torch.reshape(S[0,0:H,2], (1,H)))
        c3 = torch.matmul(torch.reshape(S[0,0:H,0], (H,1)), torch.reshape(S[0,0:H,2], (1,H)))
        c4 = torch.matmul(torch.reshape(c1, (H, H, 1)), torch.reshape(S[0,0:H,2], (1, 1, H)))
        
        svd = torch.cat((torch.reshape(c1, (1,1,H,H)),torch.reshape(c2, (1,1,H,H)),torch.reshape(c3, (1,1,H,H))), dim=1)
        svd = torch.cat((svd,torch.reshape(c4, (1,H,H,H))), dim=1)

        # concatenate SVD and Encoder
        x = torch.cat((z,svd), dim=1) # (1, k+3, k, k) + (1 ,256-k-3, k, k) = (1, 256, k, k)
        _, C, H, W= x.shape

        # Bottleneck
        mask = None
        x, mask = self.SGfomer1(x, H, W, mask)
        x = self.inv1(x)
        x, mask = self.SGfomer2(x, H, W, mask)
        x = self.inv2(x)
        x, mask = self.SGfomer3(x, H, W, mask)
        x = self.inv3(x)
        x, mask = self.SGfomer4(x, H, W, mask)
        x = self.inv4(x)
        x, mask = self.SGfomer5(x, H, W, mask)
        x = self.inv5(x)
        x, mask = self.SGfomer6(x, H, W, mask)
        #x = self.inv6(x)
        #x, mask = self.SGfomer7(x, H, W, mask)
        #x = self.inv7(x)
        #x, mask = self.SGfomer8(x, H, W, mask)

        # Up-Sampling
        x = self.dec0(x)
        x = torch.cat((x,x3), dim=1)   # skip-connection layer3
        x = self.dec3(x)
        x = torch.cat((x,x2), dim=1)   # skip-connection layer2
        x = self.dec2(x)
        x = torch.cat((x,x1), dim=1)   # skip-connection layer1
        x = self.dec1(x)
        out = x

        return out

class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64):
        super(Discriminator, self).__init__() 

        # proposed Encoder
        enc1 = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                 nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=0, bias=True)),
                 nn.GELU()]
        enc1 += [nn.utils.spectral_norm(nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=0, bias=True)),nn.GELU(),
                 torch.fft.fft2(),
                 nn.utils.spectral_norm(nn.Conv2d(ndf, ndf, kernel_size=3, stride=1, padding=0, bias=True)),nn.GELU(),
                 torch.fft.ifft2(),  nn.Conv2d(ndf, ndf, kernel_size=1, stride=1, bias=True)]
        
        enc2 = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                 nn.Conv2d(ndf, ndf*2, kernel_size=3, stride=2, padding=0, bias=True)),
                 nn.GELU()]
        enc2 += [nn.utils.spectral_norm(nn.Conv2d(ndf*2, ndf*2, kernel_size=3, stride=1, padding=0, bias=True)),nn.GELU(),
                 torch.fft.fft2(),
                 nn.utils.spectral_norm(nn.Conv2d(ndf*2, ndf*2, kernel_size=3, stride=1, padding=0, bias=True)),nn.GELU(),
                 torch.fft.ifft2(),  nn.Conv2d(ndf*2, ndf*2, kernel_size=1, stride=1, bias=True)]
        enc3 = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                 nn.Conv2d(ndf*2, ndf*4, kernel_size=3, stride=2, padding=0, bias=True)),
                 nn.GELU()]
        enc3 += [nn.utils.spectral_norm(nn.Conv2d(ndf*4, ndf*4, kernel_size=3, stride=1, padding=0, bias=True)),nn.GELU(),
                 torch.fft.fft2(),
                 nn.utils.spectral_norm(nn.Conv2d(ndf*4, ndf*4, kernel_size=3, stride=1, padding=0, bias=True)),nn.GELU(),
                 torch.fft.ifft2(),  nn.Conv2d(ndf*4, ndf*4, kernel_size=1, stride=1, bias=True)]
        #enc4 = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                 nn.Conv2d(ndf*2, ndf*4, kernel_size=3, stride=2, padding=0, bias=True)),
                 nn.GELU()]
        #enc4 += [nn.utils.spectral_norm(nn.Conv2d(ndf*4, ndf*4, kernel_size=3, stride=1, padding=0, bias=True)),nn.GELU(),
                 torch.fft.fft2(),
                 nn.utils.spectral_norm(nn.Conv2d(ndf*4, ndf*4, kernel_size=3, stride=1, padding=0, bias=True)),nn.GELU(),
                 torch.fft.ifft2(),  nn.Conv2d(ndf*4, ndf*4, kernel_size=1, stride=1, bias=True)]
        self.GELU = nn.nn.GELU()

        #Discriminator
        Dis1 = [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.nn.GELU(),
                      nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.nn.GELU(),
                      nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.nn.GELU()]
        Dis2 = [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf*4, ndf*4, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True),
                      nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf*8, ndf*8, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True),
                      nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf*16, ndf*16, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]
        
        self.conv1 = nn.utils.spectral_norm(   #1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 + 3*2^3= 70
            nn.Conv2d(ndf*4, 1, kernel_size=4, stride=1, padding=0, bias=False))
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(ndf*16, 1, kernel_size=4, stride=1, padding=0, bias=False))

        self.convz = nn.utils.spectral_norm(
            nn.Conv2d(ndf*4, (ndf*4-(ndf+3)), kernel_size=4, stride=1, padding=0, bias=False))
        

        self.pad = nn.ReflectionPad2d(1)

        self.Dis1 = nn.Sequential(*Dis1)
        self.Dis2 = nn.Sequential(*Dis2)
        
        self.enc1 = nn.Sequential(*enc1)
        self.enc2 = nn.Sequential(*enc2)
        self.enc3 = nn.Sequential(*enc3)
        #self.enc4 = nn.Sequential(*enc4)
        
    def forward(self, input):
      
        x1 = self.enc1(input)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        #x4 = self.enc4(x3)

        z = self.convz(x3)
        z = x =  self.GELU(z)    # dimention 256-(k+3)

        x1 = self.Dis1(x1)
        x3 = self.Dis2(x3)
        x1 = self.pad(x1)
        x3 = self.pad(x3)
        out1 = self.conv1(x1)
        out2 = self.conv2(x3)

        
        # number ibput and output cheked.
        #x1, x2, x3 for loss CT.
        
        return x1, x2, x3, out1, out2, z

