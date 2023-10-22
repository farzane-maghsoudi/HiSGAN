import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from utils import resize2d
from sgformer import *
from involution_pytorch import Inv2d
from adaILN import *
from FFCBlock import *

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
        C = ngf * 4 #256

        # Gamma, Beta block
        if self.light:
            FC = [nn.Linear(C, C, bias=False),
                  nn.GELU(),
                  nn.Linear(C, C, bias=False),
                  nn.GELU()]
        else:
            FC = [nn.Linear(img_size // 4 * img_size // C * 4, C, bias=False),
                  nn.GELU(),
                  nn.Linear(C, C, bias=False),
                  nn.GELU()]  #nn.ReLU(True)
        self.gamma = nn.Linear(C, C, bias=False)
        self.beta = nn.Linear(C, C, bias=False)

        # Bottleneck
        self.inv1 = Inv2d(channels=C, kernel_size=3, stride=1) # channel+k+3 => channel=256, k=32, 291 => channel=128, k=64, 195
        self.inv2 = Inv2d(channels=C, kernel_size=3, stride=1) 
        self.inv3 = Inv2d(channels=C, kernel_size=3, stride=1) 
        self.inv4 = Inv2d(channels=C, kernel_size=3, stride=1) 
        self.inv5 = Inv2d(channels=C, kernel_size=3, stride=1) 
        #self.inv6 = Inv2d(channels=C, kernel_size=3, stride=1) 
        #self.inv7 = Inv2d(channels=C, kernel_size=3, stride=1) 

        self.SGfomer1 = Block(C, mask= False, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU,sr_ratio=8, linear=False)
        self.SGfomer2 = Block(C, mask= True, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, sr_ratio=8, linear=False)
        self.SGfomer3 = Block(C, mask= False, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, sr_ratio=2, linear=False)
        self.SGfomer4 = Block(C, mask= True, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU,  sr_ratio=2, linear=False)
        self.SGfomer5 = Block(C, mask= False, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, sr_ratio=1, linear=False)
        self.SGfomer6 = Block(C, mask= True, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, sr_ratio=1, linear=False)
        #self.SGfomer7 = Block(C, mask= False, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=False, drop=0., attn_drop=0.,
        #         drop_path=0., act_layer=nn.GELU, nsr_ratio=1, linear=False)
        #self.SGfomer8 = Block(C, mask= True, num_heads=8, mlp_ratio=4., qkv_bias=True, qk_scale=False, drop=0., attn_drop=0.,
        #         drop_path=0., act_layer=nn.GELU, sr_ratio=1, linear=False)  # sr_ratio=4 deleted
 
        # Up-Sampling
        dec0 = [nn.ReflectionPad2d(1),   
                         nn.Conv2d(ngf*4, ngf*4, kernel_size=3, stride=1, padding=0, bias=False),
                         ILN(ngf*4),
                         nn.GELU()]
        dec3 = [nn.ReflectionPad2d(1),   
                         nn.Conv2d(ngf*8, ngf*8, kernel_size=3, stride=1, padding=0, bias=False),
                         nn.PixelShuffle(2),
                         ILN(ngf*2),
                         nn.GELU()]
        dec2 = [nn.ReflectionPad2d(1),   
                         nn.Conv2d(ngf*4, ngf*4, kernel_size=3, stride=1, padding=0, bias=False),
                         nn.PixelShuffle(2),
                         ILN(ngf),
                         nn.GELU()]
        dec1 = [nn.ReflectionPad2d(1),   
                         nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=0, bias=False),
                         ILN(ngf),
                         nn.GELU(),
                         nn.ReflectionPad2d(3),
                         nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                         nn.Tanh()]


        self.FC = nn.Sequential(*FC)
        self.dec0 = nn.Sequential(*dec0)
        self.dec1 = nn.Sequential(*dec1)
        self.dec2 = nn.Sequential(*dec2)
        self.dec3 = nn.Sequential(*dec3)

    def forward(self, input, z, x1, x2, x3):
        #x = z
        _, _, H, W= z.shape
        # calcuator SVD in input
        S = torch.linalg.svdvals(input)
        c1 = torch.matmul(torch.reshape(S[0,0,0:H], (H,1)), torch.reshape(S[0,1,0:H], (1,H)))
        c2 = torch.matmul(torch.reshape(S[0,1,0:H], (H,1)), torch.reshape(S[0,2,0:H], (1,H)))
        c3 = torch.matmul(torch.reshape(S[0,0,0:H], (H,1)), torch.reshape(S[0,2,0:H], (1,H)))
        c4 = torch.matmul(torch.reshape(c1, (H, H, 1)), torch.reshape(S[0,2,0:H], (1, 1, H)))
        
        svd = torch.cat((torch.reshape(c1, (1,1,H,H)),torch.reshape(c2, (1,1,H,H)),torch.reshape(c3, (1,1,H,H))), dim=1)
        svd = torch.cat((svd,torch.reshape(c4, (1,H,H,H))), dim=1)

        # concatenate SVD and Encoder
        x = torch.cat((z,svd), dim=1) # (1, k+3, k, k) + (1 ,256-k-3, k, k) = (1, 256, k, k)
        _, C, H, W= x.shape


        # computation gamma, beta
        if self.light:
            xx_ = torch.nn.functional.adaptive_avg_pool2d(x, 1)
            xx_ = self.FC(xx_.view(xx_.shape[0], -1))
        else:
            xx_ = self.FC(x.view(x.shape[0], -1))
        gamma, beta = self.gamma(xx_), self.beta(xx_)
        
        # Bottleneck
        mask = None
        x, mask = self.SGfomer1(x, H, W, mask, gamma, beta)
        x = self.inv1(x)
        x, mask = self.SGfomer2(x, H, W, mask, gamma, beta)
        x = self.inv2(x)
        x, mask = self.SGfomer3(x, H, W, mask, gamma, beta)
        x = self.inv3(x)
        x, mask = self.SGfomer4(x, H, W, mask, gamma, beta)
        x = self.inv4(x)
        x, mask = self.SGfomer5(x, H, W, mask, gamma, beta)
        x = self.inv5(x)
        x, mask = self.SGfomer6(x, H, W, mask, gamma, beta)
        #x = self.inv6(x)
        #x, mask = self.SGfomer7(x, H, W, mask, gamma, beta)
        #x = self.inv7(x)
        #x, mask = self.SGfomer8(x, H, W, mask, gamma, beta)

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
        self.FFC1 = FFC(in_channels=ndf, out_channels=ndf)

        enc2 = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                 nn.Conv2d(ndf, ndf*2, kernel_size=3, stride=2, padding=0, bias=True)),
                 nn.GELU()]
        self.FFC2 = FFC(in_channels=ndf*2, out_channels=ndf*2)

        enc3 = [nn.ReflectionPad2d(1),
                 nn.utils.spectral_norm(
                 nn.Conv2d(ndf*2, ndf*4, kernel_size=3, stride=2, padding=0, bias=True)),
                 nn.GELU()]
        self.FFC3 = FFC(in_channels=ndf*4, out_channels=ndf*4)

        #enc4 = [nn.ReflectionPad2d(1),
        #         nn.utils.spectral_norm(
        #         nn.Conv2d(ndf*2, ndf*4, kernel_size=3, stride=2, padding=0, bias=True)),
        #         nn.GELU()]
        #enc4 += [nn.utils.spectral_norm(nn.Conv2d(ndf*4, ndf*4, kernel_size=3, stride=1, padding=0, bias=True)),nn.GELU(),
        #         torch.fft.fft2(),
        #         nn.utils.spectral_norm(nn.Conv2d(ndf*4, ndf*4, kernel_size=3, stride=1, padding=0, bias=True)),nn.GELU(),
        #         torch.fft.ifft2(),  nn.Conv2d(ndf*4, ndf*4, kernel_size=1, stride=1, bias=True)]
        self.GELU = nn.GELU()

        #Discriminator
        Dis1 = [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.GELU(),
                      nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.GELU(),
                      nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.GELU()]
        Dis2 = [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf*4, ndf*4, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.GELU(),
                      nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.GELU(),
                      nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf*8, ndf*16, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.GELU()]
        
        self.conv1 = nn.utils.spectral_norm(   #1+3*2^0 + 3*2^1 + 3*2^2 +3*2^3 + 3*2^3= 70
            nn.Conv2d(ndf*4, 1, kernel_size=3, stride=1, padding=0, bias=False))
        self.conv2 = nn.utils.spectral_norm(
            nn.Conv2d(ndf*16, 1, kernel_size=3, stride=1, padding=0, bias=False))

        self.convz = nn.utils.spectral_norm(
            nn.Conv2d(ndf*4, (ndf*4-(ndf+3)), kernel_size=3, stride=1, padding=1, bias=False))
        

        self.pad = nn.ReflectionPad2d(1)

        self.Dis1 = nn.Sequential(*Dis1)
        self.Dis2 = nn.Sequential(*Dis2)
        
        self.enc1 = nn.Sequential(*enc1)
        self.enc2 = nn.Sequential(*enc2)
        self.enc3 = nn.Sequential(*enc3)
        
    def forward(self, input):
      
        x1 = self.enc1(input)
        x1 = self.FFC1(x1)
        
        x2 = self.enc2(x1)
        x2 = self.FFC2(x2)

        x3 = self.enc3(x2)
        x3 = self.FFC3(x3)

        z = self.convz(x3)
        z = x =  self.GELU(z)    # dimention 256-(k+3): 189, 64, 64

        x11 = self.Dis1(x1)
        x33 = self.Dis2(x3)
        x11 = self.pad(x11)
        x33 = self.pad(x33)
        out1 = self.conv1(x11)
        out2 = self.conv2(x33)
        
        # number ibput and output cheked.
        #x1, x2, x3 for loss CT.
        
        return x1, x2, x3, out1, out2, z
