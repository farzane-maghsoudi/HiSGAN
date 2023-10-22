import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
#from sgformer import adaILN
from adaILN import *

from timm.models.layers import DropPath, trunc_normal_
import math

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act(x + self.dwconv(x, H, W))
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

def local_conv(dim):
    return nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)

class Attention(nn.Module):
    def __init__(self, dim, mask, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.sr_ratio=sr_ratio
        if sr_ratio>1:
            if mask:
                self.q = nn.Linear(dim, dim, bias=qkv_bias)
                self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
                self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)

                if self.sr_ratio==8:
                    #r1, r2, r3 = 32,64,4  # for h = 64
                    r1, r2, r3 = 16, 8, 4
                elif self.sr_ratio==4:
                    r1, r2, r3 = 8, 4, 2
                elif self.sr_ratio==2:
                    r1, r2, r3 = 2, 1, None
                self.f1 = nn.Linear(r1, 1)
                self.f2 = nn.Linear(r2, 1)
                if r3 is not None:
                    self.f3 = nn.Linear(r3, 1)
                    
            else:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = adaILN(dim)
                self.act = nn.GELU()

                self.q1 = nn.Linear(dim, dim//2, bias=qkv_bias)
                self.kv1 = nn.Linear(dim, dim, bias=qkv_bias)
                self.q2 = nn.Linear(dim, dim // 2, bias=qkv_bias)
                self.kv2 = nn.Linear(dim, dim, bias=qkv_bias)
                
        else:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.lepe_linear = nn.Linear(dim, dim)
        self.lepe_conv = local_conv(dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, mask, gamma, beta):
        x = x.flatten(2).transpose(1, 2)
        B, L, C = x.shape
        B_, N= H, W
        lepe = self.lepe_conv(self.lepe_linear(x).transpose(1, 2).view(B, C, H, W)).view(H, C, -1).transpose(-1, -2)  #(1,H*W,C)
        x = x.view(B, H, W, C).reshape(H, W, C)

        if self.sr_ratio > 1:
            if mask is None:
                # global
                q1 = self.q1(x).reshape(B_, N, self.num_heads//2, C // self.num_heads).permute(0, 2, 1, 3)
                x_ = x.permute(2, 0, 1).reshape(B, C, H, W)
                x_1 = self.sr(x_).reshape(C, B_//self.sr_ratio,N//self.sr_ratio).permute(1,2,0)
                x_1 = self.act(self.norm(x_1, gamma, beta))
                kv1 = self.kv1(x_1).reshape(B_//self.sr_ratio, -1, 2, self.num_heads//2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                k1, v1 = kv1[0].repeat(self.sr_ratio,1,self.sr_ratio,1), kv1[1].repeat(self.sr_ratio,1,self.sr_ratio,1) #B_ head N C

                attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale #B head Nq Nkv
                attn1 = attn1.softmax(dim=-1)
                attn1 = self.attn_drop(attn1)
                x1 = (attn1 @ v1).transpose(1, 2).reshape(B_, N, C//2)

                global_mask_value = torch.mean(attn1.detach().mean(1), dim=1) # B Nk  #max ?  mean ?

                # local
                q2 = self.q2(x).reshape(B_, N, self.num_heads // 2, C // self.num_heads).permute(0, 2, 1, 3) #B head N C
                kv2 = self.kv2(x_.reshape(B_, C, -1).permute(0, 2, 1)).reshape(B_, -1, 2, self.num_heads // 2,
                                                                          C // self.num_heads).permute(2, 0, 3, 1, 4)
                k2, v2 = kv2[0], kv2[1]
                q_window = 8
                window_size= 8
                q2, k2, v2 = window_partition(q2, q_window, H, W), window_partition(k2, window_size, H, W), \
                             window_partition(v2, window_size, H, W)
                attn2 = (q2 @ k2.transpose(-2, -1)) * self.scale
                # (B*numheads*num_windows, window_size*window_size, window_size*window_size)
                attn2 = attn2.softmax(dim=-1)
                attn2 = self.attn_drop(attn2)

                x2 = (attn2 @ v2)  # B*numheads*num_windows, window_size*window_size, C   .transpose(1, 2).reshape(B, N, C)
                x2 = window_reverse(x2, q_window, H, W, self.num_heads // 2).transpose(1, 2).reshape(B_, N, C//2)

                local_mask_value = torch.mean(attn2.detach().view(B, self.num_heads//2, H*W//window_size, window_size, window_size).mean(1), dim=2)
                local_mask_value = local_mask_value.view(B, H // window_size, W // window_size, window_size, window_size)
                local_mask_value=local_mask_value.permute(0, 1, 3, 2, 4).contiguous().view(B, H, W)

                # mask B H W
                x = torch.cat([x1, x2], dim=-1)
                x = self.proj(x+lepe)
                x = self.proj_drop(x)
                # cal mask
                mask = local_mask_value+global_mask_value
                mask_1 = mask.view(1, H * W)
                mask_2 = mask.permute(0, 2, 1).reshape(1, H * W)
                mask = [mask_1, mask_2]
            else:
                q = self.q(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

                # mask [local_mask global_mask]  local_mask [value index]  value [B, H, W]
                # use mask to fuse
                mask_1, mask_2 = mask
                mask_sort1, mask_sort_index1 = torch.sort(mask_1, dim=1)
                mask_sort2, mask_sort_index2 = torch.sort(mask_2, dim=1)
                if self.sr_ratio == 8:
                    r1, r2, r3 = 16, 8, 4
                    token1, token2, token3 = H * W // (r1*4), H * W // (r2*2), H * W //(r3*4)
                elif self.sr_ratio==4:
                    r1, r2, r3 = 8, 4, 2
                    token1, token2, token3 = H * W // (r1*4), H * W // (r2*2), H * W //(r3*4)
                elif self.sr_ratio == 2:
                    token1, token2 = H * W // (r1*2), H * W // (r2*1)

                if self.sr_ratio==4 or self.sr_ratio==8:
                    x_ = x.permute( 1, 0, 2).reshape(1, H * W, C)
                    x = x.reshape(1, H * W, C)
                    p1 = torch.gather(x, 1, mask_sort_index1[:, :H * W // 4].unsqueeze(-1).repeat(1, 1, C))  # B, N//4, C
                    p2 = torch.gather(x, 1, mask_sort_index1[:, H * W // 4:H * W // 4 * 3].unsqueeze(-1).repeat(1, 1, C))
                    p3 = torch.gather(x, 1, mask_sort_index1[:, H * W // 4 * 3:].unsqueeze(-1).repeat(1, 1, C))
                    seq1 = torch.cat([self.f1(p1.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1),
                                      self.f3(p3.permute(0, 2, 1).reshape(B, C, token3, -1)).squeeze(-1)], dim=-1).permute(0,2,1)  # B N C
                    p1_ = torch.gather(x_, 1, mask_sort_index2[:, :H * W // 4].unsqueeze(-1).repeat(1, 1, C))  # B, N//4, C
                    p2_ = torch.gather(x_, 1, mask_sort_index2[:, H * W // 4:H * W // 4 * 3].unsqueeze(-1).repeat(1, 1, C))
                    p3_ = torch.gather(x_, 1, mask_sort_index2[:, H * W // 4 * 3:].unsqueeze(-1).repeat(1, 1, C))
                    seq2 = torch.cat([self.f1(p1_.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2_.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1),
                                      self.f3(p3_.permute(0, 2, 1).reshape(B, C, token3, -1)).squeeze(-1)], dim=-1).permute(0,2,1)  # B N C
                elif self.sr_ratio==2:
                    x_ = x.permute( 1, 0, 2).reshape(1, H * W, C)
                    x = x.reshape(1, H * W, C)

                    p1 = torch.gather(x, 1, mask_sort_index1[:, :H * W // 2].unsqueeze(-1).repeat(1, 1, C))  # B, N//2, C
                    p2 = torch.gather(x, 1, mask_sort_index1[:, H * W // 2:].unsqueeze(-1).repeat(1, 1, C))
                    
                    seq1 = torch.cat([self.f1(p1.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1)], dim=-1).permute(0, 2, 1)  # B N C

                    p1_ = torch.gather(x_, 1, mask_sort_index2[:, :H * W // 2].unsqueeze(-1).repeat(1, 1, C))  # B, N//4, C
                    p2_ = torch.gather(x_, 1, mask_sort_index2[:, H * W // 2:].unsqueeze(-1).repeat(1, 1, C))
                    print("p1_",p1_.shape)
                    print("p2_",p2_.shape)

                    seq2 = torch.cat([self.f1(p1_.permute(0, 2, 1).reshape(B, C, token1, -1)).squeeze(-1),
                                      self.f2(p2_.permute(0, 2, 1).reshape(B, C, token2, -1)).squeeze(-1)], dim=-1).permute(0, 2, 1)  # B N C

                kv1 = self.kv1(seq1).reshape(B_, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4) # kv B heads N C
                kv2 = self.kv2(seq2).reshape(B_, -1, 2, self.num_heads // 2, C // self.num_heads).permute(2, 0, 3, 1, 4)
                kv = torch.cat([kv1, kv2], dim=2)
                k, v = kv[0], kv[1]
                attn = (q @ k.transpose(-2, -1)) * self.scale
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)

                x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
                x = self.proj(x+lepe)
                x = self.proj_drop(x)
                mask=None

        else:
            q = self.q(x).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            kv = self.kv(x).reshape(B_, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
            x = self.proj(x+lepe)
            x = self.proj_drop(x)

            mask=None
        x = x.permute(2, 0, 1).reshape(B, H, W, C)
        return x, mask

def window_partition(x, window_size, H, W):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B_, num_heads, N, C = x.shape
    B = 1
    x = x.contiguous().view(B_*num_heads, N, C).contiguous().view(B*num_heads, H, W, C)
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view( -1,window_size, window_size, C)
    return windows  #(B*numheads*num_windows, window_size, window_size, C)


def window_reverse(windows, window_size, H, W, head):
    Bhead = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(Bhead, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(Bhead, H, W, -1).view(Bhead//head, head, H, W, -1)\
        .contiguous().permute(0,2,3,1,4).contiguous().view(Bhead//head, H, W, -1).view(Bhead//head, H*W, -1)
    return x #(B, H, W, C)

class Block(nn.Module):

    def __init__(self, dim, mask, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = adaILN(dim)
        #self.attn = Attention(
        #        dim=C, mask,
         #       num_heads=8, qkv_bias=True, qk_scale=False,
         #       attn_drop=0., proj_drop=0, sr_ratio=4, linear=False)
        self.attn = Attention(
            dim, mask,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = adaILN(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, mask, gamma, beta):
        #x = torch.randn( 1,256,64,64)
        #B, C, H, W = x.shape
        #mask= None
        #outx, mask = attn(x, H, W, mask)
        #print(outx.shape) #[64, 64, 256]
        x_ = self.norm1(x, gamma, beta)
        x_, mask = self.attn(x_, H, W, mask, gamma, beta)
        x = x + self.drop_path(x_)
        x = self.norm2(x, gamma, beta)
        x = self.mlp(x, H, W)
        x = x + self.drop_path(x)

        return x, mask
