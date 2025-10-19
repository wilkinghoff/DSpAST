# Copyright (c) Kevin Wilkinghoff, Aalborg University.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Spatial-AST: https://github.com/zszheng147/Spatial-AST
# Audio-MAE: https://github.com/facebookresearch/AudioMAE
# --------------------------------------------------------

from functools import partial

import math

import torch
import torch.nn as nn

import torchaudio

from timm.models.layers import to_2tuple, trunc_normal_

from utils.aml_losses import AdaCos
from utils.stft import STFT, LogmelFilterBank
from utils.vision_transformer import VisionTransformer as _VisionTransformer


def gcc_phat(self, sig, refsig):
    ncorr = 2*self.nfft - 1
    nfft = int(2**np.ceil(np.log2(np.abs(ncorr))))
    Px = librosa.stft(y=sig,
                    n_fft=nfft,
                    hop_length=self.hopsize,
                    center=True,
                    window=self.window, 
                    pad_mode='reflect')
    Px_ref = librosa.stft(y=refsig,
                        n_fft=nfft,
                        hop_length=self.hopsize,
                        center=True,
                        window=self.window,
                        pad_mode='reflect')

    R = Px*np.conj(Px_ref)

    n_frames = R.shape[1]
    gcc_phat = []
    for i in range(n_frames):
        spec = R[:, i].flatten()
        cc = np.fft.irfft(np.exp(1.j*np.angle(spec)))
        cc = np.concatenate((cc[-mel_bins//2:], cc[:mel_bins//2]))
        gcc_phat.append(cc)
    gcc_phat = np.array(gcc_phat)
    gcc_phat = gcc_phat[None,:,:]

    return gcc_phat

class GCC_extractor(nn.Module):
    def __init__(self, mel_bins=128):
        super().__init__()
        self.mel_bins = mel_bins

    def forward(self, x):
        Gxy = x[:,0] * torch.conj(x[:,1])
        cc = torch.fft.irfft(torch.exp(1.j*torch.angle(Gxy)))
        max_shift = int(self.mel_bins//2)
        cc = torch.cat((cc[:,:,-max_shift:], cc[:,:,:max_shift]), dim=-1)
        cc = torch.unsqueeze(cc, dim=1)
        return cc


# https://github.com/martinoywa/channel-attention
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=1):
        super(ChannelAttention, self).__init__()
        self.bn = nn.BatchNorm1d(in_channels, affine=False)
        self.bn2 = nn.BatchNorm2d(in_channels, affine=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=False),
            nn.Linear(in_channels // reduction_ratio, in_channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.bn(y)
        y = self.fc(y).view(b, c, 1, 1)
        y = self.sigmoid(y)
        return x * y

def conv3x3(in_channels, out_channels, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_channels, out_channels, stride=1):
    "1x1 convolution with padding"
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)

class PatchEmbed_new(nn.Module):
    """ Flexible Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, stride=10):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride) # with overlapped patches
        _, _, h, w = self.get_output_shape(img_size) # n, emb_dim, h, w
        self.patch_hw = (h, w)
        self.num_patches = h*w

    def get_output_shape(self, img_size):
        return self.proj(torch.randn(1, self.in_chans, img_size[0], img_size[1])).shape 

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x) # 32, 1, 1024, 128 -> 32, 768, 101, 12
        x = x.flatten(2) # 32, 768, 101, 12 -> 32, 768, 1212
        x = x.transpose(1, 2) # 32, 768, 1212 -> 32, 1212, 768
        return x

class DSpAST(_VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, num_cls_tokens=3, **kwargs):
        super().__init__(**kwargs)
        img_size = (1024, 128) # 1024, 128
        in_chans = 1
        emb_dim = 768

        del self.cls_token
        self.num_cls_tokens = num_cls_tokens
        self.cls_tokens1 = nn.Parameter(torch.zeros(1, num_cls_tokens, emb_dim))
        torch.nn.init.normal_(self.cls_tokens1, std=.02)
        self.cls_tokens2 = nn.Parameter(torch.zeros(1, num_cls_tokens, emb_dim))
        torch.nn.init.normal_(self.cls_tokens2, std=.02)
        self.cls_tokens3 = nn.Parameter(torch.zeros(1, num_cls_tokens, emb_dim))
        torch.nn.init.normal_(self.cls_tokens3, std=.02)

        self.patch_embed = PatchEmbed_new(
            img_size=img_size, patch_size=(16,16), 
            in_chans=in_chans, embed_dim=emb_dim, stride=16
        ) # no overlap. stride=img_size=16
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, emb_dim), requires_grad=False)  # fixed sin-cos embedding

        self.spectrogram_extractor = STFT(
            n_fft=1024, hop_length=320, win_length=1024, window='hann', 
            center=True, pad_mode='reflect', freeze_parameters=True
        )

        self.gcc = GCC_extractor()
        
        import librosa
        self.melW = librosa.filters.mel(
            sr=32000, n_fft=1024, n_mels=128, fmin=50, fmax=14000
        )
        self.logmel_extractor = LogmelFilterBank(
            sr=32000, n_fft=1024, n_mels=128, fmin=50, 
            fmax=14000, ref=1.0, amin=1e-10, top_db=None, freeze_parameters=True
        )

        num_feats = 6
        self.conv_downsample1 = nn.Sequential(
            ChannelAttention(num_feats),
            conv3x3(num_feats, 1),
            nn.BatchNorm2d(1),
            nn.GELU(),
        )
        self.conv_downsample2 = nn.Sequential(
            ChannelAttention(num_feats),
            conv3x3(num_feats, 1), 
            nn.BatchNorm2d(1),
            nn.GELU(),
        )
        self.conv_downsample3 = nn.Sequential(
            ChannelAttention(num_feats),
            conv3x3(num_feats, 1),
            nn.BatchNorm2d(1),
            nn.GELU(),
        )

        self.timem = torchaudio.transforms.TimeMasking(192)
        self.freqm = torchaudio.transforms.FrequencyMasking(48)

        self.bn = nn.BatchNorm2d(2, affine=False)
        del self.norm  # remove the original norm

        self.bn_ild = nn.BatchNorm2d(1, affine=False)

        self.target_frame = 1024

        self.fc_norm = kwargs['norm_layer'](int(emb_dim/3))

        self.pre_head1 = nn.Linear(emb_dim, int(emb_dim/3))
        self.pre_head2 = nn.Linear(emb_dim, int(emb_dim/3))
        self.pre_head3 = nn.Linear(emb_dim, int(emb_dim/3))

        self.head = nn.Linear(int(emb_dim/3), 355)
        # weight initialization
        trunc_normal_(self.head.weight, std=2e-5)

        trainable_heads = True
        self.distance_head = AdaCos(emb_dim=int(emb_dim/3), num_classes=21, trainable=trainable_heads)
        self.azimuth_head = AdaCos(emb_dim=int(emb_dim/3), num_classes=360, trainable=trainable_heads)
        self.elevation_head = AdaCos(emb_dim=int(emb_dim/3), num_classes=180, trainable=trainable_heads)


    def random_masking_2d(self, x, mask_t_prob, mask_f_prob):
        N, L, D = x.shape  # batch, length, dim
        T, F = 64, 8
        
        # mask T
        x = x.reshape(N, T, F, D)
        len_keep_T = int(T * (1 - mask_t_prob))
        noise = torch.rand(N, T, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_T]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, F, D)
        x = torch.gather(x, dim=1, index=index) # N, len_keep_T(T'), F, D

        # mask F
        x = x.permute(0, 2, 1, 3) # N T' F D => N F T' D
        len_keep_F = int(F * (1 - mask_f_prob))
        noise = torch.rand(N, F, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_keep = ids_shuffle[:, :len_keep_F]
        index = ids_keep.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, len_keep_T, D)
        x_masked = torch.gather(x, dim=1, index=index)
        x_masked = x_masked.permute(0,2,1,3) # N F' T' D => N T' F' D 
        x_masked = x_masked.reshape(N,len_keep_F*len_keep_T,D)
            
        return x_masked, None, None

    def forward_features_mask(self, x, mask_t_prob, mask_f_prob):
        B = x.shape[0] #bsz, 512, 768 (unmasked)
        x1 = x[:,:,:768]
        x2 = x[:,:,768:1536]
        x3 = x[:,:,1536:]

        x1 = x1 + self.pos_embed[:, 1:, :]
        x2 = x2 + self.pos_embed[:, 1:, :]
        x3 = x3 + self.pos_embed[:, 1:, :]

        if mask_t_prob > 0.0 or mask_f_prob > 0.0:
            x1, mask, ids_restore = self.random_masking_2d(x1, mask_t_prob, mask_f_prob)
            x2, mask, ids_restore = self.random_masking_2d(x2, mask_t_prob, mask_f_prob)
            x3, mask, ids_restore = self.random_masking_2d(x3, mask_t_prob, mask_f_prob)

        cls_tokens1 = self.cls_tokens1
        cls_tokens1 = cls_tokens1.expand(B, -1, -1)
        cls_tokens2 = self.cls_tokens2
        cls_tokens2 = cls_tokens2.expand(B, -1, -1)
        cls_tokens3 = self.cls_tokens3
        cls_tokens3 = cls_tokens3.expand(B, -1, -1)
        x1 = torch.cat([cls_tokens1, x1], dim=1)   # bsz, 512 + 2 + 10, 768 
        x2 = torch.cat([cls_tokens2, x2], dim=1)   # bsz, 512 + 2 + 10, 768 
        x3 = torch.cat([cls_tokens3, x3], dim=1)   # bsz, 512 + 2 + 10, 768

        x1 = self.pos_drop(x1)
        x2 = self.pos_drop(x2)
        x3 = self.pos_drop(x3)

        for blk in self.blocks:
            x1 = blk(x1)
            x2 = blk(x2)
            x3 = blk(x3)
        x = torch.cat([self.pre_head1(x1), self.pre_head2(x2), self.pre_head3(x3)], dim=2)

        return x

    # overwrite original timm
    def forward(self, waveforms, reverbs, mask_t_prob=0.0, mask_f_prob=0.0):
        waveforms = torchaudio.functional.fftconvolve(waveforms, reverbs, mode='full')[..., :waveforms.shape[-1]]
        B, C, T = waveforms.shape

        waveforms = waveforms.reshape(B * C, T)
        real, imag = self.spectrogram_extractor(waveforms)

        complex = torch.complex(real, imag).reshape(B, C, -1, 513)
        GCC = self.gcc(complex)

        log_mel = self.logmel_extractor(torch.sqrt(real**2 + imag**2)).reshape(B, C, -1, 128)
        ILD = torch.exp(log_mel[:,0:1])/(torch.exp(log_mel[:,1:2])+1e-16)

        log_mel = self.bn(log_mel)
        ILD = self.bn_ild(ILD)
        IPD = torch.atan2(imag[1::2], real[1::2]) - torch.atan2(imag[::2], real[::2])
        x = torch.cat([log_mel, GCC, ILD, torch.matmul(torch.cat([torch.cos(IPD), torch.sin(IPD)], dim=1), self.logmel_extractor.melW)], dim=1)

        if x.shape[2] < self.target_frame:
            x = nn.functional.interpolate(x, (self.target_frame, x.shape[3]), mode="bicubic", align_corners=True)
        
        x1 = self.conv_downsample1(x)
        x2 = self.conv_downsample2(x)
        x3 = self.conv_downsample3(x)

        if self.training:
            x1 = x1.transpose(-2, -1) # bsz, 4, 1024, 128 --> bsz, 4, 128, 1024
            x2 = x2.transpose(-2, -1) # bsz, 4, 1024, 128 --> bsz, 4, 128, 1024
            x3 = x3.transpose(-2, -1) # bsz, 4, 1024, 128 --> bsz, 4, 128, 1024
            x1 = self.freqm(x1)
            x2 = self.freqm(x2)
            x3 = self.freqm(x3)
            x1 = self.timem(x1)
            x2 = self.timem(x2)
            x3 = self.timem(x3)
            x1 = x1.transpose(-2, -1)
            x2 = x2.transpose(-2, -1)
            x3 = x3.transpose(-2, -1)

        x1 = self.patch_embed(x1)
        x2 = self.patch_embed(x2)
        x3 = self.patch_embed(x3)
        x = torch.cat([x1, x2, x3], dim=2)
        x = self.forward_features_mask(x, mask_t_prob=mask_t_prob, mask_f_prob=mask_f_prob)

        dis_token = (x[:,0,:256] + x[:,1,:256] + x[:,2,:256]) / 3
        doa_token = (x[:,0,256:512] + x[:,1,256:512] + x[:,2,256:512]) / 3
        cls_tokens = (x[:,0,512:] + x[:,1,512:] + x[:,2,512:]) / 3

        cls_tokens = self.fc_norm(cls_tokens)

        azi_token = doa_token
        ele_token = doa_token

        classifier = self.head(cls_tokens)
        distance = self.distance_head(dis_token)
        azimuth = self.azimuth_head(azi_token)
        elevation = self.elevation_head(ele_token)

        return classifier, distance, azimuth, elevation


def build_AST(**kwargs):
    model = DSpAST(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model