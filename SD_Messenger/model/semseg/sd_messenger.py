# -*- coding: utf-8 -*-
# @Time    : 2023/9/28 09:14
# @Author  : Qixiang ZHANG
# @File    : mask_decoder.py
# @Software: PyCharm

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from einops import rearrange, repeat
import torch.nn.functional as F
import math
from mmcv.cnn import ConvModule


class CrossAttn(nn.Module):
    def __init__(self, num_heads, embedding_channels, attention_dropout_rate, num_class, patch_num):
        super().__init__()
        self.KV_size = embedding_channels * num_heads
        self.num_class = num_class
        self.patch_num = patch_num
        self.num_attention_heads = num_heads
        self.attention_head_size = embedding_channels  # // self.num_attention_heads

        self.query_u_u = nn.Linear(embedding_channels, embedding_channels * self.num_attention_heads, bias=False)
        self.key_u_u = nn.Linear(embedding_channels, embedding_channels * self.num_attention_heads, bias=False)
        self.value_u_u = nn.Linear(embedding_channels, embedding_channels * self.num_attention_heads, bias=False)

        self.query_l_u = nn.Linear(embedding_channels, embedding_channels * self.num_attention_heads, bias=False)
        self.key_l_u = nn.Linear(embedding_channels, embedding_channels * self.num_attention_heads, bias=False)
        self.value_l_u = nn.Linear(embedding_channels, embedding_channels * self.num_attention_heads, bias=False)

        self.psi = nn.InstanceNorm2d(self.num_attention_heads)

        self.softmax = Softmax(dim=3)
        self.out_u = nn.Linear(embedding_channels * self.num_attention_heads, embedding_channels, bias=False)
        self.out_l = nn.Linear(embedding_channels * self.num_attention_heads, embedding_channels, bias=False)
        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)
        self.pseudo_label = None
        self.embedding_channels = embedding_channels

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, emb):
        emb_l, emb_u = torch.split(emb, emb.size(0) // 2, dim=0)
        _, N, C = emb_u.size()

        Q_u_u = self.query_u_u(emb_u)
        K_u_u = self.key_u_u(emb_u)
        V_u_u = self.value_u_u(emb_u)

        multi_head_Q = self.transpose_for_scores(Q_u_u).transpose(-1, -2)
        multi_head_K = self.transpose_for_scores(K_u_u)
        multi_head_V = self.transpose_for_scores(V_u_u).transpose(-1, -2)

        attn_u_u = torch.matmul(multi_head_Q, multi_head_K)

        similarity_matrix = self.attn_dropout(self.softmax(self.psi(attn_u_u)))
        context_layer = torch.matmul(similarity_matrix, multi_head_V)

        context_layer = context_layer.permute(0, 3, 2, 1).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.KV_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        O_u_u = self.out_u(context_layer)
        O_u_u = self.proj_dropout(O_u_u)

        # ==========================================================

        Q_l_u = self.query_l_u(emb_l)
        K_l_u = self.key_l_u(emb_u)
        V_l_u = self.value_l_u(emb_u)

        batch_size = Q_l_u.size(0)

        K_l_u = rearrange(K_l_u, 'b n c -> n (b c)')
        V_l_u = rearrange(V_l_u, 'b n c -> n (b c)')

        K_l_u = repeat(K_l_u, 'n bc -> r n bc', r=batch_size)
        V_l_u = repeat(V_l_u, 'n bc -> r n bc', r=batch_size)

        multi_head_Q = Q_l_u.unsqueeze(1).transpose(-1, -2)
        multi_head_K = K_l_u.unsqueeze(1)
        multi_head_V = V_l_u.unsqueeze(1).transpose(-1, -2)

        attn_l_u = torch.matmul(multi_head_Q, multi_head_K)

        similarity_matrix = self.attn_dropout(self.softmax(self.psi(attn_l_u)))
        context_layer = torch.matmul(similarity_matrix, multi_head_V)

        context_layer = context_layer.permute(0, 3, 2, 1).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.KV_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        O_l_u = self.out_l(context_layer)
        O_l_u = self.proj_dropout(O_l_u)

        out = torch.cat([O_l_u, O_u_u], dim=0)
        return out

class SDMessenger(nn.Module):
    def __init__(self, num_layers, num_heads, embedding_channels, channel_num, channel_num_out, patchSize, scale,
                 dropout_rate, attention_dropout_rate, num_class, patch_num, add_cross_attn=[False, False, False, False]):
        super().__init__()
        self.map_in = nn.Sequential(nn.Conv2d(channel_num, embedding_channels, kernel_size=1, padding=0), nn.GELU())

        self.attn_norm = LayerNorm(embedding_channels, eps=1e-6)
        self.attn = CrossAttn(num_heads, embedding_channels, attention_dropout_rate, num_class, patch_num)

        # self.ffn_norm1 = LayerNorm(embedding_channels, eps=1e-6)
        # self.ffn1 = Mlp(embedding_channels, embedding_channels * 4, dropout_rate)

        self.encoder_norm = LayerNorm(embedding_channels, eps=1e-6)

        self.map_out = nn.Sequential(nn.Conv2d(embedding_channels, channel_num_out, kernel_size=1, padding=0),
                                     nn.GELU())

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.normal_(m.bias, std=1e-6)

    def forward(self, en):
        if not self.training:
            en = torch.cat((en, en))

        _, _, h, w = en.shape
        en = self.map_in(en)
        en = en.flatten(2).transpose(-1, -2)  # (B, n_patches, hidden)

        emb = self.attn_norm(en)
        emb = self.attn(emb)
        emb = emb + en

        out = self.encoder_norm(emb)

        B, n_patch, hidden = out.size()
        out = out.permute(0, 2, 1).contiguous().view(B, hidden, h, w)

        out = self.map_out(out)

        if not self.training:
            out = torch.split(out, out.size(0) // 2, dim=0)[0]

        return out


class SDMessengerTransformer(nn.Module):
    """
    SegFormer V16: cross attn with memory bank storing unlabeled data
    """

    def __init__(self, num_layers, num_heads, num_class, in_planes, image_size, add_cross_attn=[True, True, True, True]):
        super(SDMessengerTransformer, self).__init__()

        embedding_dim = 768

        self.sd_messenger = SDMessenger(num_layers=num_layers, num_heads=num_heads,
                               embedding_channels=in_planes[3],
                               channel_num=in_planes[3],
                               channel_num_out=in_planes[3],
                               patchSize=1, scale=1,
                               dropout_rate=0.5, attention_dropout_rate=0.1,
                               patch_num=(image_size // 32 + 1) ** 2, num_class=num_class)
        
        self.add_cross_attn = add_cross_attn

        self.decoder = SegFormerHead(embedding_dim, in_planes, num_class)

    def forward(self, feats, h, w):
        e1, e2, e3, e4 = feats
        print(f'e1 shape: {e1.shape}, e2 shape: {e2.shape}, e3 shape: {e3.shape}, e4 shape: {e4.shape}')

        # Transformer encoder with U2L delivery
        if self.add_cross_attn[0]:
            e1 = self.sd_messenger(e1)
        if self.add_cross_attn[1]:
            e2 = self.sd_messenger(e2)
        if self.add_cross_attn[2]:
            e3 = self.sd_messenger(e3)
        if self.add_cross_attn[3]:
            e4 = self.sd_messenger(e4)

        # segmentation decoder
        out = self.decoder(e1, e2, e3, e4)
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)

        return out, e4


class MLP(nn.Module):
    """
    Linear Embedding
    """

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class SegFormerHead(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, embedding_dim, in_channels, num_class):
        super(SegFormerHead, self).__init__()

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * 4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.dropout = nn.Dropout2d(0.1)

        self.linear_pred = nn.Conv2d(embedding_dim, num_class, kernel_size=1)

    def forward(self, c1, c2, c3, c4):
        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).contiguous().reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).contiguous().reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).contiguous().reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).contiguous().reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
