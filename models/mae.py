# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.vision_transformer import PatchEmbed, Block

from pos_embed import get_2d_sincos_pos_embed

from math import sqrt
import numpy as np
from typing import Tuple, Union, Callable, Optional
from utils import to_2tuple
from collections import OrderedDict
import torch.nn.functional as F
from timm.models.layers import Mlp, DropPath

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class Attention_fuse(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Cross_MultiAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.attention = nn.MultiheadAttention(embed_dim=output_dim, num_heads=8)
        self.q_linear = nn.Linear(input_dim, output_dim)
        self.k_linear = nn.Linear(input_dim, output_dim)
        self.v_linear = nn.Linear(input_dim, output_dim)
        self.linear_output = nn.Linear(output_dim, output_dim)
        
    def forward(self, rgb_features, other_features):
        # 计算查询（Q），键（K）和值（V）

        rgb_features = rgb_features.permute(1, 0, 2)
        other_features = other_features.permute(1, 0, 2)

        q = self.q_linear(rgb_features)  # [batch_size, seq_len, output_dim]
        k = self.k_linear(other_features)  # [batch_size, seq_len, output_dim]
        v = self.v_linear(other_features)  # [batch_size, seq_len, output_dim]

        # 使用注意力机制进行特征融合
        attn_output, _ = self.attention(q, k, v)

        # 输出转换
        output = self.linear_output(attn_output).permute(1, 0, 2)

        # 移除无关部分
        # output = output[:, 1:, :]

        return output
    
class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.attention(self.ln_1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int,  mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, mlp_ratio, act_layer=act_layer)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                x = checkpoint(r, x, attn_mask)  
            else:
                x = r(x, attn_mask=attn_mask)
        return x
class VoxelTransformer(nn.Module):
    def __init__(
            self, image_size: int, patch_size: int, width: int, layers: int, heads: int, mlp_ratio: float,
            output_dim: int, act_layer: Callable = nn.GELU):
        super().__init__()
        '''令self.width = width'''
        self.width = width
        self.image_size = to_2tuple(image_size)
        self.patch_size = to_2tuple(patch_size)
        self.grid_size = (self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1])
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        '''voxel的处理'''
        self.class_embedding_voxel = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding_voxel = nn.Parameter(scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width)) #torch.Size([197, 768])
        self.ln_pre_voxel = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, mlp_ratio, act_layer=act_layer)

        self.ln_post_voxel = LayerNorm(width)
        self.proj_voxel = nn.Parameter(scale * torch.randn(width, output_dim))
        self.voxel_linear = nn.Linear(800, 768)  # width：768

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def forward(self, x: torch.Tensor):
        batch = x.shape[0]
        x = x[:, 0:9800, :] #torch.Size([88, 9800, 16])
        x = x.reshape(batch, 196, 800) #torch.Size([88, 196, 800])
        x = self.voxel_linear(x) #torch.Size([88, 196, 768])
        x = torch.cat([self.class_embedding_voxel.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  #torch.Size([88, 197, 768])
        x = x + self.positional_embedding_voxel.to(x.dtype)
        x = self.ln_pre_voxel(x)   
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        # x = self.ln_post_voxel(x[:, 0, :])
        # if self.proj_voxel is not None:
        #     x = x @ self.proj_voxel
        return x ##torch.Size([88, 197, 768])
class vision_cfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    timm_model_name: str = None  # a valid model name overrides layers, width, patch_size
    timm_model_pretrained: bool = False  # use (imagenet) pretrained weights for named model
    timm_pool: str = 'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    timm_proj: str = 'linear'  # linear projection for timm model output ('linear', 'mlp', '')


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def forward(self, image_features_1, image_features_2, logit_scale, no_voxel_list = None, output_dict=False):

        device = image_features_1.device

        #对图像特征进行归一化
        image_features_1 = image_features_1 / image_features_1.norm(dim=-1, keepdim=True)
        image_features_2 = image_features_2 / image_features_2.norm(dim=-1, keepdim=True)

        image_features_1_l2 = F.normalize(image_features_1, p=2, dim=1, eps=1e-12, out=None)
        image_features_2_l2 = F.normalize(image_features_2, p=2, dim=1, eps=1e-12, out=None)

        logits_per_image = logit_scale * image_features_1_l2 @ image_features_2_l2.T
        logits_per_text = logit_scale * image_features_2_l2 @ image_features_1_l2.T
        #logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        if no_voxel_list is not None:
            # 获取需要保留的样本索引（no_voxel_list == 1）
            valid_indices = torch.where(no_voxel_list == 1)[0]
            
            # 过滤 logits 和 labels
            logits_per_image = logits_per_image[valid_indices]
            logits_per_text = logits_per_text[valid_indices]
            labels = labels[valid_indices]

            # 如果没有有效样本，返回 0 损失
            if len(valid_indices) == 0:
                return {"contrastive_loss": torch.tensor(0.0, device=device)} if output_dict else torch.tensor(0.0, device=device)

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss

class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., init_values = 1e-5,
                 drop_path=0., num_heads_att = 4, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.patch_embed_rgb = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_rgb = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.pos_embed_rgb = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.rgb_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.event_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        
        self.norm = norm_layer(embed_dim)
        self.norm_rgb = norm_layer(embed_dim)

        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_embed_rgb = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.mask_token_rgb = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_pos_embed_rgb = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        
        self.decoder_blocks_rgb = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

        self.decoder_norm_rgb = norm_layer(decoder_embed_dim)
        self.decoder_pred_rgb = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        patch_num = img_size // patch_size
        self.clip_event = nn.Linear(patch_num**2 + 1 , 1, bias=True) # decoder to patch
        self.clip_rgb = nn.Linear(patch_num**2 + 1, 1, bias=True) # decoder to patch
        self.clip_voxel = nn.Linear(patch_num**2 + 1, 1, bias=True) # decoder to patch
        self.clip_voxel_li = nn.Linear(embed_dim, decoder_embed_dim, bias=True) # decoder to patch
        
        self.drop_path_fuse_re_1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_fuse_re_2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls_fuse_re_1 = LayerScale(embed_dim, init_values=init_values) if init_values else nn.Identity()
        self.ls_fuse_re_2 = LayerScale(embed_dim, init_values=init_values) if init_values else nn.Identity()
        self.attn_fuse_re = Attention_fuse(embed_dim, num_heads=num_heads_att, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp_re = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)
        self.norm_fuse_re_1 = nn.LayerNorm(embed_dim)
        self.norm_fuse_re_2 = nn.LayerNorm(embed_dim)

        #fuse_rgb_event_voxel
        self.drop_path_fuse_rev_1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.drop_path_fuse_rev_2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.ls_fuse_rev_1 = LayerScale(embed_dim, init_values=init_values) if init_values else nn.Identity()
        self.ls_fuse_rev_2 = LayerScale(embed_dim, init_values=init_values) if init_values else nn.Identity()
        self.attn_fuse_rev = Attention_fuse(embed_dim, num_heads=num_heads_att, qkv_bias=True, attn_drop=attn_drop, proj_drop=drop)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp_rev = Mlp(in_features=embed_dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=drop)
        self.norm_fuse_rev_1 = nn.LayerNorm(embed_dim)
        self.norm_fuse_rev_2 = nn.LayerNorm(embed_dim)

        vision_heads = vision_cfg.width // vision_cfg.head_width
        act_layer = nn.GELU
        self.cliploss = ClipLoss(
            local_loss=False,
            gather_with_grad=False,
            cache_labels=True,
            rank=0,
            world_size=1,
            use_horovod=False)
        self.visual_voxel = VoxelTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            output_dim=embed_dim,
            act_layer=act_layer,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.initialize_weights()


    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        pos_embed_rgb = get_2d_sincos_pos_embed(self.pos_embed_rgb.shape[-1], int(self.patch_embed_rgb.num_patches**.5), cls_token=True)
        self.pos_embed_rgb.data.copy_(torch.from_numpy(pos_embed_rgb).float().unsqueeze(0))

        decoder_pos_embed_rgb = get_2d_sincos_pos_embed(self.decoder_pos_embed_rgb.shape[-1], int(self.patch_embed_rgb.num_patches**.5), cls_token=True)
        self.decoder_pos_embed_rgb.data.copy_(torch.from_numpy(decoder_pos_embed_rgb).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        torch.nn.init.normal_(self.cls_token_rgb, std=.02)
        torch.nn.init.normal_(self.mask_token_rgb, std=.02)
        
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        # nn.init.constant_(self.logit_scale_v, np.log(1 / 0.07))

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask_need = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask_need[:, :(len_keep // 2)] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        mask_need = torch.gather(mask_need, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, mask_need, ids_keep
    
    def rgb_masking(self, x, mask_ratio, old_mask):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        old_mask = 1  - old_mask
        noise = noise + old_mask
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_rgb_encoder(self, x, mask_ratio, old_mask):
        # embed patches
        x = self.patch_embed_rgb(x) #torch.Size([64, 196, 768])

        # add pos embed w/o cls token
        x = x + self.pos_embed_rgb[:, 1:, :] #torch.Size([64, 196, 768])
        
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.rgb_masking(x, mask_ratio, old_mask) #torch.Size([64, 49, 768])

        # append cls token
        cls_token = self.cls_token_rgb + self.pos_embed_rgb[:, :1, :] #torch.Size([1, 1, 768])
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) #torch.Size([64, 1, 768])
        x = torch.cat((cls_tokens, x), dim=1) #torch.Size([64, 50, 768])

        # apply Transformer blocks
        for blk in self.rgb_blocks:
            x = blk(x)
        x = self.norm_rgb(x) #torch.Size([64, 50, 768])

        return x, mask, ids_restore

    def forward_event_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x) #torch.Size([64, 196, 768])

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :] #torch.Size([64, 196, 768])

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, mask_need, ids_keep = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :] #torch.Size([1, 1, 768])
        cls_tokens = cls_token.expand(x.shape[0], -1, -1) #torch.Size([64, 1, 768])
        x = torch.cat((cls_tokens, x), dim=1) #torch.Size([64, 197, 768])

        # apply Transformer blocks
        for blk in self.event_blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore, mask_need, ids_keep


    def forward_rgb_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed_rgb(x) #torch.Size([64, 50, 512])

        # append mask tokens to sequence
        mask_tokens = self.mask_token_rgb.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1) #torch.Size([64, 147, 512])
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token torch.Size([64, 196, 512])
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token torch.Size([64, 197, 512])

        # add pos embed
        x = x + self.decoder_pos_embed_rgb #torch.Size([64, 197, 512])

        # apply Transformer blocks
        for blk in self.decoder_blocks_rgb:
            x = blk(x)
        x = self.decoder_norm_rgb(x)
        clip_feature = x.clone()

        # predictor projection
        x = self.decoder_pred_rgb(x) #torch.Size([64, 197, 768])

        # remove cls token
        x = x[:, 1:, :] #torch.Size([64, 196, 768])

        return x, clip_feature
    
    def forward_event_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x) #torch.Size([64, 50, 512])

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1) #torch.Size([64, 147, 512])
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token torch.Size([64, 196, 512])
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token torch.Size([64, 197, 512])

        # add pos embed
        x = x + self.decoder_pos_embed #torch.Size([64, 197, 512])

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        clip_feature = x.clone()

        # predictor projection
        x = self.decoder_pred(x) #torch.Size([64, 197, 768])

        # remove cls token
        x = x[:, 1:, :] #torch.Size([64, 196, 768])

        return x, clip_feature


    def forward_multi_decoder(self, x, ids_restore):

        x = self.decoder_embed(x) #torch.Size([64, 50, 512])

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1) #torch.Size([64, 147, 512])
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token torch.Size([64, 196, 512])
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token torch.Size([64, 197, 512])

        # add pos embed
        x = x + self.decoder_pos_embed #torch.Size([64, 197, 512])

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x) #torch.Size([64, 197, 768])

        # remove cls token
        x = x[:, 1:, :] #torch.Size([64, 196, 768])

        return x

    def feature_fuse_two(self, feature_event, feature_rgb):

        N,T,D = feature_event.shape 
        feature_event = feature_event[:, 1:, :]

        feature_event_keep = feature_event[:, :(T//2), :]
        inputs_fusion= torch.cat((feature_rgb,feature_event_keep),dim=1) 
        inputs_fusion1 = inputs_fusion + self.drop_path_fuse_re_1(self.attn_fuse_re(self.norm_fuse_re_1(inputs_fusion)))
        inputs_fusion1 = self.norm_fuse_re_2(inputs_fusion1)
        inputs_fusion1_out =inputs_fusion1 + self.drop_path_fuse_re_2(self.ls_fuse_re_2(self.mlp_re(inputs_fusion1))) 

        feature_rgb_fuse = inputs_fusion1_out[:,:T,:]

        return feature_rgb_fuse
    
    def feature_fuse_three(self, feature_event, feature_rgb, feature_voxel, ids_keep):

        N,T,D = feature_event.shape
        feature_event = feature_event[:, 1:, :]

        feature_event_keep = feature_event[:, :(T//2), :]
        feature_voxel_cls = feature_voxel[:, :1, :]
        feature_voxel_patch = feature_voxel[:, 1:, :]
        feature_voxel_patch_keep = torch.gather(feature_voxel_patch, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        inputs_fusion= torch.cat((feature_rgb, feature_event_keep, feature_voxel_patch_keep),1) 
        inputs_fusion1 = inputs_fusion + self.drop_path_fuse_rev_1(self.ls_fuse_rev_1(self.attn_fuse_rev(self.norm_fuse_rev_1(inputs_fusion)))) 
        inputs_fusion1 = self.norm_fuse_rev_2(inputs_fusion1)
        inputs_fusion1_out =inputs_fusion1 + self.drop_path_fuse_rev_2(self.ls_fuse_rev_2(self.mlp_rev(inputs_fusion1))) 

        feature_rgb_fuse = inputs_fusion1_out[:,:T,:]

        return feature_rgb_fuse

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, events, voxels, mask_ratio=0.75):#voxels,
        
        event_features, event_mask, event_ids_restore, mask_need, ids_keep = self.forward_event_encoder(events,mask_ratio) 
        rgb_features, rgb_mask, rgb_ids_restore = self.forward_rgb_encoder(imgs, mask_ratio, mask_need) 
        voxels_non_neg = voxels.abs() 
        no_voxel_list = (voxels_non_neg.sum(dim=(1, 2)) != 0).long()
        voxel_features = self.visual_voxel(voxels) 

        pred_rgb, clip_feature_rgb = self.forward_rgb_decoder(rgb_features, rgb_ids_restore) 
        pred_event, clip_feature_event = self.forward_event_decoder(event_features, event_ids_restore) 

        fuse_two_feature = self.feature_fuse_two(event_features, rgb_features)
        fuse_three_feature = self.feature_fuse_three(event_features, rgb_features, voxel_features, ids_keep)

        pred_fuse_two,_ = self.forward_rgb_decoder(fuse_two_feature, event_ids_restore) 
        pred_fuse_three,_ = self.forward_rgb_decoder(fuse_three_feature, event_ids_restore) 

        loss_rgb_mae = self.forward_loss(imgs, pred_rgb, rgb_mask)
        loss_event_mae = self.forward_loss(events, pred_event, event_mask)
        loss_fuse_two_rgb_mae = self.forward_loss(imgs, pred_fuse_two, rgb_mask)
        loss_fuse_three_rgb_mae = self.forward_loss(imgs, pred_fuse_three, rgb_mask)

        clip_event = self.clip_event(clip_feature_event.transpose(1, 2)).squeeze(dim =-1)
        clip_rgb = self.clip_rgb(clip_feature_rgb.transpose(1, 2)).squeeze(dim =-1)
        clip_voxel = self.clip_voxel(voxel_features.transpose(1, 2)).squeeze(dim =-1)
        clip_voxel = self.clip_voxel_li(clip_voxel)

        loss_e_r = self.cliploss(clip_rgb, clip_event, self.logit_scale)
        loss_e_v = self.cliploss(clip_rgb, clip_voxel, self.logit_scale, no_voxel_list)
        
        return loss_rgb_mae, loss_event_mae, loss_fuse_two_rgb_mae, loss_e_r, loss_fuse_three_rgb_mae , loss_e_v


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
