# model/swin_unet.py
import torch
import torch.nn as nn

# Basit Swin Transformer Block (Minimal örnek)
class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, input_resolution=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

# Patch Embedding
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # [B, C, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, HW, C]
        x = self.norm(x)
        return x, (H, W)

# Basit Swin UNet (tek bloklu minimal versiyon)
class SwinUnet(nn.Module):
    def __init__(self, img_size=224, in_chans=6, num_classes=1, embed_dim=96):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim)
        self.swin_block = SwinTransformerBlock(embed_dim, num_heads=4)
        self.decoder_conv = nn.ConvTranspose2d(embed_dim, num_classes, kernel_size=4, stride=4)

    def forward(self, x):
        x, (H, W) = self.patch_embed(x)       # [B, L, C]
        x = self.swin_block(x)                 # [B, L, C]
        x = x.transpose(1, 2).view(-1, x.size(2), H, W)  # [B, C, H, W]
        x = self.decoder_conv(x)               # [B, num_classes, H*4, W*4]
        return x
