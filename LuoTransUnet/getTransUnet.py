import torch
import torch.nn as nn
import functools
import torch.nn.functional as F
from LuoTransUnet.vit_seg_modeling import VisionTransformer as ViT_seg
from LuoTransUnet.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

def get_transNet(n_classes, bMask = False, hidden_size = 768, num_layers = 12, decoder_channels = (256, 128, 64, 16), patch_size = 16, kernel_size: int = 1, kernel_out_channels: int = 1):
    img_size = 256
    vit_patches_size = 16
    vit_name = 'R50-ViT-B_16'

    #refactored for optuna eric
    config_vit = CONFIGS_ViT_seg[vit_name]
    config_vit.n_classes = n_classes
    config_vit.hidden_size = hidden_size
    config_vit.transformer.num_layers = num_layers
    config_vit.decoder_channels = decoder_channels
    config_vit.patch_size = patch_size
    
    
    config_vit.n_skip = 3
    if vit_name.find('R50') != -1:
        config_vit.patches.grid = (int(img_size / vit_patches_size), int(img_size / vit_patches_size))
    net = ViT_seg(config_vit, img_size=img_size, num_classes=n_classes, bMask=bMask, kernel_size=kernel_size, kernel_out_channels=kernel_out_channels)
    return net          


if __name__ == '__main__':
    net = get_transNet(2)
    img = torch.randn((2, 3, 256, 256))
    segments = net(img)
    print(segments.size())
    # for edge in edges:
    #     print(edge.size())
