from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from svimg import save_image_unique
from svimg import tensor_to_image

from foveation import FoveatedConv2d
from foveation import FastFoveatedConv2d
from foveation import UltraFastFoveatedConv2d


class AddCoords(nn.Module):
    def __init__(self, with_r=False, learnable_y_amp=True, learnable_x_amp=True):
        super().__init__()
        self.with_r = with_r
        
        if learnable_y_amp:
            # Initialize with your current value, let network optimize it
            self.y_amplifier = nn.Parameter(torch.tensor(3.0))
            self.x_amplifier = nn.Parameter(torch.tensor(1.0))
        else:
            self.y_amplifier = 1.0
            self.x_amplifier = 1.0
    
    def forward(self, x):
        b, _, h, w = x.size()
        
        xx_channel = torch.linspace(-1, 1, w, device=x.device).repeat(h, 1).unsqueeze(0).unsqueeze(0)
        yy_channel = torch.linspace(-1, 1, h, device=x.device).unsqueeze(1).repeat(1, w).unsqueeze(0).unsqueeze(0)
        
        # Use learnable amplification
        yy_channel = yy_channel.expand(b, -1, -1, -1) * self.y_amplifier
        xx_channel = xx_channel.expand(b, -1, -1, -1) * self.x_amplifier
        
        coords = [x, xx_channel, yy_channel]
        
        if self.with_r:
            rr = torch.sqrt(xx_channel ** 2 + (yy_channel / self.y_amplifier) ** 2)  # Normalize for radius
            coords.append(rr)
        
        return torch.cat(coords, dim=1)


class CoordConv(nn.Module):
    """CoordConv layer: adds coords then applies Conv2d."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, with_r=False):
        super(CoordConv, self).__init__()
        self.addcoords = AddCoords(with_r=with_r)
        self.conv = nn.Conv2d(in_channels + 2 + (1 if with_r else 0), out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        x = self.addcoords(x)
        x = self.conv(x)
        return x


@torch.jit.script
def autocrop(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor):
    """
    Center-crops the encoder_layer to the size of the decoder_layer,
    so that merging (concatenation) between levels/blocks is possible.
    This is only necessary for input sizes != 2**n for 'same' padding and always required for 'valid' padding.
    """
    if encoder_layer.shape[2:] != decoder_layer.shape[2:]:
        ds = encoder_layer.shape[2:]
        es = decoder_layer.shape[2:]
        assert ds[0] >= es[0]
        assert ds[1] >= es[1]
        if encoder_layer.dim() == 4:  # 2D
            encoder_layer = encoder_layer[
                            :,
                            :,
                            ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                            ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2)
                            ]
        elif encoder_layer.dim() == 5:  # 3D
            assert ds[2] >= es[2]
            encoder_layer = encoder_layer[
                            :,
                            :,
                            ((ds[0] - es[0]) // 2):((ds[0] + es[0]) // 2),
                            ((ds[1] - es[1]) // 2):((ds[1] + es[1]) // 2),
                            ((ds[2] - es[2]) // 2):((ds[2] + es[2]) // 2),
                            ]
    return encoder_layer, decoder_layer


def conv_layer(dim: int):
    if dim == 3:
        return nn.Conv3d
    elif dim == 2:
        return nn.Conv2d


def get_conv_layer(in_channels: int,
                   out_channels: int,
                   kernel_size: int = 3,
                   stride: int = 1,
                   padding: int = 1,
                   bias: bool = True,
                   dim: int = 2):
    return conv_layer(dim)(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                           bias=bias)


def conv_transpose_layer(dim: int):
    if dim == 3:
        return nn.ConvTranspose3d
    elif dim == 2:
        return nn.ConvTranspose2d


def get_up_layer(in_channels: int,
                 out_channels: int,
                 kernel_size: int = 2,
                 stride: int = 2,
                 dim: int = 3,
                 up_mode: str = 'transposed',
                 ):
    if up_mode == 'transposed':
        return conv_transpose_layer(dim)(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
    else:
        return nn.Upsample(scale_factor=2.0, mode=up_mode)


def maxpool_layer(dim: int):
    if dim == 3:
        return nn.MaxPool3d
    elif dim == 2:
        return nn.MaxPool2d


def get_maxpool_layer(kernel_size: int = 2,
                      stride: int = 2,
                      padding: int = 0,
                      dim: int = 2):
    return maxpool_layer(dim=dim)(kernel_size=kernel_size, stride=stride, padding=padding)


def get_activation(activation: str):
    if activation == 'relu':
        return nn.ReLU()
    elif activation == 'leaky':
        return nn.LeakyReLU(negative_slope=0.1)
    elif activation == 'elu':
        return nn.ELU()


def get_normalization(normalization: str,
                      num_channels: int,
                      dim: int):
    if normalization == 'batch':
        if dim == 3:
            return nn.BatchNorm3d(num_channels)
        elif dim == 2:
            return nn.BatchNorm2d(num_channels)
    elif normalization == 'instance':
        if dim == 3:
            return nn.InstanceNorm3d(num_channels)
        elif dim == 2:
            return nn.InstanceNorm2d(num_channels)
    elif 'group' in normalization:
        num_groups = int(normalization.partition('group')[-1])  # get the group size from string
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, layer_1, layer_2):
        x = torch.cat((layer_1, layer_2), 1)

        return x


class DownBlock(nn.Module):
    """
    A helper Module that performs 2 Convolutions and 1 MaxPool.
    An activation follows each convolution.
    A normalization layer follows each convolution.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 pooling: bool = True,
                 activation: str = 'relu',
                 normalization: str = None,
                 dim: str = 2,
                 conv_mode: str = 'same'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.normalization = normalization
        if conv_mode == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0
        self.dim = dim
        self.activation = activation

        # conv layers
        self.conv1 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True, dim=self.dim)
        self.conv2 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True, dim=self.dim)

        # pooling layer
        if self.pooling:
            self.pool = get_maxpool_layer(kernel_size=2, stride=2, padding=0, dim=self.dim)

        # activation layers
        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)

        # normalization layers
        if self.normalization:
            self.norm1 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)
            self.norm2 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)

    def forward(self, x):
        y = self.conv1(x)  # convolution 1
        y = self.act1(y)  # activation 1
        if self.normalization:
            y = self.norm1(y)  # normalization 1
        y = self.conv2(y)  # convolution 2
        y = self.act2(y)  # activation 2
        if self.normalization:
            y = self.norm2(y)  # normalization 2

        before_pooling = y  # save the outputs before the pooling operation
        if self.pooling:
            y = self.pool(y)  # pooling
        return y, before_pooling


class UpBlock(nn.Module):
    """
    A helper Module that performs 2 Convolutions and 1 UpConvolution/Upsample.
    An activation follows each convolution.
    A normalization layer follows each convolution.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 activation: str = 'relu',
                 normalization: str = None,
                 dim: int = 3,
                 conv_mode: str = 'same',
                 up_mode: str = 'transposed'
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        if conv_mode == 'same':
            self.padding = 1
        elif conv_mode == 'valid':
            self.padding = 0
        self.dim = dim
        self.activation = activation
        self.up_mode = up_mode

        # upconvolution/upsample layer
        self.up = get_up_layer(self.in_channels, self.out_channels, kernel_size=2, stride=2, dim=self.dim,
                               up_mode=self.up_mode)

        # conv layers
        self.conv0 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0,
                                    bias=True, dim=self.dim)
        self.conv1 = get_conv_layer(2 * self.out_channels, self.out_channels, kernel_size=3, stride=1,
                                    padding=self.padding,
                                    bias=True, dim=self.dim)
        self.conv2 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True, dim=self.dim)

        # activation layers
        self.act0 = get_activation(self.activation)
        self.act1 = get_activation(self.activation)
        self.act2 = get_activation(self.activation)

        # normalization layers
        if self.normalization:
            self.norm0 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)
            self.norm1 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)
            self.norm2 = get_normalization(normalization=self.normalization, num_channels=self.out_channels,
                                           dim=self.dim)

        # concatenate layer
        self.concat = Concatenate()

    def forward(self, encoder_layer, decoder_layer):
        """ Forward pass
        Arguments:
            encoder_layer: Tensor from the encoder pathway
            decoder_layer: Tensor from the decoder pathway (to be up'd)
        """
        up_layer = self.up(decoder_layer)  # up-convolution/up-sampling
        cropped_encoder_layer, dec_layer = autocrop(encoder_layer, up_layer)  # cropping

        if self.up_mode != 'transposed':
            # We need to reduce the channel dimension with a conv layer
            up_layer = self.conv0(up_layer)  # convolution 0
        up_layer = self.act0(up_layer)  # activation 0
        if self.normalization:
            up_layer = self.norm0(up_layer)  # normalization 0

        merged_layer = self.concat(up_layer, cropped_encoder_layer)  # concatenation
        y = self.conv1(merged_layer)  # convolution 1
        y = self.act1(y)  # activation 1
        if self.normalization:
            y = self.norm1(y)  # normalization 1
        y = self.conv2(y)  # convolution 2
        y = self.act2(y)  # acivation 2
        if self.normalization:
            y = self.norm2(y)  # normalization 2
        return y


class UNet(nn.Module):
    """
    activation: 'relu', 'leaky', 'elu'
    normalization: 'batch', 'instance', 'group{group_size}'
    conv_mode: 'same', 'valid'
    dim: 2, 3
    up_mode: 'transposed', 'nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'
    """
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 2,
                 n_blocks: int = 1,
                 start_filters: int = 32,
                 activation: str = 'relu',  
                 normalization: str = 'batch',
                 conv_mode: str = 'same',
                 dim: int = 2,
                 up_mode: str = 'transposed',
                 decoder_output: list = [],
                 
                 ):
        super().__init__()
    

        #commented the fusion part for original UNet 
        
        print("in constructor inchannel: " + str(in_channels))
        
        #self.fusion =  CoordConv(in_channels, out_channels = 3, kernel_size=3, padding="same")
        self.fusion = nn.Conv2d(in_channels = in_channels, out_channels = 3, kernel_size = 3, padding="same")
        self.fusion2 = nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, padding="same")
        self.fusion3 = nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, padding="same")
        
        self.fusion4 = nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, padding="same")
        self.fusion5 = nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, padding="same")
        
        
        self.bn1 = nn.BatchNorm2d(3)
        
        
        # self.fusion2 = nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 3, padding="same", dilation = 21)
        
        # self.foveation = UltraFastFoveatedConv2d(in_channels = 108, out_channels = 3)
        
        # self.foveation2 = UltraFastFoveatedConv2d(in_channels = 3, out_channels = 3)
        
        
        # Version single 1x1
        #self.fusion = nn.Conv2d(in_channels, 1, 1, padding = 'same')
        #print(self.fusion.weight)
        #Version multiple 12.1
        #self.cn1 = nn.Conv2d(in_channels, out_channels = 3, kernel_size = 1, padding='same')
        # self.cn2 = nn.Conv2d(in_channels, 1, 3, padding='same')
        # self.cn3 = nn.Conv2d(in_channels, 1, 3, dilation = 2, padding='same')
        # self.cn4 = nn.Conv2d(in_channels, 1, 5, padding='same')
        # self.cn5 = nn.Conv2d(in_channels, 1, 5, dilation = 2, padding='same')
        
        
        #Version 3 increase the convolutions again
        # default common config conv
        #self.cn2 = nn.Conv2d(in_channels, out_channels = 3, kernel_size = 1, padding="same")
        
        #self.cn1 = nn.Conv2d(in_channels, out_channels = 1, kernel_size = 1, padding="same")
        #self.cn2 = nn.Conv2d(in_channels, out_channels = 3, kernel_size = 5, padding="same")
        
        #self.cn3 = nn.Conv2d(in_channels, out_channels = 1, kernel_size = 3, padding="same", dilation = 2)
        # self.cn4 = nn.Conv2d(in_channels, out_channels = 1, kernel_size = 3, padding="same", dilation = 3)
        # self.cn5 = nn.Conv2d(in_channels, out_channels = 1, kernel_size = 3, padding="same", dilation = 4)
        
        
        
        # self.cn7 = nn.Conv2d(in_channels, out_channels = 1, kernel_size = 5, padding="same", dilation = 2)
        # self.cn8 = nn.Conv2d(in_channels, out_channels = 1, kernel_size = 5, padding="same", dilation = 3)
        # self.cn9 = nn.Conv2d(in_channels, out_channels = 1, kernel_size = 5, padding="same", dilation = 4)
        
        #self.cn3 = nn.Conv2d(in_channels, out_channels = 3, kernel_size = 7, padding="same")
        #self.cn4 = nn.Conv2d(in_channels, out_channels = 3, kernel_size = 9, padding="same")
        #self.cn5 = nn.Conv2d(in_channels, out_channels = 3, kernel_size = 11, padding="same")
        #self.cn6 = nn.Conv2d(in_channels, out_channels = 3, kernel_size = 13, padding="same")
        # self.cn11 = nn.Conv2d(in_channels, out_channels = 1, kernel_size = 7, padding="same", dilation = 2)
        # self.cn12 = nn.Conv2d(in_channels, out_channels = 1, kernel_size = 7, padding="same", dilation = 3)
        # self.cn13 = nn.Conv2d(in_channels, out_channels = 1, kernel_size = 7, padding="same", dilation = 4)
        
        
        
        
        
        
        
        self.in_channels = 9
        ##uncommented this part for original UNet
        #self.in_channels = in_channels
        print("Input channel count" + str(self.in_channels))
        
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.start_filters = start_filters
        self.activation = activation
        self.normalization = normalization
        self.conv_mode = conv_mode
        self.dim = dim
        self.up_mode = up_mode

        self.down_blocks = []
        self.up_blocks = []

        self.activations = {}
        
        
        #recently added by eric decoder hook for visualization
        self.decoder_output = []
        
        
        def hook_fn(module, input, output):
        # Capture the output just before pooling (the second element in the tuple)
            self.decoder_output.append(output)
        
        # def hook_fn(module, input, output):
        # # Capture the output just before pooling (the second element in the tuple)
        #     self.activations[module] = output[1].detach()
        
        
        # create encoder path
        for i in range(self.n_blocks):
            num_filters_in = self.in_channels if i == 0 else num_filters_out
            num_filters_out = self.start_filters * (2 ** i)
            pooling = True if i < self.n_blocks - 1 else False

            down_block = DownBlock(in_channels=num_filters_in,
                                   out_channels=num_filters_out,
                                   pooling=pooling,
                                   activation=self.activation,
                                   normalization=self.normalization,
                                   conv_mode=self.conv_mode,
                                   dim=self.dim)

            self.down_blocks.append(down_block)
            #custom added feature analyze
            # Register the hook for each DownBlock
            # for idx, down_block in enumerate(self.down_blocks):
            #     down_block.register_forward_hook(hook_fn)


        # create decoder path (requires only n_blocks-1 blocks)
        for i in range(n_blocks - 1):
            num_filters_in = num_filters_out
            num_filters_out = num_filters_in // 2

            up_block = UpBlock(in_channels=num_filters_in,
                               out_channels=num_filters_out,
                               activation=self.activation,
                               normalization=self.normalization,
                               conv_mode=self.conv_mode,
                               dim=self.dim,
                               up_mode=self.up_mode)

            self.up_blocks.append(up_block)

        # for idx, block in enumerate(self.up_blocks):
        #         up_block.register_forward_hook(hook_fn)
                
        # final convolution
        self.conv_final = get_conv_layer(num_filters_out, self.out_channels, kernel_size=1, stride=1, padding=0,
                                         bias=True, dim=self.dim)
        
        

        # add the list of modules to current module
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)

        # initialize the weights
        self.initialize_parameters()

    @staticmethod
    def weight_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.weight, **kwargs)  # weights

    @staticmethod
    def bias_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.bias, **kwargs)  # bias

    def initialize_parameters(self,
                              method_weights=nn.init.xavier_uniform_,
                              method_bias=nn.init.zeros_,
                              kwargs_weights={},
                              kwargs_bias={}
                              ):
        for module in self.modules():
            self.weight_init(module, method_weights, **kwargs_weights)  # initialize weights
            self.bias_init(module, method_bias, **kwargs_bias)  # initialize bias

    def forward(self, x: torch.tensor):
        encoder_output = []
        
        
        
        # splitting the input tensor with 108 channels into 4 tensors and applying the 
        # nn conv2d operation then concatting the output tensor
        # result
        
        # split_tensors = torch.split(x, 27, dim=1)
        #print(torch.unique(x))
        #conv = nn.Conv2d(27, 1, 1, padding = 'same').cuda()
        #conv_tensors = [self.fusion(tensor) for tensor in split_tensors]
        #output_tensor = torch.cat(conv_tensors, dim=1)
        #print(output_tensor.shape)
        #x = output_tensor
        #print(torch.unique(x))
        #print(x.shape)
        
        # same conv2d trying 4 times result is just same
        # split_tensors = []
        # for i in range(4):
        #     tensor = self.fusion(x)
        #     split_tensors.append(tensor)
        # output_tensor = torch.cat(split_tensors, dim = 1)
        # x = output_tensor
        
        # x = F.relu(self.fusion(x))
        # x = F.relu(self.fusion2(x))
        # x = F.relu(self.fusion3(x))
        # x = F.relu(self.fusion4(x))
        # x = self.fusion5(x)
        
        
        x1 = self.fusion(x)
        x2 = self.fusion2(x1)
        x3 = self.fusion3(x2)
        # x = F.relu(x)
        
        #x1 = self.fusion(x) 
        #x2 = self.foveation(x)
        #x3 = self.foveation2(x1)
        #x4 = self.fusion2(x2)
        
        #x3 = self.foveation(x1)
        #x = torch.cat((x2, x4), dim=1)
        
        # x2 = self.cn2(x)
        # x3 = self.cn3(x)
        # x4 = self.cn4(x)
        # x5 = self.cn5(x)
        # x6 = self.cn6(x)
        #x3 = self.cn3(x)
        # x4 = self.cn4(x)
        # x5 = self.cn5(x)
        # x6 = self.cn6(x)
        # x7 = self.cn7(x)
        # x8 = self.cn8(x)
        # x9 = self.cn9(x)
        # x10 = self.cn10(x)
        # x11 = self.cn11(x)
        # x12 = self.cn12(x)
        # x13 = self.cn13(x)
        #x7 = x
        
        x = torch.cat((x1, x2, x3), dim=1)
                
        #x = x3
        # Encoder pathway
        for module in self.down_blocks:
            x, before_pooling = module(x)
            encoder_output.append(before_pooling)

        # Decoder pathway
        for i, module in enumerate(self.up_blocks):
            before_pool = encoder_output[-(i + 2)]
            x = module(before_pool, x)
            

        x = self.conv_final(x)
        
        # ## logic added by eric to illustrate segmask
        # sp_cnt = 0
        # if sp_cnt==0:
        #     self.pre_x = x
            
        #     seg_ten = x.argmax(dim = 1)
        #     img_np = tensor_to_image(seg_ten)
        #     save_image_unique("/home/eric/Documents/cervicalResearchIIP/result_test/20250604-unetlowshow/postFinalConvSegMaskOneforward/presegmask.png", img_np)
            
        #     sp_cnt = 1
            
        ##
        return x

    def __repr__(self):
        attributes = {attr_key: self.__dict__[attr_key] for attr_key in self.__dict__.keys() if '_' not in attr_key[0] and 'training' not in attr_key}
        d = {self.__class__.__name__: attributes}
        return f'{d}'
