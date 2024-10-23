from torch import nn
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torchsummary import summary

@torch.jit.script
def autocrop(encoder_layer: torch.Tensor, decoder_layer: torch.Tensor):
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


# Depthwise Separable Convolution
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, dim=2):
        super(DepthwiseSeparableConv, self).__init__()
        if dim == 2:
            
            self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                       padding=padding, groups=in_channels, bias=bias)
            self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        elif dim == 3:
            self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                       padding=padding, groups=in_channels, bias=bias)
            self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias)
        self.dim = dim

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


# ここで、DepthwiseSeparableConvを応用
def get_depthwise_separable_conv(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
                                 padding: int = 1, bias: bool = True, dim: int = 2):
    return DepthwiseSeparableConv(in_channels, out_channels, kernel_size, stride, padding, bias, dim)


def conv_layer(dim: int):
    if dim == 3:
        return nn.Conv3d
    elif dim == 2:
        return nn.Conv2d


def get_conv_layer(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1,
                   bias: bool = True, dim: int = 2, use_depthwise_separable: bool = True):
    if use_depthwise_separable:
        return get_depthwise_separable_conv(in_channels, out_channels, kernel_size, stride, padding, bias, dim)
    else:
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
        num_groups = int(normalization.partition('group')[-1])
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)


class Concatenate(nn.Module):
    def __init__(self):
        super(Concatenate, self).__init__()

    def forward(self, layer_1, layer_2):
        x = torch.cat((layer_1, layer_2), 1)
        return x


class DownBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 pooling: bool = True,
                 activation: str = 'relu',
                 normalization: str = None,
                 dim: int = 2,
                 conv_mode: str = 'same',
                 dropout_rate: float = 0.3
                 ): 
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.pooling = pooling
        self.normalization = normalization
        self.padding = 1 if conv_mode == 'same' else 0
        self.dim = dim
        self.activation = activation

        # conv layer
        self.conv1 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True, dim=self.dim, use_depthwise_separable=True)
        self.conv2 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True, dim=self.dim, use_depthwise_separable=True)
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
        # Dropout
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)


    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(y)
        if self.normalization:
            y = self.norm1(y)
        y = self.dropout1(y)  # Dropout応用
        y = self.conv2(y)
        y = self.act2(y)
        if self.normalization:
            y = self.norm2(y)
        y = self.dropout2(y)  # Dropout応用
        before_pooling = y
        if self.pooling:
            y = self.pool(y)
        return y, before_pooling


class UpBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 pooling: bool = True,
                 activation: str = 'relu',
                 normalization: str = None,
                 dim: int = 3,
                 conv_mode: str = 'same',
                 up_mode: str = 'transposed',
                 dropout_rate: float = 0.3
                 ): 
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.padding = 1 if conv_mode == 'same' else 0
        self.dim = dim
        self.activation = activation
        self.up_mode = up_mode

        # upsample layer
        self.up = get_up_layer(self.in_channels, self.out_channels, kernel_size=2, stride=2, dim=self.dim,
                               up_mode=self.up_mode)
        #conv layers
        self.conv0 = get_conv_layer(self.in_channels, self.out_channels, kernel_size=1, stride=1, padding=0, bias=True,
                                    dim=self.dim, use_depthwise_separable=True)
        self.conv1 = get_conv_layer(2 * self.out_channels, self.out_channels, kernel_size=3, stride=1,
                                    padding=self.padding, bias=True, dim=self.dim, use_depthwise_separable=True)
        self.conv2 = get_conv_layer(self.out_channels, self.out_channels, kernel_size=3, stride=1, padding=self.padding,
                                    bias=True, dim=self.dim, use_depthwise_separable=True)
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
        self.concat = Concatenate()

        # dropout layers
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

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

        merged_layer = self.concat(up_layer, cropped_encoder_layer) # concatenation
        y = self.conv1(merged_layer)
        y = self.act1(y)
        if self.normalization:
            y = self.norm1(y)
        y = self.dropout1(y)  # Dropout応用
        y = self.conv2(y)
        y = self.act2(y)
        if self.normalization:
            y = self.norm2(y)
        y = self.dropout2(y)  # Dropout応用
        return y

## ここからはMDFAのコード

# 通道注意力モジュール
class ChannelAttention(nn.Module):  # 通道部分を処理するモジュール
    def __init__(self, in_channel):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自適応平均プール、出力サイズは1x1
        self.fc = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)  # 1x1畳み込みで次元を減らす
        self.relu = nn.ReLU(inplace=True)  # ReLU活性化関数、インプレースでメモリを節約

    def forward(self, x):
        b, c, _, _ = x.size()  # バッチサイズとチャネル数を取得
        y = self.avg_pool(x)  # 自適応平均プールを適用
        y = self.fc(y)  # 1x1畳み込みを適用
        y = self.relu(y)  # ReLU活性化を適用
        y = nn.functional.interpolate(y, size=(x.size(2), x.size(3)), mode='nearest')  # yのサイズをxの空間次元に合わせる
        return x * y.expand_as(x)  # 計算された通道の重みをxに適用し、特徴を再調整

# 空間注意力モジュール
class SpatialAttention(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channel, 1, kernel_size=1, bias=False)  # 1x1畳み込みで空間的な注意を生成
        self.norm = nn.Sigmoid()  # Sigmoid関数で正規化

    def forward(self, x):
        y = self.Conv1x1(x)  # 1x1畳み込みを適用
        y = self.norm(y)  # Sigmoid関数を適用
        return x * y  # 空間の重みをxに適用し、空間的な注意を実現

# チャネルと空間の特徴を統合するモジュール
class CombineAttention(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        self.channel_att = ChannelAttention(in_channel)  # チャネルモジュールを生成
        self.spatial_att = SpatialAttention(in_channel)  # 空間モジュールを生成

    def forward(self, U):
        U_spatial = self.spatial_att(U)  # 空間モジュールでUを処理
        U_channel = self.channel_att(U)  # 通道モジュールでUを処理
        return torch.max(U_channel, U_spatial)  # 両者の逐次最大値を取り、チャネルと空間の注意を統合

# 多尺度空洞融合注意モジュール (MDFA)
class MDFA(nn.Module):
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(MDFA, self).__init__()
        # 第一分岐: 1x1畳み込み、チャネル次元を保持、空洞なし
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # 第二分岐: 3x3畳み込み、空洞率6、受容野 (receptive field)を拡大
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # 第三分岐: 3x3畳み込み、空洞率12、受容野をさらに拡大
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # 第四分岐: 3x3畳み込み、空洞率18、最大限の受容野拡大
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # 第五分岐: グローバル特徴抽出、全局平均プール後の1x1畳み込み処理
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        self.branch5_bn = nn.BatchNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        # 全分岐の出力を結合し、1x1畳み込みで次元を減らす
        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.BatchNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # 通道と空間の特徴を統合するモジュール
        self.combine_att = CombineAttention(in_channel=dim_out * 5)

    def forward(self, x):
        [b, c, row, col] = x.size()
        # 各分岐を適用
        conv1x1 = self.branch1(x)
        conv3x3_1 = self.branch2(x)
        conv3x3_2 = self.branch3(x)
        conv3x3_3 = self.branch4(x)
        # グローバル特徴抽出
        global_feature = torch.mean(x, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        global_feature = self.branch5_bn(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)
        # 全特徴を結合
        feature_cat = torch.cat([conv1x1, conv3x3_1, conv3x3_2, conv3x3_3, global_feature], dim=1)
        # 統合モジュールを適用し、通道と空間の特徴を強調
        enhanced_feature_cat = self.combine_att(feature_cat)
        enhanced_feature_cat = enhanced_feature_cat * feature_cat
        # 最終出力は次元削減を経る
        result = self.conv_cat(enhanced_feature_cat)

        return result


class UNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 2,
                 n_blocks: int = 1,
                 start_filters: int = 32,
                 activation: str = 'relu',
                 normalization: str = 'batch',  # batchnormalization
                 conv_mode: str = 'same',
                 dim: int = 2,
                 up_mode:str = 'transposed',
                 dropout_rate: float = 0.3  # dropout_rate parameter
                 ):  
        super().__init__()

        # 1x1Convを使う時（ここでは普通のConv2dと同じ扱い）
        #self.fusion = get_depthwise_separable_conv(in_channels, 1, kernel_size=1, stride=1, padding=0, bias=True, dim=dim)

        #self.in_channels = 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.start_filters = start_filters
        self.activation = activation
        self.normalization = normalization
        self.conv_mode = conv_mode
        self.dim = dim
        self.up_mode = up_mode
        self.dropout_rate = dropout_rate    # dropout_rateの保存

        # add the list of modules to current module
        self.down_blocks = []
        self.up_blocks = []
        
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
                                   dim=self.dim,
                                   dropout_rate=self.dropout_rate
                                   )  # dropout_rateもパラメータ

            self.down_blocks.append(down_block)

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
                               up_mode=self.up_mode,
                               dropout_rate=self.dropout_rate
                               )  # 传递 dropout_rate

            self.up_blocks.append(up_block)

        
        # final convolution
        self.conv_final = get_conv_layer(num_filters_out, self.out_channels, kernel_size=1, stride=1, padding=0,
                                         bias=True, dim=self.dim)

        # add the list of modules to current module
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)

        self.initialize_parameters()
        # MDFAをUNetの中に入れる
        self.mdfa = MDFA(dim_in=45, dim_out=in_channels) # feature num = 45の時

    @staticmethod
    def weight_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.weight, **kwargs)

    @staticmethod
    def bias_init(module, method, **kwargs):
        if isinstance(module, (nn.Conv3d, nn.Conv2d, nn.ConvTranspose3d, nn.ConvTranspose2d)):
            method(module.bias, **kwargs)

    def initialize_parameters(self, method_weights=nn.init.xavier_uniform_, method_bias=nn.init.zeros_, kwargs_weights={},
                              kwargs_bias={}):
        for module in self.modules():
            self.weight_init(module, method_weights, **kwargs_weights)
            self.bias_init(module, method_bias, **kwargs_bias)

    def forward(self, x: torch.tensor):
        encoder_output = []
        #　ここで1X1CONV->MDFA
        x = self.mdfa(x)
        #x = self.fusion(x)
        for module in self.down_blocks:
            x, before_pooling = module(x)
            encoder_output.append(before_pooling)
        for i, module in enumerate(self.up_blocks):
            before_pool = encoder_output[-(i + 2)]
            x = module(before_pool, x)
        x = self.conv_final(x)
        return x

    def __repr__(self):
        attributes = {attr_key: self.__dict__[attr_key] for attr_key in self.__dict__.keys() if '_' not in attr_key[0] and 'training' not in attr_key}
        d = {self.__class__.__name__: attributes}
        return f'{d}'
