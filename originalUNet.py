import torch
import torch.nn as nn
import torch.nn.functional as F


#################################
# Might be used later original Eric UNet
#################################

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Original_UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.encoder = nn.ModuleList([
            DoubleConv(in_channels, 64),
            DoubleConv(64, 128),
            DoubleConv(128, 256),
            DoubleConv(256, 512),
            DoubleConv(512, 1024)
        ])
        
        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            DoubleConv(1024, 512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            DoubleConv(512, 256),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            DoubleConv(256, 128),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            DoubleConv(128, 64)
        ])

        self.output = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []
        
        # Encoder
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Decoder
        for i in range(0, len(self.decoder), 2):
            x = self.decoder[i](x)
            skip_connection = skip_connections[-(i//2 + 2)]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=True)
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[i+1](x)
        
        # Output layer
        x = self.output(x)
        return x