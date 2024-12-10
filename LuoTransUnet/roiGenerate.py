import torch
import torch.nn as nn
import numpy as np

import torch
import torch.nn as nn

class ROIGenerator(nn.Module):
    def __init__(self, inplanes, outplanes, imgSize=256):
        super(ROIGenerator, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(128 * (imgSize//4) * (imgSize//4), 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 4)  
        self.imgSize = imgSize
        self.outplanes = outplanes

    def forward(self, x):
        mask = torch.zeros((1, self.outplanes, self.imgSize, self.imgSize), dtype=torch.int)
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 128 * (self.imgSize//4) * (self.imgSize//4))
        x = self.relu3(self.fc1(x))
        roi = self.fc2(x)
        #assert False,str(roi.size())
        coor_x, coor_y, width, height = self.fc2(x)[0]
        coor_x = int(coor_x*self.imgSize+self.imgSize//2)
        width = int(width*self.imgSize)
        coor_y = int(coor_y*self.imgSize+self.imgSize//2)
        height = int(height*self.imgSize)
        #assert False,str(coor_x)+' '+str(coor_y)+' '+str(width)+' '+str(height)
        mask[:, coor_y:coor_y+height, coor_x:coor_x+width] = 1
        mask=mask.to(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return mask

if __name__ == '__main__':
    roigenerator = ROIGenerator()

    input_image = torch.randn(1, 3, 256, 256)

    output_roi = roigenerator(input_image)
    print("Generated ROI:", output_roi)
