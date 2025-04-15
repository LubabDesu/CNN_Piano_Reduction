import torch
import torch.nn as nn

class PianoReductionCNN(nn.Module) :
    def __init__(self) :
        super(PianoReductionCNN, self).__init__()

        self.down_conv1 = nn.Conv2d(4, 48, kernel_size=5, stride=2, padding=2)
        self.relu1 = nn.ReLU()
        self.flat_conv1 = nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.flat_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()

        self.down_conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.flat_conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu5 = nn.ReLU()
        self.flat_conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu6 = nn.ReLU()
        
        self.down_conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.relu7 = nn.ReLU()
        self.flat_conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu8 = nn.ReLU()
        self.flat_conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.relu9 = nn.ReLU()
        self.flat_conv7 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.relu10 = nn.ReLU()
        self.flat_conv8 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.relu11 = nn.ReLU()
        self.flat_conv9 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.relu12 = nn.ReLU()
        self.flat_conv10 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.relu13 = nn.ReLU()
        self.flat_conv11 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.relu14 = nn.ReLU()

        # Up-Convolution Layers
        self.up_conv1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.relu15 = nn.ReLU()
        self.flat_conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.relu16 = nn.ReLU()
        self.flat_conv13 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.relu17 = nn.ReLU()
        
        self.up_conv2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.relu18 = nn.ReLU()
        self.flat_conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.relu19 = nn.ReLU()
        self.flat_conv15 = nn.Conv2d(128, 48, kernel_size=3, stride=1, padding=1)
        self.relu20 = nn.ReLU()
        
        self.up_conv3 = nn.ConvTranspose2d(48, 48, kernel_size=4, stride=2, padding=1)
        self.relu21 = nn.ReLU()
        self.flat_conv16 = nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1)
        self.relu22 = nn.ReLU()
        self.flat_conv17 = nn.Conv2d(24, 1, kernel_size=3, stride=1, padding=1)

        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Down-Convolution Path
        x = self.relu1(self.down_conv1(x))
        x = self.relu2(self.flat_conv1(x))
        x = self.relu3(self.flat_conv2(x))
        
        x = self.relu4(self.down_conv2(x))
        x = self.relu5(self.flat_conv3(x))
        x = self.relu6(self.flat_conv4(x))
        
        x = self.relu7(self.down_conv3(x))
        x = self.relu8(self.flat_conv5(x))
        x = self.relu9(self.flat_conv6(x))
        x = self.relu10(self.flat_conv7(x))
        x = self.relu11(self.flat_conv8(x))
        x = self.relu12(self.flat_conv9(x))
        x = self.relu13(self.flat_conv10(x))
        x = self.relu14(self.flat_conv11(x))

        # Up-Convolution Path
        x = self.relu15(self.up_conv1(x))
        x = self.relu16(self.flat_conv12(x))
        x = self.relu17(self.flat_conv13(x))
        
        x = self.relu18(self.up_conv2(x))
        x = self.relu19(self.flat_conv14(x))
        x = self.relu20(self.flat_conv15(x))
        
        x = self.relu21(self.up_conv3(x))
        x = self.relu22(self.flat_conv16(x))
        x = self.sigmoid(self.flat_conv17(x))
        
        return x

