import torch
import torch.nn as nn


def debug(x, name):
    # Uncomment for debugging
    # print(f"{name} → sum: {x.sum().item():.2f}, max: {x.max().item():.2f}, min: {x.min().item():.2f}")
    return x


class PianoReductionCNN(nn.Module):
    def __init__(self):
        super(PianoReductionCNN, self).__init__()

        self.down_conv1 = nn.Conv2d(4, 48, kernel_size=5, stride=2, padding=2)
        self.down_conv1_bn = nn.BatchNorm2d(48)
        self.relu1 = nn.ReLU()

        self.flat_conv1 = nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1)
        self.flat_conv1_bn = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.flat_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.flat_conv2_bn = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()

        self.down_conv2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.down_conv2_bn = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()

        self.flat_conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.flat_conv3_bn = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU()

        self.flat_conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.flat_conv4_bn = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU()

        self.down_conv3 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.down_conv3_bn = nn.BatchNorm2d(256)
        self.relu7 = nn.ReLU()

        self.flat_conv5 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.flat_conv5_bn = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU()

        self.flat_conv6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.flat_conv6_bn = nn.BatchNorm2d(1024)
        self.relu9 = nn.ReLU()

        self.flat_conv7 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.flat_conv7_bn = nn.BatchNorm2d(1024)
        self.relu10 = nn.ReLU()

        self.flat_conv8 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.flat_conv8_bn = nn.BatchNorm2d(1024)
        self.relu11 = nn.ReLU()

        self.flat_conv9 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.flat_conv9_bn = nn.BatchNorm2d(1024)
        self.relu12 = nn.ReLU()

        self.flat_conv10 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.flat_conv10_bn = nn.BatchNorm2d(512)
        self.relu13 = nn.ReLU()

        self.flat_conv11 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.flat_conv11_bn = nn.BatchNorm2d(256)
        self.relu14 = nn.ReLU()

        self.up_conv1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.up_conv1_bn = nn.BatchNorm2d(256)
        self.relu15 = nn.ReLU()

        self.flat_conv12 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.flat_conv12_bn = nn.BatchNorm2d(256)
        self.relu16 = nn.ReLU()

        self.flat_conv13 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.flat_conv13_bn = nn.BatchNorm2d(128)
        self.relu17 = nn.ReLU()

        self.up_conv2 = nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1)
        self.up_conv2_bn = nn.BatchNorm2d(128)
        self.relu18 = nn.ReLU()

        self.flat_conv14 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.flat_conv14_bn = nn.BatchNorm2d(128)
        self.relu19 = nn.ReLU()

        self.flat_conv15 = nn.Conv2d(128, 48, kernel_size=3, stride=1, padding=1)
        self.flat_conv15_bn = nn.BatchNorm2d(48)
        self.relu20 = nn.ReLU()

        self.up_conv3 = nn.ConvTranspose2d(48, 48, kernel_size=4, stride=2, padding=1)
        self.up_conv3_bn = nn.BatchNorm2d(48)
        self.relu21 = nn.ReLU()

        self.flat_conv16 = nn.Conv2d(48, 24, kernel_size=3, stride=1, padding=1)
        self.flat_conv16_bn = nn.BatchNorm2d(24)
        self.relu22 = nn.ReLU()

        self.flat_conv17 = nn.Conv2d(24, 1, kernel_size=3, stride=1, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.down_conv1_bn(self.down_conv1(x)))
        debug(x, "down_conv1")
        x = self.relu2(self.flat_conv1_bn(self.flat_conv1(x)))
        debug(x, "flat_conv1")
        x = self.relu3(self.flat_conv2_bn(self.flat_conv2(x)))
        debug(x, "flat_conv2")

        x = self.relu4(self.down_conv2_bn(self.down_conv2(x)))
        debug(x, "down_conv2")
        x = self.relu5(self.flat_conv3_bn(self.flat_conv3(x)))
        debug(x, "flat_conv3")
        x = self.relu6(self.flat_conv4_bn(self.flat_conv4(x)))
        debug(x, "flat_conv4")

        x = self.relu7(self.down_conv3_bn(self.down_conv3(x)))
        debug(x, "down_conv3")
        x = self.relu8(self.flat_conv5_bn(self.flat_conv5(x)))
        debug(x, "flat_conv5")
        x = self.relu9(self.flat_conv6_bn(self.flat_conv6(x)))
        debug(x, "flat_conv6")
        x = self.relu10(self.flat_conv7_bn(self.flat_conv7(x)))
        debug(x, "flat_conv7")
        x = self.relu11(self.flat_conv8_bn(self.flat_conv8(x)))
        debug(x, "flat_conv8")
        x = self.relu12(self.flat_conv9_bn(self.flat_conv9(x)))
        debug(x, "flat_conv9")
        x = self.relu13(self.flat_conv10_bn(self.flat_conv10(x)))
        debug(x, "flat_conv10")
        x = self.relu14(self.flat_conv11_bn(self.flat_conv11(x)))
        debug(x, "flat_conv11")

        x = self.relu15(self.up_conv1_bn(self.up_conv1(x)))
        debug(x, "up_conv1")
        x = self.relu16(self.flat_conv12_bn(self.flat_conv12(x)))
        debug(x, "flat_conv12")
        x = self.relu17(self.flat_conv13_bn(self.flat_conv13(x)))
        debug(x, "flat_conv13")

        x = self.relu18(self.up_conv2_bn(self.up_conv2(x)))
        debug(x, "up_conv2")
        x = self.relu19(self.flat_conv14_bn(self.flat_conv14(x)))
        debug(x, "flat_conv14")
        x = self.relu20(self.flat_conv15_bn(self.flat_conv15(x)))
        debug(x, "flat_conv15")

        x = self.relu21(self.up_conv3_bn(self.up_conv3(x)))
        debug(x, "up_conv3")
        x = self.relu22(self.flat_conv16_bn(self.flat_conv16(x)))
        debug(x, "flat_conv16")

        x = self.sigmoid(self.flat_conv17(x))
        debug(x, "flat_conv17")
        return x


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')


model = PianoReductionCNN()
model.apply(weights_init)