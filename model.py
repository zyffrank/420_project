import torch
import torch.nn as nn

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(18432, 1000)
        self.fc2 = nn.Linear(1000, 7)

    def forward(self, x):
        out = self.layer1(x)
        # out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, kernel, padding):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel, padding=padding)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel, padding=padding)
        self.bn2 = nn.BatchNorm2d(in_channel)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=kernel, padding=padding)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.bn1(out1)
        out3 = self.relu1(out2)
        out4 = self.conv2(out3)
        out5 = self.bn2(out4)
        merge = torch.cat((out5, x), dim = 1)
        out6 = self.relu2(merge)
        out_final = self.pool1(out6)
        return out_final

# ResCNN

class ResCNN(nn.Module):
    def __init__(self, in_channel,num_class, kernel):
        super(ResCNN, self).__init__()
        padding = kernel // 2
        self.conv1 = nn.Conv2d(1, in_channel, kernel_size=kernel, padding=padding)
        self.pool1 = nn.MaxPool2d(kernel_size=kernel)
        self.res1 = ResidualBlock(in_channel, kernel, padding)
        self.res2 = ResidualBlock(in_channel*2, kernel, padding)
        self.res3 = ResidualBlock(in_channel*4, kernel, padding)
        self.res4 = ResidualBlock(in_channel*8, kernel, padding)
        self.fc1 = nn.Linear(in_channel*16, in_channel*16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(in_channel*16, in_channel*16)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(in_channel*16, num_class)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.pool1(out1)
        out3 = self.res1(out2)
        out4 = self.res2(out3)
        out5 = self.res3(out4)
        out6 = self.res4(out5)
        out6 = out6.view(out6.size(0), -1)
        out7 = self.fc1(out6)
        out7 = self.relu1(out7)
        out8 = self.fc2(out7)
        out8 = self.relu2(out8)
        out9 = self.fc3(out8)
        return out9
