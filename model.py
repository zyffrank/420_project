import torch.nn as nn

# Simple convolutional neural network
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
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))
        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=2),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.linear1 = nn.Linear(9216, 50)
        self.linear2 = nn.Linear(50, 7)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        # out = self.layer5(out)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.linear2(out)
        out = self.soft(out)
        return out

# Main block of our residual networks
class ResidualBlockAdd(nn.Module):
    def __init__(self):
        super(ResidualBlockAdd, self).__init__()
        self.conv1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = self.relu2(out)
        out = self.pool1(out)
        return out


# Main body of our residual network
class ResAdd(nn.Module):
    def __init__(self):
        super(ResAdd, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.res1 = ResidualBlockAdd()
        self.res2 = ResidualBlockAdd()
        self.res3 = ResidualBlockAdd()
        self.res4 = ResidualBlockAdd()
        self.res5 = ResidualBlockAdd()
        self.linear = nn.Linear(400, 7)
        self.soft = nn.Softmax(dim = 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.soft(out)
        return out


