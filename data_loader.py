from torch.utils.data import Dataset
from torchvision import transforms


# Data loader for Pytorch training process
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# Load test data
class TestDataLoader(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
        self.transform = transforms.ToTensor()
        self.trans = transforms.Grayscale()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return [self.transform(self.images[idx]), self.labels[idx]]


# Load training data
class TrainLoader(Dataset):
    def __init__(self, images, label):
        self.images = images[200:]
        self.labels = label[200:]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return [self.transform(self.images[idx]), self.labels[idx]]


# Load validation data
class ValidationLoader(Dataset):
    def __init__(self, images, labels):
        self.images = images[0:200]
        self.labels = labels[0:200]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return [self.transform(self.images[idx]), self.labels[idx]]

