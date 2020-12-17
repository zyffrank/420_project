from torch.utils.data import Dataset
from torchvision import transforms


# Data loader for Pytorch training process
# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# Load test data
class TestDataLoader(Dataset):
    def __init__(self, input_images, output):
        self.input_images = input_images
        self.output = output
        self.transform = transforms.ToTensor()
        self.trans = transforms.Grayscale()

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        output = self.output[idx]
        image = self.transform(image)
        return [image, output]

# Load training data
class TrainLoader(Dataset):
    def __init__(self, input_images, output):

        train_images = input_images[200:]
        train_labels = output[200:]
        self.input_images, self.output = train_images, train_labels
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        output = self.output[idx]
        image = self.transform(image)
        return [image, output]

# Load validation data
class ValidationLoader(Dataset):
    def __init__(self, input_images, output):
        self.input_images = input_images[0:200]
        self.output = output[0:200]
        self.transform = transforms.ToTensor()
    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        output = self.output[idx]
        image = self.transform(image)
        return [image, output]

