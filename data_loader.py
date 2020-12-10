import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


# https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
class TestDataLoader(Dataset):
    def __init__(self, input_images, output):
        self.input_images = input_images
        self.output = output
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        output = self.output[idx]
        image = self.transform(image)
        return [image, output]

class TrainLoader(Dataset):
    def __init__(self, input_images, output, numth_fold):
        group_size = 120
        split_i = numth_fold * group_size
        train_images = np.vstack((input_images[:split_i], input_images[split_i+group_size:]))
        train_labels = np.hstack((output[:split_i], output[split_i+group_size:]))
        self.input_images, self.output = train_images, train_labels
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        output = self.output[idx]


        image = self.transform(image)

        return [image, output]

class TestLoader(Dataset):
    def __init__(self, input_images, output, numth_fold):
        group_size = 120
        split_i = numth_fold * group_size

        self.input_images = input_images[split_i:split_i+group_size]
        self.output = output[split_i:split_i+group_size]
        self.transform = transforms.ToTensor()
    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        image = self.input_images[idx]
        output = self.output[idx]



        image = self.transform(image)

        return [image, output]