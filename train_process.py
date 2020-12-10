import numpy as np
import torch
import torch.nn as nn
from model import ResCNN, ConvNet
from data_loader import TrainLoader, TestDataLoader, TestLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def train(epoch, num_epochs, optimizer, trainloader, model, criterion):
    model.train()
    running_loss = 0
    accuracy = 0
    for images, labels in trainloader:
        print(labels)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        ps = torch.exp(outputs)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
        running_loss += loss.item()
    train_acc = accuracy / len(trainloader)
    print('Training: Epoch [%d/%d] Loss: %.4f, Accuracy: %.4f' % (
        epoch + 1, num_epochs, running_loss / len(trainloader), train_acc))
    return model, running_loss / len(trainloader)


def validation_process(criterion, testloader, model):
    accuracy = 0
    running_loss = 0
    for images, labels in testloader:
        outputs = model(images)
        print(outputs)
        loss = criterion(outputs, labels)
        ps = torch.exp(outputs)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
        running_loss += loss.item()
    val_loss = running_loss / len(testloader)
    val_acc = accuracy / len(testloader)

    return val_acc, val_loss, outputs


def mainLoop(model, num_epochs, learn_rate, trainloader, testloader, criterion):
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    best_loss = 1e10
    best_model = None
    loss_list_train = []
    loss_list_test = []
    epoch_list = range(1,num_epochs+1)
    for epoch in range(num_epochs):
        model, loss = train(epoch, num_epochs, optimizer, trainloader, model, criterion)
        model.eval()
        val_acc, val_loss, _ = validation_process(criterion, testloader, model)
        print('Validation: Epoch [%d/%d], Val Loss: %.4f, Accuracy: %.4f' % (
            epoch + 1, num_epochs, val_loss, val_acc))
        epoch_loss = val_loss
        loss_list_train += [loss]
        loss_list_test += [epoch_loss]
        if epoch_loss < best_loss:
            print("Saving best model")
            best_loss = epoch_loss
            best_model = model
    train1, = plt.plot(epoch_list,[item + 1 for item in loss_list_train],  label = "train_loss")
    validation,  = plt.plot(epoch_list,[item + 1 for item in loss_list_test],  label = "validation loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend([train1, validation], ['Loss of training process', 'Loss of validation process'])
    plt.show()
    return best_model, best_loss

def final_test(model):
    test_set = TestDataLoader(val_pixels, val_labels)
    test_loader = torch.utils.data.DataLoader(test_set)
    criterion = nn.CrossEntropyLoss()
    accuracy = 0
    running_loss = 0
    for images, labels in test_loader:

        outputs = model(images)
        print(outputs)
        loss = criterion(outputs, labels)
        ps = torch.exp(outputs)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor))
        running_loss += loss.item()
    accuracy /= len(test_loader)
    print("test accuracy")
    print(accuracy)


def train_model():
    models_store = []
    for i in range(k_fold):
        print("%dth fold" % (i+1))
        train_set = TrainLoader(train_pixels , train_labels, i)
        val_set = TestLoader(train_pixels , train_labels, i)
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True)
        testloader = torch.utils.data.DataLoader(val_set, batch_size=test_batch_size, shuffle=True)
        model, test_loss = mainLoop(ConvNet(), num_epochs, learn_rate, trainloader, testloader, nn.CrossEntropyLoss())
        final_test(model)
        models_store.append(model)
    return models_store




if __name__ == '__main__':

    model = ConvNet()
    print(model)
    #https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    train_pixels = np.load('./numpy_data/train_images.npy')
    train_labels = np.load('./numpy_data/train_labels.npy')
    val_pixels = np.load('./numpy_data/val_images.npy')
    val_labels = np.load('./numpy_data/val_labels.npy')
    num_epochs = 20
    train_batch_size = 128
    test_batch_size=20
    learn_rate = 0.001
    k_fold = 1


    models_store = train_model()
