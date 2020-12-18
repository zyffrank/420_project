import numpy as np
from model import ConvNet, ResAdd
from data_loader import TrainLoader, TestDataLoader, ValidationLoader
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


# Main training process
def train(epoch, optimizer, trainloader, model, criterion):
    model.train()
    total_right = 0
    total_loss = 0
    for images, labels in trainloader:
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        predict = torch.argmax(output, dim=1)
        equals = torch.eq(labels, predict).type(torch.FloatTensor)
        total_right += torch.mean(equals)
        total_loss += loss.item()
    accuracy = total_right/len(trainloader)
    loss = total_loss/len(trainloader)
    epoch += 1
    print('Training process: Epoch {}, Accuracy {}, Loss {}'.format(str(epoch), str(accuracy), str(loss)))
    return model, loss, accuracy


# Validation process
def validation_process(criterion, testloader, model):
    total_right = 0
    total_loss = 0
    total = len(testloader)
    for images, labels in testloader:
        output = model(images)
        loss = criterion(output, labels)
        predict_label = torch.argmax(output, dim=1)
        equals = torch.eq(labels, predict_label).type(torch.FloatTensor)
        total_right += torch.mean(equals)
        total_loss += loss.item()
    accuracy = total_right/total
    loss = total_loss/total
    return accuracy, loss


# Loop over each epoch and generate graph
def entry(model, num_epochs, learn_rate, trainloader, testloader, criterion):
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    loss_list_train = []
    loss_list_test = []
    acc_list_train = []
    acc_list_test = []
    epoch_list = range(1, num_epochs+1)
    for epoch in range(num_epochs):
        model, loss, acc = train(epoch, optimizer, trainloader, model, criterion)
        model.eval()
        validation_accuracy, validation_loss = validation_process(criterion, testloader, model)
        epoch_count = epoch + 1
        print('Validation process: Epoch {}, Accuracy {}, Loss: {}'.format(str(epoch_count), str(validation_accuracy),
                                                                           str(validation_loss)))
        epoch_loss = validation_loss
        loss_list_train += [loss]
        loss_list_test += [epoch_loss]
        acc_list_train += [acc]
        acc_list_test += [validation_accuracy]
    train1, = plt.plot(epoch_list, [item for item in loss_list_train],  label="Train loss")
    validation,  = plt.plot(epoch_list, [item for item in loss_list_test],  label="Validation loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend([train1, validation], ['Loss of training process', 'Loss of validation process'])
    plt.show()
    train1, = plt.plot(epoch_list, [item for item in acc_list_train],  label="Train accuracy")
    validation,  = plt.plot(epoch_list, [item for item in acc_list_test],  label="Validation accuracy")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend([train1, validation], ['Accuracy of training process', 'Accuracy of validation process'])
    plt.show()
    return model


# Test our trained model on test data
def final_test(model):
    model.eval()
    test_set = TestDataLoader(test_images, test_labels)
    test_loader = torch.utils.data.DataLoader(test_set)
    criterion = nn.CrossEntropyLoss()
    num_right = 0
    for images, labels in test_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)
        predict_label = torch.argmax(outputs, dim=1)
        equals = torch.eq(labels, predict_label).type(torch.FloatTensor)
        # if torch.mean(equals) == 0:
        #     image = torch.Tensor.numpy(images)
        #     print(labels)
        #     plt.imshow(image[0][0])
        #     plt.show()
        num_right += torch.mean(equals)
    accuracy = num_right/len(test_loader)
    print("test accuracy: " + str(accuracy))


def prepare():
    training = TrainLoader(train_images, train_labels)
    validation = ValidationLoader(train_images, train_labels)
    train_data_loader = torch.utils.data.DataLoader(training, 50, shuffle=True)
    test_data_loader = torch.utils.data.DataLoader(validation, 10, shuffle=True)
    model = entry(ResAdd(), num_epochs, learning_rate, train_data_loader, test_data_loader, nn.CrossEntropyLoss())
    final_test(model)
    return model


if __name__ == '__main__':
    # https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    train_images = np.load('./numpy/train_images.npy')
    train_labels = np.load('./numpy/train_labels.npy')
    test_images = np.load('./numpy/test_images.npy')
    test_labels = np.load('./numpy/test_labels.npy')

    num_epochs = 40
    learning_rate = 0.0002
    models_store = prepare()
