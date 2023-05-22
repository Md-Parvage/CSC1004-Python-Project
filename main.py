from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import multiprocessing as mp
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from utils.config_utils import read_args, load_config, Dict2Object


#mean values
m_trainacc123, m_trainacc321, m_trainacc666 = [], [], []
m_trainloss123, m_trainloss321, m_trainloss666 = [], [], []
m_testacc123, m_testacc321, m_testacc666 = [], [], []
m_testloss123, m_testloss321, m_testloss666 = [], [], []

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    """
    tain the model and return the training accuracy
    :param args: input arguments
    :param model: neural network model
    :param device: the device where model stored
    :param train_loader: data loader
    :param optimizer: optimizer
    :param epoch: current epoch
    :param log_file: path to the log file
    :return:
    """
    model.train()
    correct = 0
    loss_value = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        # Calculate training accuracy and loss
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        loss_value += loss.item()

    # Calculate training accuracy and loss
    training_acc = 100. * correct / len(train_loader.dataset)
    training_loss = loss_value / len(train_loader)

    # Write to log file
    with open("train.txt", "a") as f:
        f.write(f"Training: epoch {epoch} | loss: {training_loss:.4f} | accuracy: {training_acc:.2f}\n")

    # Print training accuracy and loss
    print(f"Training: epoch {epoch} | loss: {training_loss:.4f} | accuracy: {training_acc:.2f}")

    return training_acc, training_loss


def test(model, device, test_loader, epoch):
    """
    test the model and return the testing accuracy
    :param model: neural network model
    :param device: the device where model stored
    :param test_loader: data loader
    :param filename: the name of the file to which to write the testing results
    :return:
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)

        #write the current testing accuracy and loss to the file
        with open("test.txt", 'a') as f:
            f.write(f"Epoch {epoch}, Testing accuracy: {(100. * correct / len(test_loader.dataset)):.2f}%, "
                    f"Testing loss: {test_loss:.4f}\n")

    testing_acc = 100. * correct / len(test_loader.dataset)
    testing_loss = test_loss
    print(f"Testing: epoch {epoch} | loss: {testing_loss:.4f} | accuracy: {testing_acc:.2f}")

    return testing_acc, testing_loss


def plot(title, xlabel, ylabel, xdata, ydata):
    plt.plot(xdata, ydata, marker = "o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def run(config):
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    use_mps = not config.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(config.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': config.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': config.test_batch_size, 'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # download data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)

    """add random seed to the DataLoader, pls modify this function"""
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)

    """record the performance"""
    epoches = [i+1 for i in range(config.epochs)]
    training_accuracies = []
    training_loss = []
    testing_accuracies = []
    testing_loss = []

    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    for epoch in range(1, config.epochs + 1):
        train_acc, train_loss = train(config, model, device, train_loader, optimizer, epoch)
        """record training info, Fill your code"""
        test_acc, test_loss = test(model, device, test_loader, epoch)
        """record testing info, Fill your code"""
        scheduler.step()
        """update the records, Fill your code"""
        training_accuracies.append(train_acc)
        training_loss.append(train_loss)
        testing_accuracies.append(test_acc)
        testing_loss.append(test_loss)

        if(config.seed == 123):
            m_trainacc123.append(train_acc)
            m_trainloss123.append(train_loss)
            m_testacc123.append(test_acc)
            m_testloss123.append(test_loss)
        elif(config.seed == 321):
            m_trainacc321.append(train_acc)
            m_trainloss321.append(train_loss)
            m_testacc321.append(test_acc)
            m_testloss321.append(test_loss)
        else:
            m_trainacc666.append(train_acc)
            m_trainloss666.append(train_loss)
            m_testacc666.append(test_acc)
            m_testloss666.append(test_loss)

    """plotting training performance with the records"""
    plot(f"Training Loss (seed:{config.seed})", "Epoch", "Loss", epoches, training_loss)
    plot(f"Training Accuracy (seed:{config.seed})", "Epoch", "Accuracy (%)", epoches, training_accuracies)

    """plotting testing performance with the records"""
    plot(f"Testing Loss (seed:{config.seed})", "Epoch", "Loss", epoches, testing_loss)
    plot(f"Testing Accuracy (seed:{config.seed})", "Epoch", "Accuracy (%)", epoches, testing_accuracies)


    mean_training_loss = np.mean(training_loss)
    mean_testing_loss = np.mean(testing_loss)
    mean_testing_accuracy = np.mean(testing_accuracies)

    plt.figure()
    plt.plot(training_loss, label='Training Loss')
    plt.plot(training_accuracies, label='Training Accuracy')
    plt.plot(testing_loss, label='Testing Loss')
    plt.plot(testing_accuracies, label='Testing Accuracy')
    plt.axhline(y=mean_training_loss, color='r', linestyle='--', label=f'Mean Training Loss: {mean_training_loss:.4f}')
    plt.axhline(y=mean_training_loss, color='r', linestyle='--', label=f'Mean Training Loss: {mean_training_loss:.4f}')
    plt.axhline(y=mean_testing_loss, color='g', linestyle='--', label=f'Mean Testing Loss: {mean_testing_loss:.4f}')
    plt.axhline(y=mean_testing_accuracy, color='b', linestyle='--', label=f'Mean Testing Accuracy: {mean_testing_accuracy:.2f}%')
    plt.title(f'Training and Testing Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Loss / Accuracy')
    plt.legend()
    plt.show()

    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")

def main(config, seed):
    # Set the random seed for the process
    config.seed = seed

    # Call the run function with the updated configuration
    run(config)

if __name__ == '__main__':
    arg = read_args()

    # Load training settings
    config = load_config(arg)

    # Create a list of processes
    processes = [mp.Process(target=main, args=(config, seed)) for seed in config.seed]

    # Start each process
    for p in processes:
        p.start()

    # Wait for each process to finish
    for p in processes:
        p.join()



    #avg. training loss
    mean_trainloss123 = np.mean(m_trainloss123)
    mean_trainloss321 = np.mean(m_trainloss321)
    mean_trainloss666 = np.mean(m_trainloss666)

    plt.figure()
    plt.axhline(y=mean_trainloss123, color='r', linestyle='--', label=f'Mean Training Loss (123): {mean_trainloss123:.4f}')
    plt.axhline(y=mean_trainloss321, color='g', linestyle='--', label=f'Mean Training Loss (321): {mean_trainloss321:.4f}')
    plt.axhline(y=mean_trainloss666, color='b', linestyle='--', label=f'Mean Training Loss (666): {mean_trainloss666:.4f}')
    plt.title(f'Mean Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    #avg. training acc
    mean_trainacc123 = np.mean(m_trainacc123)
    mean_trainacc321 = np.mean(m_trainacc321)
    mean_trainacc666 = np.mean(m_trainacc666)

    plt.figure()
    plt.axhline(y=mean_trainacc123, color='r', linestyle='--', label=f'Mean Training Accuracy (123): {mean_trainacc123:.2f}%')
    plt.axhline(y=mean_trainacc321, color='g', linestyle='--', label=f'Mean Training Accuracy (321): {mean_trainacc321:.2f}%')
    plt.axhline(y=mean_trainacc666, color='b', linestyle='--', label=f'Mean Training Accuracy (666): {mean_trainacc666:.2f}%')
    plt.title(f'Mean Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


    #avg. testing loss
    mean_testloss123 = np.mean(m_testloss123)
    mean_testloss321 = np.mean(m_testloss321)
    mean_testloss666 = np.mean(m_testloss666)

    plt.figure()
    plt.axhline(y=mean_testloss123, color='r', linestyle='--', label=f'Mean Testing Loss (123): {mean_testloss123:.4f}')
    plt.axhline(y=mean_testloss321, color='g', linestyle='--', label=f'Mean Testing Loss (321): {mean_testloss321:.4f}')
    plt.axhline(y=mean_testloss666, color='b', linestyle='--', label=f'Mean Testing Loss (666): {mean_testloss666:.4f}')
    plt.title(f'Mean Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


    #avg. testing acc
    mean_testacc123 = np.mean(m_testacc123)
    mean_testacc321 = np.mean(m_testacc321)
    mean_testacc666 = np.mean(m_testacc666)

    plt.figure()
    plt.axhline(y=mean_testacc123, color='r', linestyle='--', label=f'Mean Testing Accuracy (123): {mean_testacc123:.2f}%')
    plt.axhline(y=mean_testacc321, color='g', linestyle='--', label=f'Mean Testing Accuracy (321): {mean_testacc321:.2f}%')
    plt.axhline(y=mean_testacc666, color='b', linestyle='--', label=f'Mean Testing Accuracy (666): {mean_testacc666:.2f}%')
    plt.title(f'Mean Testing Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
