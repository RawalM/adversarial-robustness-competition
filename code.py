import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import argparse
import time
import copy

# input id
id_ = 201765354

# setup training parameters
parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

args = parser.parse_args(args=[]) 

# judge cuda is available or not
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

torch.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

############################################################################
################    don't change the below code    #####################
############################################################################
train_set = torchvision.datasets.FashionMNIST(root='data', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

test_set = torchvision.datasets.FashionMNIST(root='data', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

# define fully connected network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output

##############################################################################
#############    end of "don't change the below code"   ######################
##############################################################################

#generate adversarial data using Fast Gradient Sign Method (FGSM)
def adv_attack(model, X, y, device, epsilon=0.1):
    X_adv = Variable(X.data, requires_grad=True)
    
    # Forward pass
    output = model(X_adv)
    loss = F.nll_loss(output, y)
    
    # Backward pass
    model.zero_grad()
    loss.backward()

    # Generate adversarial examples
    perturbation = epsilon * X_adv.grad.sign()
    X_adv = X_adv + perturbation
    X_adv = torch.clamp(X_adv, 0, 1)  # Ensure valid pixel range

    return X_adv

#train function with adversarial training
def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), 28*28)
        
        # Generate adversarial examples
        adv_data = adv_attack(model, data, target, device)
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Compute loss (combine clean and adversarial losses)
        output_clean = model(data)
        output_adv = model(adv_data)
        loss_clean = F.nll_loss(output_clean, target)
        loss_adv = F.nll_loss(output_adv, target)
        loss = (loss_clean + loss_adv) / 2
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

#evaluate function
def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), 28*28)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def eval_adv_test(model, device, test_loader, epsilon=0.1):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), 28*28)
            adv_data = adv_attack(model, data, target, device, epsilon)
            output = model(adv_data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

#main function

def train_model():
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Training
        train(args, model, device, train_loader, optimizer, epoch)

        # Evaluation
        trn_loss, trn_acc = eval_test(model, device, train_loader)
        adv_loss, adv_acc = eval_adv_test(model, device, train_loader)

        print(f'Epoch {epoch}: {int(time.time() - start_time)}s, ', end='')
        print(f'trn_loss: {trn_loss:.4f}, trn_acc: {trn_acc*100:.2f}%, ', end='')
        print(f'adv_loss: {adv_loss:.4f}, adv_acc: {adv_acc*100:.2f}%')

    adv_tst_loss, adv_tst_acc = eval_adv_test(model, device, test_loader)
    print(f'Your estimated attack ability is: {1/adv_tst_acc:.4f}')
    print(f'Your estimated defense ability is: {adv_tst_acc:.4f}')

    torch.save(model.state_dict(), f'{id_}.pt')
    return model

#compute perturbation distance
def p_distance(model, train_loader, device):
    p = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), 28*28)
        data_ = copy.deepcopy(data.data)
        adv_data = adv_attack(model, data, target, device)
        p.append(torch.norm(data_ - adv_data, float('inf')))
    print('epsilon p: ', max(p))

# Run the model training
model = train_model()
p_distance(model, train_loader, device)
