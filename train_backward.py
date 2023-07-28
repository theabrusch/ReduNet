import argparse
import os

import torch
import torch.nn as nn
import wandb

from redunet import *
import evaluate
import load as L
import functional as F
import utils
import plot
PYTORCH_ENABLE_MPS_FALLBACK=1



parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=False, help='choice of dataset', default='mnistvector')
parser.add_argument('--mnist_binary', type=eval, default='True', help='set to True if mnist binary')
parser.add_argument('--layers', type=int, required=False, help='choice of architecture', default=5)
parser.add_argument('--channels', type=int, required=False, help='choice of architecture', default=16)

parser.add_argument('--samples', type=eval, required=False, help="number of samples per update", default=None)
parser.add_argument('--tail', type=str, default='', help='extra information to add to folder name')
parser.add_argument('--log', default=True, help='set to True if log to wandb')
parser.add_argument('--load_model', default=False, help='set to True if load model')
parser.add_argument('--model_dir', type=str, help='model directory')
parser.add_argument('--save_dir', type=str, default='./saved_models/', help='base directory for saving.')
parser.add_argument('--data_dir', type=str, default='./data/', help='base directory for saving.')

parser.add_argument('--batch_size', type=int, default=100, help='batch size for evaluation')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate for training')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
args = parser.parse_args()
## CUDA
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

if args.log:
    if args.load_model:
        group = 'ReduNet_init_bp'
    else:
        group = 'ReduNet_bp'
    wandb.init(project='redunet', group = group, entity='theabrusch')
    wandb.config.update(args)

## Model Directory
model_dir = os.path.join(args.save_dir, 
                         'backward',
                         f'{args.data}+ch{args.channels}+l{args.layers}' if not args.mnist_binary else f'{args.data}+ch{args.channels}+l{args.layers}+binary',
                         f'samples{args.samples}'
                         f'{args.tail}')
os.makedirs(model_dir, exist_ok=True)
utils.save_params(model_dir, vars(args))
print(model_dir)

## Data
trainset, testset, num_classes = L.load_dataset(args.data, data_dir=args.data_dir, )
X_train, y_train = F.get_samples(trainset, args.samples, binary=args.mnist_binary)
X_train, y_train = X_train.to(device), y_train.to(device)
train_dset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=args.batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size)

## Architecture
net = L.load_architecture(args.data, args.channels, args.layers)
if args.load_model:
    net = utils.load_ckpt(args.model_dir, 'model', net)

channels = 1 if args.data == 'mnistvector' else args.channels
num_classes = 2 if args.mnist_binary else num_classes

classifier = nn.Sequential(
    net, 
    nn.Flatten(),
    nn.Linear(channels*28*28, num_classes),
    nn.Softmax(dim=1)
)
classifier.to(device)
optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate, weight_decay=1e-4)

## Training
for epoch in range(args.epochs):
    classifier.train()
    train_loss = 0
    for i, (X, y) in enumerate(train_loader):
        X, y = X.to(device), y.to(device)
        y_pred = classifier(X)
        loss = nn.CrossEntropyLoss()(y_pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
    train_loss /= i+1
    classifier.eval()
    test_loss = 0
    for i, (X, y) in enumerate(test_loader):
        X, y = X.to(device), y.to(device)
        y_pred = classifier(X)
        loss = nn.CrossEntropyLoss()(y_pred, y)
        test_loss += loss.item()
    test_loss /= i+1
    if args.log:
        wandb.log({'train_loss': train_loss, 'test_loss': test_loss})
    else:
        print(f'Epoch {epoch+1}/{args.epochs}: train_loss: {train_loss}, test_loss: {test_loss}')
    
# evaluate model on test set
classifier.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for X, y in test_loader:
        X, y = X.to(device), y.to(device)
        if args.mnist_binary:
            y = (y > 4).long()
        y_pred = classifier(X)
        test_loss += nn.CrossEntropyLoss()(y_pred, y).item()
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(y.view_as(pred)).sum().item()

accuracy = correct / len(test_loader.dataset)
if args.log:
    wandb.log({'accuracy': accuracy})
else:
    print(f'Test set: Average loss: {test_loss}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy})')

# save model
utils.save_ckpt(model_dir, 'model', classifier[0])