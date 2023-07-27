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
parser.add_argument('--data', type=str, required=False, help='choice of dataset', default='mnist2d')
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

## Model Directory
model_dir = os.path.join(args.save_dir, 
                         'backward',
                         f'{args.data}+ch{args.channels}+l{args.layers}',
                         f'samples{args.samples}'
                         f'{args.tail}')
data_dir = os.path.join(args.data_dir,
                        f'trainsamples{args.trainsamples}'
                        f'_testsamples{args.testsamples}')

os.makedirs(model_dir, exist_ok=True)
params = utils.load_params(args.model_dir)

## Data
trainset, testset, num_classes = L.load_dataset(params['data'], data_dir=params['data_dir'])
X_train, y_train = F.get_samples(trainset, args.trainsamples)
X_test, y_test = F.get_samples(testset, args.testsamples)
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

## Architecture
net = L.load_architecture(params['data'], params['channels'], params['layers'])
net = utils.load_ckpt(args.model_dir, 'model', net)
net.to(device)

# extract representations
with torch.no_grad():
    print('train')
    Z_train = net.batch_forward(X_train, batch_size=args.batch_size, loss=args.loss, device=device, return_representations=args.save_representations)
    if args.save_representations:
        Z_train, representations = Z_train
        utils.save_representations(data_dir, 'train', representations.numpy(), labels=y_train.cpu().numpy())
    X_train, y_train, Z_train = F.to_cpu(X_train, y_train, Z_train)

    print('test')
    Z_test = net.batch_forward(X_test, batch_size=args.batch_size, loss=args.loss, device=device, return_representations=args.save_representations)
    if args.save_representations:
        Z_test, representations = Z_test
        utils.save_representations(data_dir, 'test', representations.numpy(), labels=y_test.cpu().numpy())
    X_test, y_test, Z_test = F.to_cpu(X_test, y_test, Z_test)

