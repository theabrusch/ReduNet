import argparse
import os

import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.trainer import Trainer


from redunet import *
import evaluate
import load as L
import functional as F
import utils
import plot



parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, required=False, help='choice of dataset', default='mnist2d')
parser.add_argument('--arch', type=str, required=False, help='choice of architecture', default='lift2d_channels35_layers5')
parser.add_argument('--samples', type=int, required=False, help="number of samples per update", default=1000)
parser.add_argument('--tail', type=str, default='', help='extra information to add to folder name')
parser.add_argument('--save_dir', type=str, default='./saved_models/', help='base directory for saving.')
parser.add_argument('--data_dir', type=str, default='./data/', help='base directory for saving.')
args = parser.parse_args()

## CUDA
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.backends.mps.is_available()
## Model Directory
model_dir = os.path.join(args.save_dir, 
                         'forward',
                         f'{args.data}+{args.arch}',
                         f'samples{args.samples}'
                         f'{args.tail}')
os.makedirs(model_dir, exist_ok=True)
utils.save_params(model_dir, vars(args))
print(model_dir)

## Data
trainset, testset, num_classes = L.load_dataset(args.data, data_dir=args.data_dir)
X_train, y_train = F.get_samples(trainset, args.samples)
X_train, y_train = X_train.to(device), y_train.to(device)
train_dset = torch.utils.data.TensorDataset(X_train, y_train)
train_loader = torch.utils.data.DataLoader(train_dset, batch_size=120, shuffle=True, num_workers = 10)

test_loader = torch.utils.data.DataLoader(testset, batch_size=120, num_workers = 10)


model = nn.Sequential(
    nn.Conv2d(1, 16, 5, 1),
    nn.ReLU(),
    nn.Dropout2d(0.2),
    nn.MaxPool2d(2, 2),
    nn.Conv2d(16, 32, 5, 1),
    nn.ReLU(),
    nn.Dropout2d(0.2),
    nn.MaxPool2d(2, 2),
    nn.Flatten(),
    nn.Linear(512, 10),
    nn.Dropout(0.2),
    nn.Linear(10, 5),
    nn.Dropout(0.2),
    nn.Linear(5, 10),
    nn.Softmax(dim=1)
)

# define the LightningModule
class LitAutoEncoder(pl.LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x, y = batch
        z = self.encoder(x)
        loss = nn.functional.cross_entropy(z, y)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        z = self.encoder(x)
        loss = nn.functional.cross_entropy(z, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-5)
        return optimizer
    
    def test_step(self, batch, batch_idx):
        # this is the test loop
        x, y = batch
        z = self.encoder(x)
        test_loss = nn.functional.cross_entropy(z, y)
        # compute test accuracy
        _, y_hat = torch.max(z, dim=1)
        test_acc = torch.sum(y_hat == y).item() / (len(y) * 1.0)
        # log the results
        self.log("test_acc", test_acc, on_step=False, on_epoch=True)
        self.log("test_loss", test_loss, on_step=False, on_epoch=True)

net = LitAutoEncoder(model)
## Architecture
net = net.to(device)

trainer = Trainer(callbacks=[EarlyStopping(monitor="val_loss", mode="min")], max_epochs=50, accelerator='cpu')
trainer.fit(net, train_dataloaders=train_loader, val_dataloaders=test_loader)
trainer.save_checkpoint("saved_models/backward/mnist2d/mnist.pt", weights_only=True)
trainer.test(net, test_loader)


