import time
import os
from os import listdir
from os.path import join, isfile, isdir, expanduser
from tqdm import tqdm

import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import functional as F
from redunet import *


def load_architecture(data, channels, layers, seed=0):
    if data == 'mnist2d':
        from architectures.mnist.lift2d import lift2d
        return lift2d(channels=channels, layers=layers, num_classes=10, seed=seed)
    if data == 'mnist2d_2class':
        from architectures.mnist.lift2d import lift2d
        return lift2d(channels=channels, layers=layers, num_classes=2, seed=seed)
    if data == 'mnistvector':
        from architectures.mnist.flatten import flatten
        return flatten(layers=layers, num_classes=10)
    if data == 'mnistvector_2class':
        from architectures.mnist.flatten import flatten
        return flatten(layers=layers, num_classes=2)   
    raise NameError('Cannot find architecture: {}.')

def load_dataset(choice, data_dir='./data/'):
    if choice == 'mnist2d':
        from datasets.mnist import mnist2d_10class
        return mnist2d_10class(data_dir)
    if choice == 'mnist2d_2class':
        from datasets.mnist import mnist2d_2class
        return mnist2d_2class(data_dir)
    if choice =='mnistvector':
        from datasets.mnist import mnistvector_10class
        return mnistvector_10class(data_dir)
    raise NameError(f'Dataset {choice} not found.')


def load_concepts(choice, data_dir='./data/'):
    if choice == 'mnist2d':
        from datasets.mnist import mnist2d_concepts
        return mnist2d_concepts(data_dir)
    if choice =='mnistvector':
        from datasets.mnist import mnistvector_concepts
        return mnistvector_concepts(data_dir)
    raise NameError(f'Dataset {choice} not found.')