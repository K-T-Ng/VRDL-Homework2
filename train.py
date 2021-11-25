import os

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader

from function.Trainer import Trainer

if __name__ == '__main__':
    T = Trainer()
    T.train()
