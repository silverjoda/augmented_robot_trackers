import numpy as np
import torch as T
import torch.nn as nn
import torch.functional as F
from torch.utils.tensorboard import SummaryWriter

input = T.tensor([0.3, 0.1, 0.6], requires_grad=True)
output = T.softmax(input * 2.846, dim=0)
print(output)