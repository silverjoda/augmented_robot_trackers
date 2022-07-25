from torch.utils.tensorboard import SummaryWriter
import numpy as np
writer = SummaryWriter('tb')
for i in range(10):
    x = np.random.random(10)
    writer.add_histogram('distribution centers', values=x, global_step=i, bins=10)
writer.close()