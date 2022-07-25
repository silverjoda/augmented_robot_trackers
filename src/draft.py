import torch as T
import torch.nn as nn
import torch.functional as F
from torch.utils.tensorboard import SummaryWriter

class MiniMLP(nn.Module):
    def __init__(self, obs_dim, out_dim):
        super(MiniMLP, self).__init__()
        tfunc1 = nn.Linear(obs_dim, out_dim)
        tfunc2 = nn.Linear(obs_dim, out_dim)
        tfuncs = [tfunc1, tfunc2]
        self.tfuncs = nn.ModuleList(tfuncs)

    def forward(self, x, idx):
        return self.tfuncs[idx](x)

print(T.cuda.is_available(), T.cuda.current_device(), T.cuda.device_count(), T.cuda.get_device_name(0))

mlp = MiniMLP(2, 2)

input0 = T.tensor([[0.2,0.1]], dtype=T.float32, requires_grad=True)
input1 = T.tensor([[-0.6,0.4]], dtype=T.float32, requires_grad=True)

idx0 = T.tensor(0)
out0 = mlp(input0, idx0)
probs_0 = T.softmax(out0, 1)

idx1 = T.argmax(probs_0)
out1 = probs_0[0][idx1] * mlp(input1, idx1)

lossfun = T.nn.CrossEntropyLoss()
loss = lossfun(out1, T.tensor([0]))
loss.backward()



# print(out0)
# print(idx0)
# print(out1)
# print(idx1)
# print(loss)
# print("INPUT 0: ", input0.grad)
# print("INPUT 1: ", input1.grad)
# print("WEIGHT 0: ", mlp.tfuncs[0].weight.grad)
# print("WEIGHT 1: ", mlp.tfuncs[1].weight.grad)

#<include>
#      <static>1</static>
#      <uri>https://fuel.ignitionrobotics.org/1.0/openrobotics/models/Euro pallet</uri>

