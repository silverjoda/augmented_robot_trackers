import torch.nn as nn
import torch.nn.functional as F
import torch as T
from torch.nn.utils import weight_norm
import numpy as np

class SLP_ES(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(SLP_ES, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.fc = T.nn.Linear(self.obs_dim, self.act_dim, bias=True)

    def forward(self, x):
        x_tensor = T.Tensor(x).unsqueeze(0)
        out = self.fc(x_tensor)
        out_conditioned = F.tanh(out) * 1
        out_np = out_conditioned.detach().numpy()[0]
        return out_np

class MLP_ES(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dim=128):
        super(MLP_ES, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hid_dim = hid_dim

        self.fc1 = T.nn.Linear(self.obs_dim, self.hid_dim, bias=True)
        self.fc2 = T.nn.Linear(self.hid_dim, self.act_dim, bias=True)

    def forward(self, x):
        x_tensor = T.Tensor(x).unsqueeze(0)
        fc1 = F.tanh(self.fc1(x_tensor))
        fc2 = F.tanh(self.fc2(fc1)) * 1
        out_np = fc2.detach().numpy()[0]
        return out_np

class MLP(nn.Module):
    def __init__(self, obs_dim, act_dim, hid_dim=128):
        super(MLP, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hid_dim = hid_dim

        self.fc1 = T.nn.Linear(self.obs_dim, self.hid_dim, bias=True)
        self.fc2 = T.nn.Linear(self.hid_dim, self.hid_dim, bias=True)
        self.fc3 = T.nn.Linear(self.hid_dim, self.act_dim, bias=True)

    def forward(self, x):
        fc1 = F.tanh(self.fc1(x))
        fc2 = F.tanh(self.fc2(fc1)) * 1
        out = self.fc3(fc2)
        return out

class TrackerNN(nn.Module):
    def __init__(self, config, obs_dim, act_dim):
        super(TrackerNN, self).__init__()
        self.config = config

        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, act_dim)

    def forward(self, x):
        f1 = F.relu(self.fc1(x))
        f2 = F.relu(self.fc2(f1))
        out = self.fc3(f2)
        out_flippers = T.repeat_interleave(out, 2, dim=1)
        return out_flippers

class TrackerNNDual(nn.Module):
    def __init__(self, config, obs_dim, act_dim):
        super(TrackerNNDual, self).__init__()
        self.config = config

        self.fc1 = nn.Linear(obs_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, act_dim)

    def forward(self, x):
        f1 = F.relu(self.fc1(x))
        f2 = F.relu(self.fc2(f1))
        out = self.fc3(f2)
        out_flippers_A = T.repeat_interleave(out[:,:2], 2, dim=1)
        out_flippers_B = T.repeat_interleave(out[:,2:4], 2, dim=1)
        return out_flippers_A, out_flippers_B

class MiniMLP(nn.Module):
    def __init__(self, obs_dim, out_dim):
        super(MiniMLP, self).__init__()
        hid_dim = 64
        self.l1 = nn.Linear(obs_dim, hid_dim)
        nn.init.xavier_normal_(self.l1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        self.l2 = nn.Linear(hid_dim, out_dim)
        nn.init.xavier_normal_(self.l2.weight)

    def forward(self, x):
        f1 = F.leaky_relu(self.l1(x))
        f2 = self.l2(f1)
        return f2

class SM:
    def __init__(self):
        self.params = [0] * 18

    def set_handcrafted_params(self):
        self.params[0] = 0.2 # flat_pitch_dev
        self.params[1] = 0.2 # small_pitch
        self.params[2] = 0.4 # large_pitch
        self.params[3] = 0.2 # small_dip
        self.params[4] = 0.4 # large_dip
        self.params[5] = 0.04 # flat_ground
        self.params[6] = 0.1 # untraversable_elevation
        self.params[7] = 0.06 # small_frontal_elevation
        self.params[8] = 0.12 # large_frontal_elevation
        self.params[9] = 0.06 # small_frontal_lowering
        self.params[10] = 0.12 # large_frontal_lowering
        self.params[11] = 0.15 # low_frontal_point_presence
        self.params[12] = 0.06 # small_rear_elevation
        self.params[13] = 0.12 # large_rear_elevation
        self.params[14] = 0.06 # small_rear_lowering
        self.params[15] = 0.12 # large_rear_lowering
        self.params[16] = 0.03 # not_rear_lowering
        self.params[17] = 0.5 # low_rear_point_presence

    def set_params(self, w):
        self.params = w

    def get_params(self):
        return self.params

    def decide_next_state(self, obs_dict, current_state):
        # Intrinsics
        pitch = obs_dict["euler"][1]
        flat_pitch_dev = abs(pitch) < self.params[0]  # 0.2
        small_pitch = pitch < -self.params[1]  # -0.2
        large_pitch = pitch < -self.params[2]  # -0.4
        small_dip = pitch > self.params[3]  # 0.2
        large_dip = pitch > self.params[4]  # 0.4

        # Extrinsics general
        flat_ground = obs_dict["frontal_low_feat"][1] > -self.params[5] and \
                      obs_dict["frontal_low_feat"][2] < self.params[5] and \
                      obs_dict["rear_low_feat"][1] > -self.params[5] and \
                      obs_dict["rear_low_feat"][2] < self.params[5]

        # Extrinsics front
        untraversable_elevation = obs_dict["frontal_mid_feat"][3] > self.params[6]  # 0.1
        small_frontal_elevation = obs_dict["frontal_low_feat"][2] > self.params[7]  # 0.06
        large_frontal_elevation = obs_dict["frontal_low_feat"][2] > self.params[8]  # 0.12
        small_frontal_lowering = obs_dict["frontal_low_feat"][1] < -self.params[9]  # -0.06
        large_frontal_lowering = obs_dict["frontal_low_feat"][1] < -self.params[10]  # -0.12
        low_frontal_point_presence = obs_dict["frontal_low_feat"][3] < self.params[11]  # 0.15

        # Extrinsics rear
        small_rear_elevation = obs_dict["rear_low_feat"][2] > self.params[12]  # -0.03
        large_rear_elevation = obs_dict["rear_low_feat"][2] > self.params[13]  # 0.12
        small_rear_lowering = obs_dict["rear_low_feat"][1] < -self.params[14]  # -0.06
        large_rear_lowering = obs_dict["rear_low_feat"][1] < -self.params[15]  # -0.12
        not_rear_lowering = obs_dict["rear_low_feat"][2] > -self.params[16]
        low_rear_point_presence = obs_dict["rear_low_feat"][3] < self.params[17]  # 0.5

        new_current_state = current_state
        if current_state == "NEUTRAL":
            if (small_frontal_elevation) and not untraversable_elevation:
                new_current_state = "ASCENDING_FRONT"
            elif low_frontal_point_presence or small_frontal_lowering:
                new_current_state = "DESCENDING_FRONT"
            elif small_pitch and not small_frontal_elevation:
                new_current_state = "ASCENDING_REAR"
        elif current_state == "ASCENDING_FRONT":
            # -> ascending_rear
            if small_pitch and not large_frontal_elevation:
                new_current_state = "ASCENDING_REAR"
            # -> up_stairs
            elif large_pitch and small_frontal_elevation:
                new_current_state = "UP_STAIRS"
            # -> neutral
            elif flat_ground and flat_pitch_dev:
                new_current_state = "NEUTRAL"
        elif current_state == "ASCENDING_REAR":
            if not_rear_lowering or flat_ground:
                new_current_state = "NEUTRAL"
        elif current_state == "DESCENDING_FRONT":
            # -> descending_rear
            if small_rear_lowering or low_rear_point_presence:
                new_current_state = "DESCENDING_REAR"
            # -> down_stairs
            elif (large_frontal_lowering or low_frontal_point_presence) and large_dip:
                new_current_state = "DOWN_STAIRS"
            # -> neutral
            elif flat_ground and flat_pitch_dev:
                new_current_state = "NEUTRAL"
        elif current_state == "DESCENDING_REAR":
            # -> neutral
            if flat_pitch_dev or not small_frontal_lowering:
                new_current_state = "NEUTRAL"
        elif current_state == "UP_STAIRS":
            # -> ascending_rear
            if low_frontal_point_presence or not large_frontal_elevation:
                new_current_state = "ASCENDING_REAR"
            elif flat_ground and flat_pitch_dev:
                new_current_state = "NEUTRAL"
        elif current_state == "DOWN_STAIRS":
            # -> descending_rear
            if not large_frontal_lowering or flat_ground:
                new_current_state = "DESCENDING_REAR"
            elif flat_ground and flat_pitch_dev:
                new_current_state = "NEUTRAL"
        else:
            raise NotImplementedError

        return new_current_state

class DSM(nn.Module):
    def __init__(self, feat_dim, state_transition_dict, initial_state="N", linear=True):
        super(DSM, self).__init__()
        self.state_to_short_name_dict = {"NEUTRAL": "N",
                                         "ASCENDING_FRONT": "AF",
                                         "UP_STAIRS": "US",
                                         "ASCENDING_REAR": "AR",
                                         "DESCENDING_FRONT": "DF",
                                         "DOWN_STAIRS": "DS",
                                         "DESCENDING_REAR": "DR"}
        self.short_to_state_name_dict = {v: k for k, v in self.state_to_short_name_dict.items()}

        self.state_list = [s for s in state_transition_dict.keys()]
        self.state_transition_dict = state_transition_dict
        self.n_states = len(self.state_list)
        self.initial_state_idx = self.state_list.index(initial_state)

        self.transition_func_dict = {}
        for k, v in state_transition_dict.items():
            if linear:
                tfunc = nn.Linear(feat_dim, len(v))
            else:
                tfunc = MiniMLP(feat_dim, len(v))
            self.transition_func_dict[k] = tfunc
        self.tfuncs = nn.ModuleList(self.transition_func_dict.values())

    def calculate_next_state_diff(self, x, state_distrib):
        if state_distrib is None:
            state_distrib = np.zeros(self.n_states)
            state_distrib[self.initial_state_idx] = 1.0
            state_distrib = T.tensor(state_distrib, requires_grad=True)

        x_T = T.tensor(x).unsqueeze(0)
        new_state_distrib = T.zeros(len(self.state_list), requires_grad=True)

        for i, s in enumerate(self.state_list):
            logits = self.transition_func_dict[s](x_T)[0]
            distrib = F.softmax(logits, 0) * (state_distrib[i].detach()) # Detach here for no gradient propagation through time
            logits_indeces = T.tensor([self.state_list.index(m) for m in self.state_transition_dict[s]])
            new_state_distrib = new_state_distrib.index_add(0, logits_indeces, distrib)

        current_state = self.short_to_state_name_dict[self.state_list[T.argmax(new_state_distrib)]]
        return current_state, new_state_distrib

    def calculate_next_state_diff_rnd(self, x, state_distrib):
        if state_distrib is None:
            state_distrib = np.zeros(self.n_states)
            state_distrib[self.initial_state_idx] = 1.0
            state_distrib = T.tensor(state_distrib, requires_grad=True)

        x_T = T.tensor(x).unsqueeze(0)
        new_state_distrib = T.zeros(len(self.state_list), requires_grad=True)

        for i, s in enumerate(self.state_list):
            logits = self.transition_func_dict[s](x_T)[0]
            distrib = F.softmax(logits) * state_distrib[i]
            logits_indeces = T.tensor([self.state_list.index(m) for m in self.state_transition_dict[s]])
            new_state_distrib = new_state_distrib.index_add(0, logits_indeces, distrib)

        current_state = self.short_to_state_name_dict[self.state_list[new_state_distrib.multinomial(num_samples=1)[0]]]
        return current_state, new_state_distrib

    def calculate_next_state_detached(self, x, s):
        s = self.state_to_short_name_dict[s]
        x_T = T.tensor(x).unsqueeze(0)
        new_state_distrib = T.zeros(len(self.state_list), requires_grad=True)

        logits = self.transition_func_dict[s](x_T)[0]
        distrib = F.softmax(logits, 0)
        logits_indeces = T.tensor([self.state_list.index(m) for m in self.state_transition_dict[s]])
        new_state_distrib = new_state_distrib.index_add(0, logits_indeces, distrib)

        current_state = self.short_to_state_name_dict[self.state_list[T.argmax(new_state_distrib)]]
        return current_state, new_state_distrib

    def forward(self, _):
        raise NotImplementedError

class StateClassifier(nn.Module):
    def __init__(self, feat_dim, state_list, linear=True):
        super(StateClassifier, self).__init__()
        self.state_list = state_list

        if linear:
            self.tfunc = nn.Linear(feat_dim, len(self.state_list))
        else:
            self.tfunc = MiniMLP(feat_dim, len(self.state_list))

    def decide_next_state(self, x):
        x_T = T.tensor(x).unsqueeze(0)
        out = F.softmax(self.tfunc(x_T)[0], dim=0)
        current_state = self.state_list[T.argmax(out)]
        return current_state, out

    def forward(self, x):
        out = self.tfunc(x)
        return out

class NeutralClassifier:
    def __init__(self):
        pass

    def set_params(self, x=None):
        pass

    def decide_next_state(self, x=None, y=None):
        return "NEUTRAL"

class RandomClassifier:
    def __init__(self, state_list):
        self.state_list = state_list
        self.n_states = len(self.state_list)

    def set_params(self, x=None):
        pass

    def decide_next_state(self, x=None, y=None):
        return np.random.choice(self.state_list, 1)[0]
