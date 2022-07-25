import os
import pickle
import numpy as np
import torch as T

class TradrDatasetFlipperEnv:
    def __init__(self, max_datasets=100, dir_name="sm"):

        self.state_list = ["NEUTRAL",
                           "ASCENDING_FRONT",
                           "UP_STAIRS",
                           "ASCENDING_REAR",
                           "DESCENDING_FRONT",
                           "DOWN_STAIRS",
                           "DESCENDING_REAR"]

        self.flipper_pos_dict = {"NEUTRAL": [-2, -2, 1.5, 1.5],
                                 "ASCENDING_FRONT": [-0.4, -0.4, 0.0, 0.0],
                                 "UP_STAIRS": [0.1, 0.1, -0.1, -0.1],
                                 "ASCENDING_REAR": [0.1, 0.1, -0.6, -0.6],
                                 "DESCENDING_FRONT": [0.35, 0.35, -0.7, -0.7],
                                 "DOWN_STAIRS": [0, 0, 0.05, 0.05],
                                 "DESCENDING_REAR": [-0.3, -0.3, 0.4, 0.4]}

        self.state_to_short_name_dict = {"NEUTRAL" : "N",
                                         "ASCENDING_FRONT" : "AF",
                                         "UP_STAIRS" : "US",
                                         "ASCENDING_REAR" : "AR",
                                         "DESCENDING_FRONT" : "DF",
                                         "DOWN_STAIRS" : "DS",
                                         "DESCENDING_REAR" : "DR"}

        self.state_transition_dict = {"N": ["N", "AF", "DF"],
                                 "AF": ["AF", "N", "US", "AR"],
                                 "US": ["US", "AR"],
                                 "AR": ["AR", "N"],
                                 "DF": ["DF", "N", "DS", "DR"],
                                 "DS": ["DS", "DR"],
                                 "DR": ["DR", "N"]}

        self.short_to_state_name_dict = {v: k for k, v in self.state_to_short_name_dict.items()}

        # Load all datasets
        self.data_dict_list = []
        dataset_dir = os.path.join(os.path.dirname(__file__), "supervised_dataset/tradr/{}".format(dir_name))
        for i in range(max_datasets):
            file_path = os.path.join(dataset_dir, "dataset_{}.pkl".format(i))
            if os.path.exists(file_path):
                #self.data_dict_list.extend(pickle.load(open(file_path, "rb"), encoding='latin1'))
                self.data_dict_list.extend(pickle.load(open(file_path, "rb")))

        self.n_data_points = len(self.data_dict_list)
        assert self.n_data_points > 100
        print("Loaded dataset with {} points".format(self.n_data_points))
        self.current_dataset_idx = 0

        self.current_state = "NEUTRAL"

        self.neutral_indeces = []
        for i in range(self.n_data_points - 100):
            if self.data_dict_list[i]["teleop_state"] == "N":
                self.neutral_indeces.append(i)

        self.state_count_dict = {k : 0 for k in self.state_to_short_name_dict.keys()}
        self.calculate_step_counts()

    def get_current_obs_dict(self):
        return self.data_dict_list[self.current_dataset_idx]

    def get_current_feature_vec(self):
        obs_dict = self.get_current_obs_dict()

        # Intrinsics
        pitch = obs_dict["euler"][1]
        # flat_pitch_dev = abs(pitch) < 0.2
        # small_pitch = pitch < -0.2
        # large_pitch = pitch < -0.4
        # small_dip = pitch > 0.2
        # large_dip = pitch > 0.4
        #
        # # Extrinsics general
        # flat_ground = obs_dict["frontal_low_feat"][1] > -0.04 and obs_dict["frontal_low_feat"][2] < 0.04 and \
        #               obs_dict["rear_low_feat"][1] > -0.04 and obs_dict["rear_low_feat"][2] < 0.04
        #
        # # Extrinsics front
        # untraversable_elevation = obs_dict["frontal_mid_feat"][3] > 0.1
        # small_frontal_elevation = obs_dict["frontal_low_feat"][2] > 0.06
        # large_frontal_elevation = obs_dict["frontal_low_feat"][2] > 0.12
        # small_frontal_lowering = obs_dict["frontal_low_feat"][1] < -0.06
        # large_frontal_lowering = obs_dict["frontal_low_feat"][1] < -0.12
        # low_frontal_point_presence = obs_dict["frontal_low_feat"][3] < 0.15
        #
        # # Extrinsics rear
        # small_rear_elevation = obs_dict["rear_low_feat"][2] > 0.06
        # large_rear_elevation = obs_dict["rear_low_feat"][2] > 0.12
        # small_rear_lowering = obs_dict["rear_low_feat"][1] < -0.06
        # large_rear_lowering = obs_dict["rear_low_feat"][1] < -0.12
        # not_rear_lowering = obs_dict["rear_low_feat"][2] > -0.03
        # low_rear_point_presence = obs_dict["rear_low_feat"][3] < 0.5

        feature_vec = [pitch] + list(obs_dict["frontal_low_feat"]) + list(obs_dict["rear_low_feat"])

        #feature_vec = [pitch, flat_pitch_dev, small_pitch, large_pitch, small_dip, large_dip, flat_ground,
        #               untraversable_elevation, small_frontal_elevation, large_frontal_elevation,
        #               small_frontal_lowering, large_frontal_lowering, low_frontal_point_presence,
        #               small_rear_elevation, large_rear_elevation, small_rear_lowering, large_rear_lowering,
        #               not_rear_lowering, low_rear_point_presence]  # 19

        return feature_vec

    def get_random_batch_feature_vec(self, batchsize, loss_mask_list=None):
        rnd_indeces = np.random.choice(np.where(loss_mask_list == 0)[0], batchsize, replace=False)
        feature_list = []
        label_list = []
        for ri in rnd_indeces:
            o = self.data_dict_list[ri]
            feature_list.append([o["euler"][1]] + list(o["frontal_low_feat"]) + list(o["rear_low_feat"]))
            label_list.append(self.state_list.index(self.short_to_state_name_dict[o["teleop_state"]]))
        feature_arr = T.tensor(feature_list)
        label_arr = T.tensor(label_list)

        return feature_arr, label_arr

    def get_random_sequence(self, max_length=200):
        #start_idx = np.random.choice(self.neutral_indeces, 1)
        #end_idx = np.minimum(self.n_data_points, start_idx + max_length - 1)
        start_idx = 0
        end_idx = self.n_data_points

        feature_list = []
        label_list = []
        for ri in range(start_idx, end_idx):
            o = self.data_dict_list[ri]
            feature_list.append([o["euler"][1]] + list(o["frontal_low_feat"]) + list(o["rear_low_feat"]))
            label_list.append(o["teleop_state"])
        return feature_list, label_list

    def calculate_step_counts(self):
        for i in range(len(self.data_dict_list)):
            self.state_count_dict[self.short_to_state_name_dict[self.data_dict_list[i]["teleop_state"]]] += 1

    def step(self, new_state, debug=False):
        self.current_dataset_idx += 1
        state_label = self.get_current_obs_dict()["teleop_state"]

        r = int(state_label == self.state_to_short_name_dict[new_state])
        done = self.current_dataset_idx == (self.n_data_points - 1)

        if debug:
            self.state_count_dict[self.short_to_state_name_dict[state_label]] += 1
            self.predicted_state_count_dict[new_state] += 1

        #if (self.current_state, new_state) in self.predicted_state_trans_dict:
        #    self.predicted_state_trans_dict[(self.current_state, new_state)] += 1
        #else:
        #    self.predicted_state_trans_dict[(self.current_state, new_state)] = 1

        if self.current_dataset_idx > 0:
            prev_state_label = self.data_dict_list[self.current_dataset_idx - 1]["teleop_state"]
        else:
            prev_state_label = self.data_dict_list[self.current_dataset_idx]["teleop_state"]

        #if (prev_state_label, state_label) in self.state_trans_dict:
        #    self.state_trans_dict[(prev_state_label, state_label)] += 1
        #else:
        #    self.state_trans_dict[(prev_state_label, state_label)] = 1

        self.current_state = new_state

        return self.get_current_obs_dict(), r, done, {}

    def reset(self):
        self.current_dataset_idx = 0
        self.state_count_dict = {k : 0 for k in self.state_to_short_name_dict.keys()}
        self.predicted_state_count_dict = {k : 0 for k in self.state_to_short_name_dict.keys()}
        self.state_trans_dict = {}
        self.predicted_state_trans_dict = {}
        return self.get_current_obs_dict()

if __name__=="__main__":
    TradrDatasetFlipperEnv(max_datasets=1, dir_name="sm_stairs") # HERE ITS MAX 1 DATASET FOR DEBUGGING PURPOSES, DONT CHANGE. CHANGE IT IN train_sm_flip....



