import os
import pickle
import numpy as np
import torch as T
from src.policies.policies import TrackerNN, TrackerNNDual
T.set_num_threads(1)

class MarvImitationNNTrainer:
    def __init__(self, config):
        self.config = config

        # Load all datasets
        data_dict_list = []
        dataset_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "envs/supervised_dataset/sm")
        for i in range(100):
            file_path = os.path.join(dataset_dir, "dataset_{}.pkl".format(i))
            if os.path.exists(file_path):
                file_path = os.path.join(dataset_dir, "dataset_{}.pkl".format(i))
                data_dict_list.extend(pickle.load(open(file_path, "rb")))

        #data_dict_list = pickle.load(open(file_path, "rb"), encoding='latin1') # Keep for now
        if self.config["dual_output"]:
            policy_nn = TrackerNNDual(config, obs_dim=18, act_dim=4)
            self.train_dual(data_dict_list, policy_nn)
        else:
            policy_nn = TrackerNN(config, obs_dim=18, act_dim=2)
            self.train(data_dict_list, policy_nn)

    def calc_loss_dual(self, A, B, L):
        l_1 = T.sum(T.pow(A - L, 2), dim=1, keepdim=True)
        l_2 = T.sum(T.pow(B - L, 2), dim=1, keepdim=True)
        cat_diffs = T.cat((l_1, l_2), dim=1)
        loss = T.mean(T.min(cat_diffs, dim=1).values)
        return loss

    def train(self, dataset_dict_list, policy_nn):
        x_trn, y_trn, x_tst, y_tst = self.make_dataset(dataset_dict_list)
        n_trn = len(x_trn)

        lossfun = T.nn.MSELoss()

        opt = T.optim.Adam(policy_nn.parameters(), lr=0.0005)

        for i in range(self.config["iters"]):
            indeces = np.random.choice(range(n_trn), self.config["batchsize"], replace=False)

            x_batch = x_trn[indeces]
            y_batch = y_trn[indeces]

            loss = lossfun(policy_nn(x_batch), y_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 100 == 0:
                with T.no_grad():
                    tst_loss = lossfun(policy_nn(x_tst), y_tst)
                    print("Iter: {}, trn_loss: {}, tst_loss: {}".format(i, loss.data, tst_loss.data))

        print("Finished training, saving")
        T.save(policy_nn.state_dict(), "agents/imitation_nn.pth")

    def train_dual(self, dataset_dict_list, policy_nn):
        x_trn, y_trn, x_tst, y_tst = self.make_dataset(dataset_dict_list)

        n_trn = len(x_trn)

        opt = T.optim.Adam(policy_nn.parameters(), lr=0.0005)

        for i in range(self.config["iters"]):
            indeces = np.random.choice(range(n_trn), self.config["batchsize"], replace=False)

            x_batch = x_trn[indeces]
            y_batch = y_trn[indeces]

            y_A, y_B = policy_nn(x_batch)

            loss = self.calc_loss_dual(y_A, y_B, y_batch)

            #loss = mse(y_, y_batch)
            opt.zero_grad()
            loss.backward()
            opt.step()

            if i % 100 == 0:
                with T.no_grad():
                    y_A, y_B = policy_nn(x_tst)
                    tst_loss = self.calc_loss_dual(y_A, y_B, y_tst)
                    print("Iter: {}, trn_loss: {}, tst_loss: {}".format(i, loss.data, tst_loss.data))

        print("Finished training, saving")
        T.save(policy_nn.state_dict(), "agents/imitation_nn.pth")

    def make_dataset(self, dataset_dict_list):
        X = []
        Y = []

        for d in dataset_dict_list:
            X.append(d["pc_feat_vec"])

            flippers_label = [d["front_left_flipper"],
                              d["front_right_flipper"],
                              d["rear_left_flipper"],
                              d["rear_right_flipper"]]

            Y.append(flippers_label)

        X = T.tensor(X)
        Y = T.tensor(Y)
        n_total = len(X)
        n_thresh = int(0.9 * n_total)

        indeces = np.arange(n_total)
        np.random.shuffle(indeces)

        X_trn = X[indeces[:n_thresh]]
        Y_trn = Y[indeces[:n_thresh]]

        X_tst = X[indeces[n_thresh:]]
        Y_tst = Y[indeces[n_thresh:]]

        return X_trn, Y_trn, X_tst, Y_tst

if __name__=="__main__":
    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/train_nn_imitation.yaml"), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # ID of this session
    config["session_ID"] = "TRN"
    trainer = MarvImitationNNTrainer(config)