import nntplib
import os
import pickle
import time

import torch.nn

from src.envs.tradr_dataset_flipper_env import TradrDatasetFlipperEnv
from src.policies.policies import *
from torch.nn.utils.convert_parameters import vector_to_parameters, parameters_to_vector

class SupervisedSMFlipperEnvTrainer:
    def __init__(self, config, env):
        self.config = config

        if self.config["linear"]:
            self.policy_suffix = "lin"
        else:
            self.policy_suffix = "non_lin"

        self.env = env
        self.n_objective_evals = 0

        np.random.seed(1338)
        self.loss_mask_list = np.zeros(self.env.n_data_points)
        for i in range(0, self.env.n_data_points, 10):
            if np.random.rand() < self.config["random_mask_prob"]:
                self.loss_mask_list[i - np.random.randint(2, 6): i + np.random.randint(2, 6)] = 1
        print("Loss mask with {} % dropout".format(sum(self.loss_mask_list) / len(self.loss_mask_list)))

    def objective_bbx_dep(self, w, policy):
        self.n_objective_evals += 1
        cum_rew = 0

        policy.set_params(w)
        obs = self.env.reset()
        current_state = self.env.current_state
        while True:
            next_state = policy.decide_next_state(obs, current_state)
            obs, r, done, _ = self.env.step(next_state)
            current_state = next_state
            cum_rew += r
            if done: break
        return -cum_rew

    def train_bbx_sm_dep(self):
        policy = SM()

        print("Starting training with cma")
        t1 = time.time()

        w = policy.get_params()
        es = cma.CMAEvolutionStrategy(w, self.config["cma_std"])

        print('N_params: {}'.format(len(w)))
        objective = lambda x : self.objective_bbx(x, policy)

        it = 0
        try:
            while not es.stop():
                it += 1
                if it > self.config["bbx_iters"]:
                    break
                X = es.ask()
                es.tell(X, [objective(x) for x in X])
                es.disp()

        except KeyboardInterrupt:
            print("User interrupted process.")

        policy.set_params(es.result.xbest)

        t2 = time.time()
        print("Training time: {}".format(t2 - t1))
        print(self.config)

        path = "agents/imitation_sm.pkl"
        pickle.dump(es.result.xbest, open(path, "wb"))

        return es.result.fbest

    def objective_bbx(self, w, policy):
        self.n_objective_evals += 1
        cum_rew = 0

        vector_to_parameters(torch.from_numpy(w).float(), policy.parameters())

        obs = self.env.reset()
        current_state = self.env.current_state

        while True:
            # Get action from policy
            with torch.no_grad():
                # Get state 'action'
                new_state, state_distrib = policy.calculate_next_state_detached(self.env.get_current_feature_vec(), current_state)

            obs, r, done, _ = self.env.step(new_state)
            current_state = new_state
            cum_rew += r
            if done: break
        return -cum_rew

    def train_bbx_sm(self):
        policy = DSM(feat_dim=self.config["feat_vec_dim"],
                     state_transition_dict=self.env.state_transition_dict,
                     initial_state="N", linear=self.config["linear"])

        print("Starting training with cma")
        t1 = time.time()

        w = parameters_to_vector(policy.parameters()).detach().numpy()
        es = cma.CMAEvolutionStrategy(w, self.config["cma_std"])

        print('N_params: {}'.format(len(w)))
        objective = lambda x : self.objective_bbx(x, policy)

        it = 0
        try:
            while not es.stop():
                it += 1
                if it > self.config["bbx_iters"]:
                    break
                X = es.ask()
                es.tell(X, [objective(x) for x in X])
                es.disp()

        except KeyboardInterrupt:
            print("User interrupted process.")

        vector_to_parameters(torch.from_numpy(es.result.xbest).float(), policy.parameters())
        T.save(policy.state_dict(),
               "agents/{}/imitation_dsm_bbx_{}.p".format(self.config["data_dir_name"], self.policy_suffix))

        t2 = time.time()
        print("Training time: {}".format(t2 - t1))
        print(self.config)

        return es.result.fbest

    def train_differentiable_sm(self):
        policy = DSM(feat_dim=self.config["feat_vec_dim"],
                     state_transition_dict=self.env.state_transition_dict,
                     initial_state="N", linear=self.config["linear"])

        print("Starting training with dsm")

        optim = T.optim.Adam(policy.parameters(), lr=self.config["dsm_lr"], weight_decay=self.config["weight_decay"])

        try:
            for i in range(self.config["dsm_iters"]):
                loss_list = []
                state_distrib = None
                rnd_seq, labels = self.env.get_random_sequence(self.config["dsm_batchsize"])

                for feat_vec, label, loss_mask in zip(rnd_seq, labels, self.loss_mask_list):
                    # Get state 'action'
                    current_state, state_distrib = policy.calculate_next_state_diff(feat_vec, state_distrib)

                    # Add label loss
                    if not loss_mask:
                        onehot_index = policy.state_list.index(label)
                        loss = -T.log(state_distrib[onehot_index])
                        loss_list.append(loss)

                loss_sum = T.stack(loss_list).mean()
                loss_sum.backward()

                optim.step()
                optim.zero_grad()

                if i % 10 == 0:
                    print("Iter: {}, trn_loss: {}".format(i, loss_sum.data))
        except KeyboardInterrupt:
            print("User interrupted training process")

        print("Finished training, saving")

        T.save(policy.state_dict(), "agents/{}_tradr/imitation_dsm_{}.p".format(self.config["data_dir_name"], self.policy_suffix))

    def train_differentiable_detached_sm(self):
        policy = DSM(feat_dim=self.config["feat_vec_dim"],
                     state_transition_dict=self.env.state_transition_dict,
                     initial_state="N", linear=self.config["linear"])

        print("Starting training with dsm detached")

        optim = T.optim.Adam(policy.parameters(), lr=self.config["dsm_lr"], weight_decay=self.config["weight_decay"])
        for i in range(self.config["dsm_d_iters"]):

            current_state = self.env.current_state
            loss_list = []

            rnd_seq, labels = self.env.get_random_sequence(self.config["dsm_batchsize"])

            for feat_vec, label in zip(rnd_seq, labels):
                # Get state 'action'
                new_state, state_distrib = policy.calculate_next_state_detached(feat_vec, current_state)

                # USE IF YOU WANT TO TRAIN ON DISTRIBUTION OF SM
                current_state = new_state

                # USE IF YOU WANT TO TRAIN ON DISTRIBUTION OF GT
                #current_state = self.env.short_to_state_name_dict[obs_dict["teleop_state"]]

                onehot_index = policy.state_list.index(label)
                if state_distrib[onehot_index] > 0:
                    loss = -T.log(state_distrib[onehot_index])
                    loss_list.append(loss)

            loss_sum = T.stack(loss_list).mean()
            optim.zero_grad()
            loss_sum.backward()
            optim.step()

            if i % 10 == 0:
                print("Iter: {}, trn_loss: {}".format(i, loss_sum.data))

        print("Finished training, saving")

        T.save(policy.state_dict(), "agents/{}_tradr/imitation_dsm_detached_{}.p".format(self.config["data_dir_name"], self.policy_suffix))

    def train_state_classification(self):
        policy = StateClassifier(feat_dim=self.config["feat_vec_dim"], state_list=self.env.state_list, linear=self.config["linear"])

        print("Starting training with state classifier")
        state_count_dict = self.env.state_count_dict
        total_states = sum([v for v in state_count_dict.values()])
        state_weights_dict = {}
        for k, v in state_count_dict.items():
            state_weights_dict[k] = v / float(total_states)

        state_p_list = [state_weights_dict[k] for k in self.env.state_list]
        state_p_inv_list = [1./p for p in state_p_list]
        state_w_list = [inv_p/sum(state_p_inv_list) for inv_p in state_p_inv_list]

        optim = T.optim.Adam(policy.parameters(), lr=self.config["sc_lr"], weight_decay=self.config["weight_decay"])
        if self.config["balance_dataset"]:
            lossfun = torch.nn.CrossEntropyLoss(weight=T.tensor(state_w_list))
        else:
            lossfun = torch.nn.CrossEntropyLoss()

        for i in range(self.config["state_classifier_iters"]):
            X, Y = self.env.get_random_batch_feature_vec(batchsize=self.config["sc_batchsize"], loss_mask_list=self.loss_mask_list)

            y_pred = policy(X)
            loss = lossfun(y_pred, Y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % 10 == 0:
                print("Iter: {}, trn_loss: {}".format(i, loss.data))

        print("Finished training, saving")

        T.save(policy.state_dict(), "agents/{}_tradr/imitation_state_classification_{}.p".format(self.config["data_dir_name"], self.policy_suffix))

    def train_reimp(self):
        policy = StateClassifier(feat_dim=13, state_list=self.env.state_list, linear=self.config["linear"])

        print("Starting training with reimp")

        optim = T.optim.Adam(policy.parameters(), lr=self.config["sc_lr"], weight_decay=self.config["weight_decay"])
        lossfun = nn.MSELoss()
        xentropy = nn.CrossEntropyLoss()
        for i in range(self.config["state_classifier_iters"]):
            X, Y = self.env.get_random_batch_feature_vec_haar(batchsize=self.config["sc_batchsize"], loss_mask_list=self.loss_mask_list)
            y_pred = policy(X)

            loss = lossfun(y_pred, F.one_hot(Y, num_classes=7).type(torch.FloatTensor))
            #loss = xentropy(y_pred, Y)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if i % 10 == 0:
                print("Iter: {}, trn_loss: {}".format(i, loss.data))

        print("Finished training, saving")

        T.save(policy.state_dict(), "agents/{}_tradr/imitation_reimp_{}.p".format(self.config["data_dir_name"], self.policy_suffix))

    def test_handcrafted_sm(self):
        policy = SM()
        policy.set_handcrafted_params()
        rew = self.objective_bbx(policy.get_params(), policy)
        print("Policy evaluation finished with rew: {}".format(-rew))

    def test_bbx_sm(self):
        policy = SM()
        policy.set_params(pickle.load(open("agents/imitation_sm.pkl", "rb")))
        rew = self.objective_bbx(policy.get_params(), policy)
        print("Policy evaluation finished with rew: {}".format(-rew))

    def test_differential_sm(self):
        policy = DSM(feat_dim=self.config["feat_vec_dim"],
                     state_transition_dict=self.env.state_transition_dict,
                     initial_state="N", linear=self.config["linear"])
        policy.load_state_dict(T.load("agents/{}_tradr/imitation_dsm_{}.p".format(self.config["data_dir_name"], self.policy_suffix)), strict=False)

        self.n_objective_evals += 1
        cum_rew = 0

        _ = self.env.reset()
        state_distrib = None
        while True:
            current_feature_vec = self.env.get_current_feature_vec()
            current_state, state_distrib = policy.calculate_next_state_diff(current_feature_vec,
                                                                            state_distrib)
            obs, r, done, _ = self.env.step(current_state, debug=True)
            cum_rew += r
            if done: break

        print("Policy evaluation finished with rew: {}".format(cum_rew))

    def test_differential_detached_sm(self):
        policy = DSM(feat_dim=self.config["feat_vec_dim"],
                     state_transition_dict=self.env.state_transition_dict,
                     initial_state="N", linear=self.config["linear"])
        policy.load_state_dict(T.load("agents/{}_tradr/imitation_dsm_detached_{}.p".format(self.config["data_dir_name"], self.policy_suffix)), strict=False)

        self.n_objective_evals += 1
        cum_rew = 0

        _ = self.env.reset()
        current_state = self.env.current_state
        while True:
            current_feature_vec = self.env.get_current_feature_vec()
            next_state, _ = policy.calculate_next_state_detached(current_feature_vec, current_state)
            obs, r, done, _ = self.env.step(next_state, debug=True)
            current_state = next_state
            cum_rew += r
            if done: break

        print("Policy evaluation finished with rew: {}".format(cum_rew))

    def test_state_classification(self):
        policy = StateClassifier(feat_dim=self.config["feat_vec_dim"], state_list=self.env.state_list, linear=self.config["linear"])
        policy.load_state_dict(T.load("agents/{}_tradr/imitation_state_classification_{}.p".format(self.config["data_dir_name"], self.policy_suffix)), strict=False)

        self.n_objective_evals += 1
        cum_rew = 0

        _ = self.env.reset()
        while True:
            current_feature_vec = self.env.get_current_feature_vec()
            current_state, state_distrib = policy.decide_next_state(current_feature_vec)
            obs, r, done, _ = self.env.step(current_state, debug=True)
            cum_rew += r
            if done: break

        print("Policy evaluation finished with rew: {}".format(cum_rew))

    def test_reimp(self):
        policy = StateClassifier(feat_dim=13, state_list=self.env.state_list, linear=self.config["linear"])
        policy.load_state_dict(T.load("agents/{}_tradr/imitation_reimp_{}.p".format(self.config["data_dir_name"], self.policy_suffix)), strict=False)

        self.n_objective_evals += 1
        cum_rew = 0

        _ = self.env.reset()
        while True:
            dict = self.env.get_current_obs_dict()
            feats = dict["haar_feats"]
            current_feature_vec = feats
            current_state, state_distrib = policy.decide_next_state(current_feature_vec)
            obs, r, done, _ = self.env.step(current_state)
            cum_rew += r
            if done: break

        print("Policy evaluation finished with rew: {}".format(cum_rew))

    def test_neutral(self):
        policy = NeutralClassifier()
        rew = self.objective_bbx(None, policy)
        print("Policy evaluation finished with rew: {}".format(-rew))

    def test_random(self):
        policy = RandomClassifier(self.env.state_list)
        rew = self.objective_bbx(None, policy)
        print("Policy evaluation finished with rew: {}".format(-rew))

def main(trainer_config, max_datasets=100):
    env = TradrDatasetFlipperEnv(max_datasets=max_datasets, dir_name=trainer_config["data_dir_name"])
    trainer = SupervisedSMFlipperEnvTrainer(trainer_config, env)

    if trainer_config["train"]:
        if trainer_config["mode"] == "bbx_sm":
            trainer.train_bbx_sm()
        elif trainer_config["mode"] == "differentiable_sm":
            trainer.train_differentiable_sm()
        elif trainer_config["mode"] == "differentiable_detached_sm":
            trainer.train_differentiable_detached_sm()
        elif trainer_config["mode"] == "state_classification":
            trainer.train_state_classification()
        elif trainer_config["mode"] == "reimp":
            trainer.train_reimp()
        else:
            print("Configured with static policy, training doesn't apply")

    if trainer_config["test"]:
        if trainer_config["mode"] == "handcrafted_sm":
            trainer.test_handcrafted_sm()
        elif trainer_config["mode"] == "bbx_sm":
            trainer.test_bbx_sm()
        elif trainer_config["mode"] == "differentiable_sm":
            trainer.test_differential_sm()
        elif trainer_config["mode"] == "differentiable_detached_sm":
            trainer.test_differential_detached_sm()
        elif trainer_config["mode"] == "state_classification":
            trainer.test_state_classification()
        elif trainer_config["mode"] == "reimp":
            trainer.test_reimp()
        elif trainer_config["mode"] == "neutral":
            trainer.test_neutral()
        elif trainer_config["mode"] == "random":
            trainer.test_random()
        else:
            raise NotImplementedError

        print("True state count: {}".format(env.state_count_dict))
        print("Predicted state count: {}".format(env.predicted_state_count_dict))

        print("True transitions count: {}".format(env.state_trans_dict))
        print("Predicted transitions count: {}".format(env.predicted_state_trans_dict))

if __name__=="__main__":
    import yaml
    with open(os.path.join(os.path.dirname(__file__), "configs/train_sm_flipper_env_supervised_tradr.yaml"), 'r') as f:
        trainer_config = yaml.load(f, Loader=yaml.FullLoader)

    if not os.path.exists("agents"):
        os.makedirs("agents")

    T.set_num_threads(1)

    # ID of this session
    trainer_config["session_ID"] = "TRN"

<<<<<<< HEAD
    # List of policies that we want to train
    #modes = ["handcrafted_sm", "bbx_sm", "differentiable_sm", "differentiable_detached_sm", "state_classification", "neutral", "random"]
    modes = ["state_classification"]

    for m in modes:
=======
    for m in trainer_config["modes"]:
>>>>>>> bf35b9f9db911efa610c8f3daa1c80d9a381fc08
        trainer_config["mode"] = m
        main(trainer_config, max_datasets=1)



