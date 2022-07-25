def objective_optuna(self, trial):
    w = [0] * self.env.get_num_params()

    # Set parameter ranges
    w[0] = trial.suggest_uniform('w_0', 0.1, 0.3)
    w[1] = trial.suggest_uniform('w_1', 0.1, 0.3)
    w[2] = trial.suggest_uniform('w_2', 0.3, 0.5)
    w[3] = trial.suggest_uniform('w_3', 0.1, 0.3)
    w[4] = trial.suggest_uniform('w_4', 0.3, 0.5)
    w[5] = trial.suggest_uniform('w_5', 0.03, 0.05)
    w[6] = trial.suggest_uniform('w_6', 0.05, 0.2)
    w[7] = trial.suggest_uniform('w_7', 0.03, 0.12)
    w[8] = trial.suggest_uniform('w_8', 0.08, 0.15)
    w[9] = trial.suggest_uniform('w_9', 0.03, 0.09)
    w[10] = trial.suggest_uniform('w_10', 0.09, 0.15)
    w[11] = trial.suggest_uniform('w_11', 0.12, 0.18)
    w[12] = trial.suggest_uniform('w_12', 0.03, 0.09)
    w[13] = trial.suggest_uniform('w_13', 0.09, 0.15)
    w[14] = trial.suggest_uniform('w_14', 0.02, 0.09)
    w[15] = trial.suggest_uniform('w_15', 0.09, 0.15)
    w[16] = trial.suggest_uniform('w_16', 0.1, 0.7)

    # "FLIPPERS_NEUTRAL": [-2, -2, 2, 2]
    # "FLIPPERS_ROUGH_TERRAIN": [-0.7, -0.7, 0.2, 0.2]
    # "FLIPPERS_ASCENDING_FRONT": [-0.4, -0.4, 0.0, 0.0]
    # "FLIPPERS_ASCENDING_REAR": [0, 0, -0.7, -0.7]
    # "FLIPPERS_DESCENDING_FRONT": [0.25, 0.25, -0.4, -0.4]
    # "FLIPPERS_DESCENDING_REAR": [0.5, 0.5, -0.9, -0.9]
    # "FLIPPERS_UP_STAIRS": [0, 0, 0.05, 0.05]
    # "FLIPPERS_DOWN_STAIRS": [0, 0, 0.05, 0.05]

    cum_rew = 0
    self.env.set_policy_parameters(w)
    obs = self.env.reset()
    step_ctr = 0
    while True:
        obs, r, done, _ = self.env.step(obs)
        cum_rew += r
        if done: break
        step_ctr += 1

    return cum_rew


def train_optuna(self):
    import optuna
    import sqlite3
    import sqlalchemy.exc

    print("Starting training with optuna")
    N_trials = self.config["n_trials"]

    t1 = time.time()
    study = optuna.create_study(direction='maximize')

    while True:
        try:
            study.optimize(lambda x: self.objective_optuna(x), n_trials=N_trials, show_progress_bar=True)
            break
        except (sqlite3.OperationalError, sqlalchemy.exc.InvalidRequestError):
            print("Optimize failed, restarting")

    t2 = time.time()
    print("Time taken: ", t2 - t1)
    print("Best params: ", study.best_params, " Best value: ", study.best_value)
    return study.best_value
