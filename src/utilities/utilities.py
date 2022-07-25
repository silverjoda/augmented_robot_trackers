import numpy as np

class ButtonToEventTranslator:
    def __init__(self, n_indeces=2, init_idx=0):
        self.n_indeces = n_indeces
        self.state = 0
        self.idx = init_idx % self.n_indeces

    def update(self, button_val):
        if not self.state and button_val:
            self.state = 1
            self.idx = (self.idx + 1) % self.n_indeces
            return 1, self.idx
        elif self.state and not button_val:
            self.state = 0
            return 0, self.idx
        elif self.state and button_val:
            self.state = 1
            return 0, self.idx
        else:
            self.state = 0
            return 0, self.idx

def dist_between_poses(p1, p2):
    return np.sqrt((p1.pose.position.x - p2.pose.position.x) ** 2 + (p1.pose.position.y - p2.pose.position.y) ** 2)

def dist_between_pose_and_position(p1, p2):
    return np.sqrt((p1.pose.position.x - p2.x) ** 2 + (p1.pose.position.y - p2.y) ** 2)

def merge_two_dicts(x, y):
    z = x.copy()  # start with x's keys and values
    z.update(y.copy())  # modifies z with y's keys and values & returns None
    return z

def merge_dicts(dicts):
    d = dicts[0]
    for i in range(1, len(dicts)):
        d = merge_two_dicts(d, dicts[i])
    return d

