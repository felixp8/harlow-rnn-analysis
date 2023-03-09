import os
import glob

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

### Configure files to load

trajectory_format = "/home/fpei2/learning/harlow_analysis/runs/harlowdelay_gru256/trajectories/epoch4999_sample*.npz"

### Load all trajectories

filepaths = sorted(glob.glob(trajectory_format))

trajectories = []
for filepath in filepaths:
    data = np.load(filepath)
    trajectories.append((data['obs'], data['actions'], data['rewards'], data['states'], data['obj1'], data['obj2']))

### Extract trial variables
# Choice (left vs. right)
# Object (2 obj, 5-d each)
# Value (2 obj)

obj_dim = 5
num_var = 1 + 2*obj_dim + 2

trial_var = np.full((6, num_var, len(trajectories)), np.nan)
for i, traj in enumerate(trajectories):
    obj1, obj2 = traj[4], traj[5]

    act_idx = np.nonzero(traj[1])[0]
    assert len(act_idx) == 6
    assert act_idx[-1] + 2 == len(traj[1])

    trial_start = 0
    for j, trial_end in enumerate(act_idx + 1):
        trial_traj = [t[trial_start:trial_end] for t in traj[:4]]

        choice = trial_traj[1].max()
        assert choice in [1, 2]
        trial_var[j, 0, i] = int(round((choice - 1.5) * 2)) # -1 for left, 1 for right

        obj_left = trial_traj[0][:, 1:(1+obj_dim)]
        obj1_ldist = np.linalg.norm(obj_left - obj1[None, :], axis=1).min()
        obj2_ldist = np.linalg.norm(obj_left - obj2[None, :], axis=1).min()

        obj_right = trial_traj[0][:, (1+obj_dim):(1+2*obj_dim)]
        obj1_rdist = np.linalg.norm(obj_right - obj1[None, :], axis=1).min()
        obj2_rdist = np.linalg.norm(obj_right - obj2[None, :], axis=1).min()

        obj_left_true = obj1 if (obj1_ldist < obj2_ldist) else obj2
        obj_right_true = obj1 if (obj1_rdist < obj2_rdist) else obj2
        assert not np.all(obj_left_true == obj_right_true)

        trial_var[j, 1:(1+obj_dim), i] = obj_left_true
        trial_var[j, (1+obj_dim):(1+2*obj_dim), i] = obj_right_true

        if j == 0:
            trial_var[j, (1+2*obj_dim):, i] = 0.5 # both unknown value
        else:
            reward = 1 if np.any(trial_traj[2] >= 0.8) else 0 # in case of dumb float stuff
            if choice == 1 and reward == 1: # there's a more efficient way to code this
                trial_var[j, (1+2*obj_dim), i] = 1.
                trial_var[j, (2+2*obj_dim), i] = -1.
            elif choice == 2 and reward == 1:
                trial_var[j, (2+2*obj_dim), i] = 1.
                trial_var[j, (1+2*obj_dim), i] = -1.
            elif choice == 1 and reward != 1:
                trial_var[j, (2+2*obj_dim), i] = 1.
                trial_var[j, (1+2*obj_dim), i] = -1.
            elif choice == 2 and reward != 1:
                trial_var[j, (1+2*obj_dim), i] = 1.
                trial_var[j, (2+2*obj_dim), i] = -1.
            else:
                raise ValueError

### Extract states and traj lens

