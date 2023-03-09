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
    trajectories.append((data['obs'], data['actions'], data['rewards'], data['states']))

### Extract states and traj lens

states = [traj[-1] for traj in trajectories]
sizes = [s.shape[0] for s in states]

### Perform stacked PCA

states = np.concatenate(states, axis=0)
states_scaled = StandardScaler().fit_transform(states)
states_pca = PCA().fit_transform(states_scaled)

### Split out trials and plot

states_pca = np.split(states_pca, np.cumsum(sizes)[:-1])

import pdb; pdb.set_trace()

plt.plot(states_pca[0][:, :3])

