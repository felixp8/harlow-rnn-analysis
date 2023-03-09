import os
import glob
import itertools

from scipy.linalg import LinAlgWarning
import warnings
warnings.simplefilter("ignore", LinAlgWarning)

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

### Configure files to load

trajectory_format = "/home/fpei2/learning/harlow_analysis/runs/harlowdelay_gru256/trajectories/epoch4999_sample*.npz"

### Load all trajectories

filepaths = sorted(glob.glob(trajectory_format))

trajectories = []
for filepath in filepaths:
    data = np.load(filepath)
    trajectories.append((data['obs'], data['actions'], data['rewards'], data['states'], data['obj1'], data['obj2']))

### Things to average over

# Everything except choice
# Everything except feedback reward
# Then what?

obj_dim = 5

num_trials = np.sum([t[0].shape[0] * 6 for t in trajectories])
num_feat = (
    1 + # choice, 2 possible val
    1 + # reward, 2 possible val
    1 + # past reward, 2 possible val
    1 + # stimulus, 2*len(trajectories) possible val - 2 obj config per set
    1 # unknown/known soln state, 1 + len(trajectories) possible val - pre-feedback and (num obj set) post-feedback
)
feat_names = ['c', 'r', 'p', 's', 'k']

trial_info = np.empty((num_trials, num_feat))
trial_state_list = []

for i, traj_objset in enumerate(trajectories):
    obs, actions, rewards, states, obj1, obj2 = traj_objset

    num_objset_episodes = obs.shape[0]

    for j in range(num_objset_episodes):
        assert obs[j].shape[0] == 37

        past_reward = 0
        for k in range(6):
            trial_obs = obs[j][k*6:((k+1)*6)+1] # include overlap of 1 with next trial - receiving reward during next fixation
            trial_acts = actions[j][k*6:((k+1)*6)+1]
            trial_rews = rewards[j][k*6:((k+1)*6)+1]
            trial_states = states[j][k*6:((k+1)*6)+1]

            choice = np.max(trial_acts)
            assert choice in [1, 2]
            assert np.sum(trial_acts == choice) == 1
            choice -= 1

            reward = int(np.any(trial_rews == 1.))

            obj_left_obs = trial_obs[:, 1:(1+obj_dim)]
            obj1_ldist = np.linalg.norm(obj_left_obs - obj1[None, :], axis=1).min()
            obj2_ldist = np.linalg.norm(obj_left_obs - obj2[None, :], axis=1).min()
            obj_left = 0 if obj1_ldist < obj2_ldist else 1
            
            obj_right_obs = trial_obs[:, (1+obj_dim):(1+2*obj_dim)] # unnecessary sanity check
            obj1_rdist = np.linalg.norm(obj_right_obs - obj1[None, :], axis=1).min()
            obj2_rdist = np.linalg.norm(obj_right_obs - obj2[None, :], axis=1).min()
            obj_right = 0 if obj1_rdist < obj2_rdist else 1
            if obj_left == obj_right:
                import pdb; pdb.set_trace()

            stim_cond = obj_left + 2 * i

            soln_state = (i+1) if k > 0 else 0

            trial_info[(i*num_objset_episodes + j)*6 + k, :] = \
                np.array([choice, reward, past_reward, stim_cond, soln_state])
            trial_state_list.append(trial_states)
            
            past_reward = reward

all_states = np.stack(trial_state_list)

### Decomposition

decomp = {}

full_mean = np.tile(
    all_states.mean(axis=(0,1), keepdims=True), 
    (all_states.shape[0], all_states.shape[1], 1))

decomp['mean'] = full_mean

all_states_c = all_states - full_mean

feat_name_to_idx = {feat_name: i for (i, feat_name) in enumerate(feat_names)}

for feat_order in range(0, len(feat_names) + 1):
    feat_names_all = ['t'] + feat_names
    feat_combs = sorted(itertools.combinations(feat_names_all, feat_order + 1))
    subtract_list = []

    for feat_comb in feat_combs:
        if 't' in feat_comb:
            meandim = (0,)
        else:
            meandim = (0,1)
        
        feat_mean = np.full(all_states.shape, np.nan)

        feat_idxs = [feat_name_to_idx.get(fn) for fn in feat_comb if (fn != 't')]
        if len(feat_idxs) == 0:
            masks = [np.full(all_states.shape[0], True, dtype=bool)]
        else:
            masks = []
            combs = np.unique(trial_info[:, feat_idxs], axis=0)
            for comb in combs:
                masks.append(np.all(trial_info[:, feat_idxs] == comb, axis=1))
        
        for mask in masks:
            cond_trials = all_states_c[mask, :, :]
            cond_trials_mean = cond_trials.mean(axis=meandim, keepdims=True)
            feat_mean[mask] = cond_trials_mean

        comb_name = ''.join(fn for fn in feat_comb)
        decomp[comb_name] = feat_mean
        subtract_list.append(feat_mean)
    
    for arr in subtract_list:
        all_states_c -= arr

# import pdb; pdb.set_trace()

regress_feat = { # only look at first-order terms
    't': ['t'],
    'tc': ['c', 'tc'],
    'tk': ['k', 'tk'],
    'tp': ['p', 'tp'],
    'tr': ['r', 'tr'],
    'ts': ['s', 'ts'],
}

subspace_traj = {}

for feat_name, feat_list in regress_feat.items():
    subspace_traj[feat_name] = {}

    target = np.sum([decomp[fn] for fn in feat_list], axis=0)
    states_c = all_states - full_mean

    target = target.reshape(-1, target.shape[-1])
    states_c = states_c.reshape(-1, states_c.shape[-1])

    gscv = GridSearchCV(Ridge(fit_intercept=False), {'alpha': np.logspace(-4, 2, 7)})
    gscv.fit(states_c, target)
    # weights = gscv.best_estimator_.coef_

    target_est = gscv.predict(states_c)
    pca = PCA()
    target_pca = pca.fit_transform(target_est)
    keepdims = np.nonzero(np.cumsum(pca.explained_variance_ratio_) >= 0.9)[0][0]
    keepdims = 2 if keepdims < 2 else keepdims
    # target_pca[:, keepdims:] = 0.

    target_pca = target_pca[:, :keepdims]
    pca_axes = pca.components_.T[:, :keepdims]
    pca_axes_ev = pca.explained_variance_ratio_[:keepdims]

    target_pca = target_pca.reshape(all_states.shape[0], all_states.shape[1], keepdims)

    subspace_traj[feat_name]['pca'] = target_pca
    subspace_traj[feat_name]['pca_axes'] = pca_axes
    subspace_traj[feat_name]['pca_ev'] = pca_axes_ev

# import pdb; pdb.set_trace()

plot_feat = ['t', 'tc', 'tk', 'tp', 'tr', 'ts']

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

for feat in plot_feat:

    pca_traj = subspace_traj[feat]['pca']

    fig = plt.figure()
    if pca_traj.shape[-1] > 2:
        ax = plt.axes(projection='3d')
    else:
        ax = plt.axes()

    group_cols = {
        'tc': 0,
        'tr': 1,
        'tp': 2,
        'ts': 3,
        'tk': 4,
    }

    if feat in group_cols:
        group_idx = group_cols.get(feat)
        uniques = np.unique(trial_info[:, group_idx])
        for i, val in enumerate(uniques):
            mask = (trial_info[:, group_idx] == val)
            cond_trials = pca_traj[mask]

            label = f'{feat}={int(val)}'

            for j, trial in enumerate(cond_trials):
                if trial.shape[-1] > 2:
                    ax.plot(trial[:, 0], trial[:, 1], trial[:, 2], color=colors[i], linewidth=0.3, label=(label if j == 0 else None))
                else:
                    ax.plot(trial[:, 0], trial[:, 1], color=colors[i], linewidth=0.3, label=(label if j == 0 else None))
    
    else:
        for i, trial in enumerate(pca_traj):
            if trial.shape[-1] > 2:
                ax.plot(trial[:, 0], trial[:, 1], trial[:, 2], linewidth=0.3)
            else:
                ax.plot(trial[:, 0], trial[:, 1], linewidth=0.3)

    plt.legend()
    plt.savefig(f'{feat}_pca.png')
    plt.close()
            
                

