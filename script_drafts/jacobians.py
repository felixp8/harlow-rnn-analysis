# %%

import os
import glob
import itertools
import copy
import sys
sys.path.insert(1, '/home/fpei2/learning/ttrnn/')

import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from ttrnn.trainer import A2C
from ttrnn.tasks.harlow import HarlowMinimal, Harlow1D, HarlowMinimalDelay
from neurogym.wrappers import PassAction, PassReward, Noise
from ttrnn.tasks.wrappers import DiscreteToBoxWrapper, RingToBoxWrapper, ParallelEnvs

# %%

ckpt_path = '/home/fpei2/learning/harlow_analysis/runs/harlowdelay3_gru256/epoch=19999-step=20000-v1.ckpt'

task = HarlowMinimalDelay(
    dt=100,
    obj_dim=5,
    obj_mode="kb", 
    obj_init="normal",
    orthogonalize=True,
    abort=True,
    rewards={'abort': -0.1, 'correct': 1.0, 'fail': 0.0},
    timing={'fixation': 200, 'stimulus': 400, 'delay': 200, 'decision': 200},
    num_trials_before_reset=6,
    r_tmax=-1.0,
)

std_noise = 0.1
wrappers = [
    (Noise, {'std_noise': std_noise}),
    (PassAction, {'one_hot': True}), 
    (PassReward, {}), 
    (ParallelEnvs, {'num_envs': 8}),
]

if len(wrappers) > 0:
    for wrapper, wrapper_kwargs in wrappers:
        task = wrapper(task, **wrapper_kwargs)

### Load checkpoint

pl_module = A2C.load_from_checkpoint(ckpt_path, env=task)

model = pl_module.model
hx = model.rnn.build_initial_state(1, pl_module.device, pl_module.dtype)

# %%

### Load trajectory samples

template_path = "/home/fpei2/learning/harlow_analysis/runs/harlowdelay3_gru256/trajectories/epoch19999-v1_sample_multi{}.npz"

num_objsets = 20

states_list = []
condition_list = []
obj_list = []
skipped_list = []
for i in range(num_objsets):
    trajectories = np.load(template_path.format(i))
    states_list.append(trajectories['states'])
    condition_list.append(trajectories['condition'])
    obj_list.append((trajectories['obj1'], trajectories['obj2']))
    skipped_list.append(trajectories['num_skipped'])

# %%

states = np.stack(states_list) # objset x cond x trial x time x dim
conditions = np.stack(condition_list) # objset x cond x feat
feat_names = [
    'objset', # (arbitrary) index of set of objects used as stimulus
    'reward_idx', # which object is rewarded
    'obj_left_1', # which object presented on left in first trial
    'choice', # which object was chosen in first trial,
    'obj_left_2', # which object presented on left in second trial
]

# %%

all_mean = states.mean(axis=(0,1,2,3), keepdims=True) # mean channel value
states_c = states - all_mean # centered data

time_mean = states_c.mean(axis=(0,1,2), keepdims=True)
states_residual = states_c - time_mean

condition_means = states_c.mean(axis=2)

condition_means_stacked = condition_means.reshape(-1, condition_means.shape[-1])

pca_allcond = PCA()
allcond_pca = pca_allcond.fit_transform(condition_means_stacked)

max_dim = min(allcond_pca.shape[0], condition_means.shape[-1])
condition_means_pca = allcond_pca.reshape(condition_means.shape[:-1] + (max_dim,))

# %%

obj_left_weights = model.rnn.rnn_cell.weights.weight_ih[-256:, 1:6].detach().cpu().numpy().T
obj_right_weights = model.rnn.rnn_cell.weights.weight_ih[-256:, 6:11].detach().cpu().numpy().T

obj_left_weights_pca = pca_allcond.transform(obj_left_weights)
obj_right_weights_pca = pca_allcond.transform(obj_right_weights)

obj_left_weight_norm = np.linalg.norm(obj_left_weights, axis=1)
obj_right_weight_norm = np.linalg.norm(obj_right_weights, axis=1)

weight_angle = np.arccos(
    np.diag(np.dot(obj_left_weights, obj_right_weights.T)) / 
    np.linalg.norm(obj_left_weights, axis=1) /
    np.linalg.norm(obj_right_weights, axis=1)
)

num_pcs = 5
weight_angle_pca = np.arccos(
    np.diag(np.dot(obj_left_weights_pca[:, :num_pcs], obj_right_weights_pca[:, :num_pcs].T)) / 
    np.linalg.norm(obj_left_weights_pca[:, :num_pcs], axis=1) /
    np.linalg.norm(obj_right_weights_pca[:, :num_pcs], axis=1)
)

# %%

# fit logistic regressor to weight difference to first trial behavior

# %%

obj1, obj2 = obj_list[0]

fix_state = states[0, 0, 0, 1, :]

stim_input = np.concatenate([
    np.array([1.]),
    obj1,
    obj2,
    np.array([1., 0., 0., 0.])
], axis=0)

# %%

input_jac, rec_jac = torch.autograd.functional.jacobian(
    model.rnn.rnn_cell, 
    (
        torch.from_numpy(stim_input[None, :]).to(pl_module.dtype),
        torch.from_numpy(fix_state[None, :]).to(pl_module.dtype)
    )
)

# %%

### Dynamics time

fix_state = states_list
for obj1, obj2 in obj_list:
    stim_input = np.concatenate([
        np.array([1.]),
        obj1,
        obj2,
        np.array([1., 0., 0., 0.])
    ], axis=0)

obj_grads = []

for objset in range(states.shape[0]):
    obj1, obj2 = obj_list[objset]

    obj1_left_grad = []
    obj2_left_grad = []

    for i, cond in enumerate(conditions[objset]):
        obj_left_1 = cond[2]
        obj_left_2 = cond[4]

        trial_1_stim = np.concatenate([
            np.array([1.]), 
            obj1 if obj_left_1 == 0 else obj2,
            obj2 if obj_left_1 == 0 else obj1,
            np.array([1., 0., 0., 0.]),
        ], axis=0)
        trial_2_stim = np.concatenate([
            np.array([1.]), 
            obj1 if obj_left_2 == 0 else obj2,
            obj2 if obj_left_2 == 0 else obj1,
            np.array([1., 0., 0., 0.]),
        ], axis=0)

        for trial_ix in range(states[objset][i].shape[0]):
            trial_states = states[objset][i][trial_ix]
            stim1 = trial_states[2:6]
            stim2 = trial_states[11:15]
            # delay = np.concat([trial_states[6:8], trial_states[15:17]], axis=0)

            stim1 = interp1d(np.arange(4), stim1, axis=0, kind='cubic')(np.arange(0, 3.5, 0.5))
            stim2 = interp1d(np.arange(4), stim2, axis=0, kind='cubic')(np.arange(0, 3.5, 0.5))

            for stim_state in stim1:
                next_state = model.rnn.rnn_cell(
                    torch.from_numpy(trial_1_stim[None, :]).to(pl_module.dtype),
                    torch.from_numpy(stim_state[None, :]).to(pl_module.dtype), 
                    cached=True
                ).detach().cpu().numpy()
                state_diff = next_state.squeeze() - stim_state
                if obj_left_1 == 0:
                    obj1_left_grad.append((stim_state, state_diff))
                else:
                    obj2_left_grad.append((stim_state, state_diff))
            
            for stim_state in stim2:
                next_state = model.rnn.rnn_cell(
                    torch.from_numpy(trial_2_stim[None, :]).to(pl_module.dtype),
                    torch.from_numpy(stim_state[None, :]).to(pl_module.dtype), 
                    cached=True
                ).detach().cpu().numpy()
                state_diff = next_state.squeeze() - stim_state
                if obj_left_2 == 0:
                    obj1_left_grad.append((stim_state, state_diff))
                else:
                    obj2_left_grad.append((stim_state, state_diff))
            
            # for delay_state in delay:
            #     next_state = model.rnn.rnn_cell(
            #         torch.from_numpy(delay_stim[None, :]).to(pl_module.dtype),
            #         torch.from_numpy(delay_state[None, :]).to(pl_module.dtype), 
            #         cached=True
            #     ).detach().cpu().numpy()
            #     state_diff = next_state.squeeze() - delay_state
            #     delay_grad.append((delay_state, state_diff))
    
    obj1_left_states, obj1_left_deltas = zip(*obj1_left_grad)
    obj1_left_states = np.stack(obj1_left_states)
    obj1_left_deltas = np.stack(obj1_left_deltas)

    obj2_left_states, obj2_left_deltas = zip(*obj2_left_grad)
    obj2_left_states = np.stack(obj2_left_states)
    obj2_left_deltas = np.stack(obj2_left_deltas)
    obj_grads.append((obj1_left_states, obj1_left_deltas, obj2_left_states, obj2_left_deltas))