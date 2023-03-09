# import os
# os.chdir('/home/fpei2/learning/ttrnn/')

import copy
import sys
sys.path.insert(1, '/home/fpei2/learning/ttrnn/')

import numpy as np
import torch

from ttrnn.trainer import A2C
from ttrnn.tasks.harlow import HarlowMinimal, Harlow1D, HarlowMinimalDelay
from neurogym.wrappers import PassAction, PassReward
from ttrnn.tasks.wrappers import DiscreteToBoxWrapper, RingToBoxWrapper, ParallelEnvs

ckpt_path = "/home/fpei2/learning/harlow_analysis/runs/harlowdelay_gru256/epoch=4999-step=5000.ckpt"
save_path = "/home/fpei2/learning/harlow_analysis/runs/harlowdelay_gru256/trajectories/epoch4999_sample{}.npz"

### Build env

task = HarlowMinimalDelay(
    dt=100,
    obj_dim=5,
    obj_mode="kb", 
    obj_init="normal",
    orthogonalize=True,
    abort=True,
    rewards={'abort': -0.1, 'correct': 1.0, 'fail': 0.0},
    timing={'fixation': 200, 'stimulus': 300, 'delay': 0, 'decision': 500},
    num_trials_before_reset=6,
)

wrappers = [(PassAction, {'one_hot': True}), (PassReward, {}), (ParallelEnvs, {'num_envs': 8})]

if len(wrappers) > 0:
    for wrapper, wrapper_kwargs in wrappers:
        task = wrapper(task, **wrapper_kwargs)

### Load checkpoint

pl_module = A2C.load_from_checkpoint(ckpt_path, env=task)

model = pl_module.model
env = copy.deepcopy(task.env_list[0])
env.seed(5)

np.random.seed(2)

### Sample trajectories

trajectories_per_objset = 20
num_objsets = 4
trials_per_episode = 6

num_trajectories = trajectories_per_objset * num_objsets

for o in range(num_objsets):

    i = 0
    obs, _ = env.reset(reset_obj=True)
    obj1, obj2 = env.get_objects()
    print(obj1, obj2)
    num_skipped = 0

    trajectories = []

    while i < trajectories_per_objset:
        # import pdb; pdb.set_trace()

        traj_env = copy.deepcopy(env)

        seed = np.random.randint(0, 10000) # to get different trials
        obs, _ = traj_env.reset(reset_obj=False, seed=seed)
        hx = model.rnn.build_initial_state(1, pl_module.device, pl_module.dtype)

        obs_list = [obs[None, :], obs[None, :]]
        reward_list = [0.]
        action_list = [0]
        state_list = [hx.detach().cpu().numpy(), hx.detach().cpu().numpy()]

        episode_performance = []
        episode_trial_lengths = []

        stop = False
        skip = False
        trial_count = 0
        trial_len = 1 # dumb workaround

        assert np.all(traj_env.obj1 == obj1) and np.all(traj_env.obj2 == obj2)

        while not stop:

            trial_len += 1

            action_logits, value, hx = model(
                torch.from_numpy(obs).to(device=pl_module.device, dtype=pl_module.dtype).unsqueeze(0), 
                hx=hx, 
                cached=True)
            action = action_logits.mode.item() # take highest prob action always

            obs, reward, done, trunc, info = traj_env.step(action)

            max_trials_reached = False
            if info.get('new_trial', False):
                trial_count += 1
                episode_performance.append(info.get('performance', 0.0))
                episode_trial_lengths.append(trial_len)
                trial_len = 0
                if trial_count == trials_per_episode:
                    # import pdb; pdb.set_trace()
                    # if np.sum(episode_performance) < 5:
                    #     skip = True
                    if np.any(np.array(episode_performance[1:]) != 1):
                        skip = True
                    if not np.all(np.array(episode_trial_lengths) == episode_trial_lengths[0]):
                        skip = True
                    max_trials_reached = True

            obs_list.append(obs[None, :])
            reward_list.append(reward)
            action_list.append(action)
            state_list.append(hx.detach().cpu().numpy())

            stop = done or trunc or max_trials_reached
        
        if skip:
            num_skipped += 1
            print(episode_performance)
            if num_skipped >= 50:
                import pdb; pdb.set_trace()
            continue
        
        reward_list.append(0)
        action_list.append(0)
        
        obs = np.concatenate(obs_list, axis=0)
        states = np.concatenate(state_list, axis=0)
        rewards = np.array(reward_list)[:, None]
        actions = np.array(action_list)[:, None]

        trajectories.append((obs, actions, rewards, states))

        print(episode_performance)
        i += 1
    
    obs, actions, rewards, states = zip(*trajectories)
    obs = np.stack(obs)
    actions = np.stack(actions)
    rewards = np.stack(rewards)
    states = np.stack(states)

    np.savez(save_path.format(o), obs=obs, actions=actions, rewards=rewards, states=states, obj1=obj1, obj2=obj2)

# import pdb; pdb.set_trace()

# np.save(save_path, trajectories)
