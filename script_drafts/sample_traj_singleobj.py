# import os
# os.chdir('/home/fpei2/learning/ttrnn/')

import copy
import sys
sys.path.insert(1, '/home/fpei2/learning/ttrnn/')

import numpy as np
import torch
import itertools

from ttrnn.trainer import A2C
from ttrnn.tasks.harlow import HarlowMinimal, Harlow1D, HarlowMinimalDelay
from neurogym.wrappers import PassAction, PassReward, Noise
from ttrnn.tasks.wrappers import DiscreteToBoxWrapper, RingToBoxWrapper, ParallelEnvs

ckpt_path = "/home/fpei2/learning/harlow_analysis/runs/harlowdelay3_gru256/epoch=19999-step=20000-v1.ckpt"
save_path = "/home/fpei2/learning/harlow_analysis/runs/harlowdelay3_gru256/trajectories/epoch19999-v1_sample3.npz"

## Build env

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

# sample.npz: env seed = 0, random seed = 0
# sample2.npz: env seed = 1, random seed = 1
# sample3.npz: env seed = 69, random seed = 12
env = copy.deepcopy(pl_module.env.env_list[0])
env.seed(69)

np.random.seed(12)

### Sample trajectories

trials_per_condition = 40

obs, _ = env.reset(reset_obj=True)
obj1, obj2 = env.get_objects()

print(obj1, obj2)

condition_space = {
    'reward_idx': [0, 1],
    'trial_1_objleft': [0, 1],
    # 'trial_1_choice': [1, 2], # dunno how best to enforce this
    'trial_2_objleft': [0, 1],
}

conditions = itertools.product(*condition_space.values())

all_data = []

seen_list = []

save_idx = {
    1: 0,
    2: 1,
}
# save_idx = {
#     2: 0,
# }

# for (r, o1, c, o2) in conditions:
for (r, o1, o2) in conditions:

    print(f'Testing {r}, {o1}, {o2}')

    i = 0
    num_skipped = 0

    trajectories = [[] for _ in range(len(save_idx))]

    # while i < trials_per_condition:
    while any([(len(t) < trials_per_condition) for t in trajectories]):
        # import pdb; pdb.set_trace()

        traj_env = copy.deepcopy(env)

        seed = np.random.randint(0, 10000) # to get different trials
        obs, _ = traj_env.reset(reset_obj=False, seed=seed)
        traj_env.unwrapped.reward_idx = r
        trial = traj_env.new_trial(obj_left=o1)
        traj_env.unwrapped.t = 0
        traj_env.unwrapped.t_ind = 0
        traj_env.unwrapped.num_tr -= 1
        assert trial['reward_idx'] == r
        assert trial['obj_left'] == o1
        obs = traj_env.ob[traj_env.t_ind]
        obs += traj_env.rng.normal(loc=0, scale=std_noise, size=obs.shape)
        obs = np.concatenate((obs, np.array([1., 0., 0.]), np.array([0.])))

        # hx = model.rnn.build_initial_state(1, pl_module.device, pl_module.dtype)
        if np.random.rand() <= 1.0:
            hx = model.rnn.build_initial_state(1, pl_module.device, pl_module.dtype)
            # hx = hx + torch.clip(torch.from_numpy(
            #     np.random.normal(loc=0., scale=std_noise, size=tuple(hx.shape))
            # ), -1.0, 1.0).to(pl_module.device).to(pl_module.dtype)

        obs_list = [obs[None, :]]
        reward_list = []
        action_list = []
        state_list = []

        episode_performance = []
        episode_trial_lengths = []

        save_list = None
        stop_logging = False
        stop = False
        skip = False
        trial_count = 0
        trial_len = 0

        assert np.all(traj_env.obj1 == obj1) and np.all(traj_env.obj2 == obj2)

        while not stop:

            trial_len += 1

            action_logits, value, hx = model(
                torch.from_numpy(obs).to(device=pl_module.device, dtype=pl_module.dtype).unsqueeze(0), 
                hx=hx, 
                cached=True)
            action = action_logits.mode.item() # take highest prob action always

            obs, reward, done, trunc, info = traj_env.step(action)

            if not stop_logging:
                reward_list.append(reward)
                action_list.append(action)
                state_list.append(hx.detach().cpu().numpy())

            max_trials_reached = False
            if info.get('new_trial', False):
                trial_count += 1
                episode_performance.append(info.get('performance', 0.0))
                episode_trial_lengths.append(trial_len)
                trial_len = 0
                if trial_count == 1:
                    save_list = save_idx.get(action, None)
                    if save_list is None:
                        skip = True
                    elif len(trajectories[save_list]) == trials_per_condition:
                        skip = True
                    # if action != c:
                    #     import pdb; pdb.set_trace()
                    trial = traj_env.new_trial(obj_left=o2)
                    # _ = traj_env.ob[traj_env.t_ind]
                    assert trial['reward_idx'] == r
                    assert trial['obj_left'] == o2
                    traj_env.unwrapped.num_tr -= 1
                if trial_count == 2:
                    if episode_performance[1] != 1:
                        skip = True
                    if not episode_trial_lengths[0] == episode_trial_lengths[1]:
                        skip = True
                    if not episode_trial_lengths[0] == 9:
                        skip = True
                    stop_logging = True
                    if (r, o1, save_list, o2) not in seen_list:
                        # import pdb; pdb.set_trace()
                        if not skip:
                            seen_list.append((r, o1, save_list, o2))
                if trial_count == 6:
                    if not np.all(np.array(episode_performance[1:]) == 1):
                        skip = True
                    max_trials_reached = True
            
            if not stop_logging:
                obs_list.append(obs[None, :])

            stop = done or trunc or max_trials_reached
    
        # print(episode_performance + [save_list])
        
        if skip:
            num_skipped += 1
            list_len_str = ', '.join([f'{len(t)}' for t in trajectories])
            print(f'{num_skipped}: {episode_performance}. ' + list_len_str, end='\r')
            if num_skipped >= 300:
                print('\r\033[Kgiving up')
                # import pdb; pdb.set_trace()
                break
            continue
        
        # reward_list.append(0)
        # action_list.append(0)
        
        obs = np.concatenate(obs_list, axis=0)
        states = np.concatenate(state_list, axis=0)
        rewards = np.array(reward_list)[:, None]
        actions = np.array(action_list)[:, None]

        assert save_list is not None
        trajectories[save_list].append((obs, actions, rewards, states))
        if len(trajectories[save_list]) == trials_per_condition:
            print(f'\r\033[KAction {save_list + 1} done')

        # print(episode_performance)
        i += 1
    
    for i in range(len(trajectories)):
        if len(trajectories[i]) < trials_per_condition:
            continue
        obs, actions, rewards, states = zip(*(trajectories[i]))
        obs = np.stack(obs)
        actions = np.stack(actions)
        rewards = np.stack(rewards)
        states = np.stack(states)

        all_data.append((obs, actions, rewards, states, np.array([r, o1, i, o2])))

obs, actions, rewards, states, condition = zip(*all_data)
obs = np.stack(obs)
actions = np.stack(actions)
rewards = np.stack(rewards)
states = np.stack(states)
condition = np.stack(condition)

np.savez(save_path.format(), obs=obs, actions=actions, rewards=rewards, states=states, condition=condition, obj1=obj1, obj2=obj2)

# import pdb; pdb.set_trace()

# np.save(save_path, trajectories)
