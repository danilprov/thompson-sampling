from dataclasses import dataclass
import time
import torch
import sparklines

from envs.contextual_bandit import run_contextual_bandit

from algos.uniform_sampling import UniformSampling
from algos.linear_ts_sampling import ThompsonSampling

from data.synthetic_data_sampler import sample_dummy_data
from data.synthetic_data_sampler import sample_linear_data
from data.synthetic_data_sampler import sample_semilinear_data


@dataclass
class ParamConfig:
    num_actions: int = 3
    context_dim: int = 5
    action_dim: int = 3
    greedy: bool = True
    regularization_strength: float = 1.0
    warmup_batches: int = 5

def sample_data(data_type, num_contexts=None):
    """Sample data from given 'data_type'.
    TODO: Add more data_types.
    """
    if data_type == 'dummy':
        # Create dummy dataset
        num_actions = 8
        action_dim = 3
        context_dim = 10
        dataset, available_actions, opt_dummy = sample_dummy_data(num_contexts, context_dim,
                                                                  num_actions, action_dim)
        opt_rewards, opt_actions = opt_dummy
    elif data_type == 'linear':
        # Create linear dataset
        num_actions = 8
        action_dim = 3
        context_dim = 10
        dataset, available_actions, opt_linear = sample_linear_data(num_contexts, context_dim,
                                                                    num_actions, action_dim)
        opt_rewards, opt_actions = opt_linear
    elif data_type == 'semilinear':
        # Create semi linear dataset
        num_actions = 8
        action_dim = 3
        context_dim = 10
        dataset, available_actions, opt_semilinear = sample_semilinear_data(num_contexts, context_dim,
                                                                            num_actions, action_dim)
        opt_rewards, opt_actions = opt_semilinear
    else:
        raise ValueError('Invalid data_type %s' % data_type)

    return dataset, available_actions, context_dim, action_dim, opt_rewards, opt_actions

def display_results(algos, opt_rewards, opt_actions, h_rewards, h_actions, t_init, name):
    """Displays summary statistics of the performance of each algorithm."""

    print('---------------------------------------------------')
    print('---------------------------------------------------')
    print('{} bandit completed after {} seconds.'.format(
    name, time.time() - t_init))
    print('---------------------------------------------------')

    performance_pairs = []
    algname_to_index = {a.name: i for i, a in enumerate(algos)}
    for j, a in enumerate(algos):
        performance_pairs.append((a.name, torch.sum(h_rewards[:, j])))
    performance_pairs = sorted(performance_pairs,
                               key=lambda x: x[1],
                               reverse=True)
    for i, (name, reward) in enumerate(performance_pairs):
        print('{:3}) {:20}| \t \t total reward = {:10}.'.format(i, name, reward))
        alg_idx = algname_to_index[name]
        chosen_actions = h_actions[:, alg_idx]
        print([[elt, chosen_actions.tolist().count(elt)] for elt in set(chosen_actions.tolist())])
        print(sparklines.sparklines([chosen_actions.tolist().count(elt) for elt in sorted(set(chosen_actions.tolist()))])[0])


    print('---------------------------------------------------')
    print('Optimal total reward = {}.'.format(torch.sum(opt_rewards)))
    print('Frequency of optimal actions (action, frequency):')
    print([[elt, opt_actions.tolist().count(elt)] for elt in set(opt_actions.tolist())])
    print('---------------------------------------------------')
    print('---------------------------------------------------')

def main():
    batch_size = 32
    data_type = 'linear'
    num_contexts = 10000

    sampled_vals = sample_data(data_type, num_contexts)
    dataset, available_actions, context_dim, action_dim, opt_rewards, opt_actions = sampled_vals

    hparams = {
        'context_dim': context_dim,
        'action_dim': action_dim,
        'num_actions': len(available_actions)
    }
    hparams = ParamConfig(**hparams)

    hparams_ts_linear = {
        'context_dim': context_dim,
        'action_dim': action_dim,
        'num_actions': len(available_actions),
        'greedy': False,
        'regularization_strength': 0.1,
        'warmup_batches': 0#min(5, num_contexts // batch_size)
    }
    hparams_ts_linear = ParamConfig(**hparams_ts_linear)

    algos = [UniformSampling('Uniform Sampling', hparams),
             ThompsonSampling('Thompson linear', hparams_ts_linear)]
    # Run contextual bandit problem
    t_init = time.time()
    results = run_contextual_bandit(context_dim, available_actions, dataset, algos, batch_size)
    h_actions, h_rewards, h_regrets = results
    display_results(algos, opt_rewards, opt_actions, h_rewards, h_actions, t_init, data_type)

if __name__ == '__main__':
    import numpy as np
    torch.manual_seed(123)
    np.random.seed(1)
    main()