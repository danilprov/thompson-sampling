import torch

from algos.bandit_algorithm import BanditAlgorithm


class UniformSampling(BanditAlgorithm):
    """Algorithm that samples actions uniformly at random."""

    def __init__(self, name, hparams):
        super(UniformSampling, self).__init__()
        self.name = name
        self.hparams = hparams

    def action(self, context, action_features=None):
        all_actions = self.hparams.num_actions
        action = torch.randint(0, all_actions, (len(context),))

        return action

    def update(self, context, action, reward):
        pass

    def get_stats(self, true_scores):
        return None, None, None


if __name__ == '__main__':
    hparams = {}
    algo = UniformSampling('Uniform Sampling', hparams)
