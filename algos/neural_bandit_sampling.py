import torch

from algos.bandit_algorithm import BanditAlgorithm


class NeuralBandit(BanditAlgorithm):

    def __init__(self, name, hparams, optimizer):
        super(NeuralBandit, self).__init__()
        self.name = name
        self.hparams = hparams

        self.t = 0
        self.update_freq_nn = self.hparams.training_freq_network
        self.n_epochs = self.hparams.n_epochs

        self.data_h = ...
        self.nn = ...
        self.optimizer = optimizer

    def get_action_scores(self, context, all_actions):
        Z = torch.cat([
            context.repeat_interleave(len(all_actions), dim=0),
            all_actions.repeat(len(context), 1)
        ], dim=1)

        with torch.no_grad():
            self.nn.eval()
            scores_pred = self.nn(Z)

        scores_pred = scores_pred.reshape(len(context), len(all_actions))
        self.nn.train()

        return scores_pred

    def action(self, context):
        all_actions = ...
        scores_pred = self.get_action_scores(context, all_actions)

        # choose the action with the highest score
        action = all_actions[scores_pred.argmax()]

        return action

    def update(self, context, action, reward):
        # necessary updates
        #self.data_h.add(context, action, reward)
        #self.t += 1

        # retrain the network on the original data (data_h)
        #if self.t % self.update_freq_nn == 0:
        #    self.nn.train(self.data_h, self.n_epochs)
        pass