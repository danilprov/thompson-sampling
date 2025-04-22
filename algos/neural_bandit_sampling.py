import torch

from algos.bandit_algorithm import BanditAlgorithm
from algos.neural_bandit_model import NeuralBanditModel


class NeuralBandit(BanditAlgorithm):

    def __init__(self, name, hparams):
        super(NeuralBandit, self).__init__()
        self.name = name
        self.hparams = hparams

        self.t = 0
        self.update_freq_nn = self.hparams.training_freq_network
        self.n_epochs = self.hparams.n_epochs

        self.data_h = ...
        self.nn = NeuralBanditModel(hparams)

        self.last_pred_scores = None

    def action(self, context, all_actions):
        scores_pred = self.nn.predict(context)
        self.last_pred_scores = scores_pred

        # choose the action with the highest score
        action = scores_pred.argmax(dim=1)

        return action

    def update(self, context, action, reward):
        # necessary updates
        #self.data_h.add(context, action, reward)
        #self.t += 1

        # retrain the network on the original data (data_h)
        #if self.t % self.update_freq_nn == 0:
        #    self.nn.train(self.data_h, self.n_epochs)
        pass

    def get_stats(self, true_scores):
        if self.last_pred_scores is not None:
            from sklearn.metrics import mean_squared_error
            mean_squared_error = mean_squared_error(true_scores, self.last_pred_scores)
            loss = None
            variance = None

            return mean_squared_error, loss, variance
        else:
            return None, None, None

if __name__ == '__main__':
    class ParamConfig:
        num_actions: int = 2
        context_dim: int = 3
        action_dim: int = 3
        layer_sizes: int = [20, 20]
        activation = torch.nn.ReLU()
        use_dropout: bool = True
        keep_prob: float = 0.8
        initial_lr: float = 0.01
        lr_decay_rate: float = 0.95
        training_freq_network: int = 10
        n_epochs: int = 10


    X = torch.tensor([[1.0, 2.0, 3.0],
                      [2.0, 3.0, 4.0],
                      [1.5, 2.5, 3.5]])
    Aall = torch.tensor([[0.1, 0.2, 0.3],
                         [0.4, 0.5, 0.6]])

    hparams = ParamConfig()
    bm = NeuralBandit('neural_bandit', hparams)
    actions = bm.action(X, Aall)

    print(actions)