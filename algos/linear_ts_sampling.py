import numpy as np
import scipy as sp
import unittest
import torch

from algos.bandit_algorithm import BanditAlgorithm
from utils import sigmoid


class RegularizedLogisticRegression:
    """
    The implementation of Regularized logistic regression with batch updates.
    Algorithm 3 from the paper: "An Empirical Evaluation of Thompson Sampling".
    https://papers.nips.cc/paper_files/paper/2011/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html
    """
    def __init__(self, d, regularization_strength=1.):
        self.d = d # problem dimension (context + action)
        self.lambda_ = regularization_strength
        self.ms = np.zeros(d) # mean of the weights
        self.qs = np.ones(d) * self.lambda_ # inverse variance of the weights

    def regularized_loglikelihood(self, theta, X, Y):
        """
        Regularized log-likelihood for logistic regression.
        source - https://stats.stackexchange.com/questions/17436/logistic-regression-with-lbfgs-solver

        :param theta: weights
        :param X: features
        :param Y: labels
        :return: regularized log-likelihood
        """
        logit = np.dot(X, theta)
        loglikelihood = - np.sum(Y * (logit - np.log((1.0 + np.exp(logit)))) +
                                (1 - Y) * (-np.log((1.0 + np.exp(logit)))))
        regularization = np.sum(self.qs * (theta - self.ms) ** 2)

        return loglikelihood + regularization / 2

    def fit(self, X, Y):
        """
        Fit the regularized logistic regression using BFGS optimization and apply Laplace approximation.

        :param X: features
        :param Y: labels
        :return: point estimate of the weights
        """
        optimLogitBFGS = sp.optimize.minimize(
            self.regularized_loglikelihood,
            x0=self.ms,
            args=(X, Y), method='BFGS',
            options={'gtol': 1e-3, 'disp': False}
        )
        mean_m = optimLogitBFGS.x

        # Laplace approximation
        X_squared = X ** 2
        prob = sigmoid(np.dot(X, mean_m))
        q_diag = X_squared.T @ (prob * (1 - prob))

        # dp: in the original paper, variance update is qs += q_diag, however,
        # I noticed that it leads to rapid variance reduction for bigger batches and
        # insufficient exploartion. I decided to normalize this update by the number
        # of samples in the batch - divide it by const * len(X) - to make it more stable.
        self.qs += q_diag / (0.5 * len(X))
        self.ms = mean_m

    def sample(self, greedy):
        if greedy:
            theta_hat = self.ms
        else:
            std = np.sqrt(1 / self.qs)
            theta_hat = np.random.randn(self.d) * std + self.ms

        return theta_hat


class ThompsonSampling(BanditAlgorithm):
    def __init__(self, name, hparams):
        super(ThompsonSampling, self).__init__()
        self.t = 0
        self.hparams = hparams
        self.name = name

        self.d = self.hparams.context_dim + self.hparams.action_dim
        self.greedy = self.hparams.greedy
        self.lambda_ = self.hparams.regularization_strength
        self.policy = RegularizedLogisticRegression(self.d, self.lambda_)

        self.last_pred_scores = None

    def update(self, context, action, reward):
        self.t += 1
        context = context.cpu().numpy()
        action = action.cpu().numpy()
        reward = reward.cpu().numpy()
        Z = np.concatenate([context, action], axis=1)
        self.policy.fit(Z, reward)

    def action(self, context, all_actions):
        """
        Generate action scores for all actions given the context.

        :param X: context
        :param Aall: all possible actions
        :return: scores of shape (len(X) x len(Aall))
        """
        if self.t < self.hparams.warmup_batches:
            return torch.randint(0, len(all_actions), (len(context),))
        # Z = [X, A] for all possible action
        Z = torch.cat([
            context.repeat_interleave(len(all_actions), dim=0),
            all_actions.repeat(len(context), 1)
        ], dim=1)
        theta_hat = self.policy.sample(self.greedy)
        theta_hat = torch.tensor(theta_hat, dtype=torch.float32)
        scores_pred = (Z @ theta_hat).reshape(len(context), len(all_actions))

        self.last_pred_scores = scores_pred

        return scores_pred.argmax(dim=1)

    def get_stats(self, true_scores):
        if self.last_pred_scores is not None:
            from sklearn.metrics import mean_squared_error
            mean_squared_error = mean_squared_error(true_scores, self.last_pred_scores)
            loss = None
            variance = (1 / self.policy.qs).mean()

            return mean_squared_error, loss, variance
        else:
            return None, None, None


class TestRegularizedLogisticRegression(unittest.TestCase):
    def test_regularized_loglikelihood(self):
        np.random.seed(1)
        d = 3
        X = np.array([[1.0, 2.0, 3.0],
                      [2.0, 3.0, 4.0],
                      [1.5, 2.5, 3.5]])
        theta = np.array([0.1, 0.2, 0.3])
        Y = np.random.binomial(1, sigmoid(np.dot(X, theta)))

        clf = RegularizedLogisticRegression(d, regularization_strength=1.)
        loglikelihood = clf.regularized_loglikelihood(theta, X, Y)

        self.assertEqual(loglikelihood, 0.5851314503476897)

    def test_fit(self):
        np.random.seed(1)
        n = 500000
        d = 4
        X = np.random.rand(n, d)

        theta = np.array([0.5, -0.15, 1.46, -0.88])
        Y = np.random.binomial(1, sigmoid(np.dot(X, theta)))

        clf = RegularizedLogisticRegression(d)
        clf.fit(X, Y)

        self.assertTrue(np.allclose(clf.ms, theta, rtol=1e-01))


class TestThompsonSampling(unittest.TestCase):
    def test_generate_action_scores(self):
        X = torch.tensor([[1.0, 2.0, 3.0],
                          [2.0, 3.0, 4.0],
                          [1.5, 2.5, 3.5]])
        Aall = torch.tensor([[0.1, 0.2, 0.3],
                             [0.4, 0.5, 0.6]])

        class ParamConfig:
            num_actions: int = 2
            context_dim: int = 3
            action_dim: int = 3
            greedy: bool = True
            regularization_strength: float = 1.0
            warmup_batches: int = 0

        hparams = ParamConfig()

        clf = ThompsonSampling('ts', hparams)
        clf.policy.ms = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        actual_actions = clf.action(X, Aall)
        expected_scores = np.array([[1.72, 2.17],
                                    [2.32, 2.77],
                                    [2.02, 2.47]])
        expected_actions = np.array([1, 1, 1])

        self.assertEqual(actual_actions.shape, expected_actions.shape)
        self.assertTrue(np.allclose(actual_actions, expected_actions, rtol=1e-05))


if __name__ == '__main__':
    unittest.main()