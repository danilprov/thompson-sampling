import numpy as np
import scipy as sp
import unittest
from utils import sigmoid


class RegularizedLogisticRegression:
    """
    The implementation of Regularized logistic regression with batch updates.
    Algorithm 3 from the paper: "An Empirical Evaluation of Thompson Sampling".
    https://papers.nips.cc/paper_files/paper/2011/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html
    """
    def __init__(self, d, regularization_strength=1., greedy=False):
        self.d = d # problem dimension (context + action)
        self.lambda_ = regularization_strength
        self.greedy = greedy
        self.ms = np.zeros(d) # mean of the weights
        self.qs = np.ones(d) * self.lambda_ # variance of the weights

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

        return mean_m

    def generate_action_scores(self, X, Aall):
        """
        Generate action scores for all actions given the context.

        :param X: context
        :param Aall: all possible actions
        :return: scores of shape (len(X) x len(Aall))
        """
        # Z = [X, A] for all possible actions
        Z = np.hstack([np.repeat(X, len(Aall), axis=0), np.tile(Aall, (len(X), 1))])
        if self.greedy:
            scores_pred = np.dot(Z, self.ms).reshape(len(X), len(Aall))
        else:
            theta_hat = np.random.randn(self.d) * np.sqrt(1 / self.qs) + self.ms
            scores_pred = np.dot(Z, theta_hat).reshape(len(X), len(Aall))

        return scores_pred

    def reset(self):
        self.__init__(self.d, self.lambda_, self.greedy)

    def get_params(self):
        return self.ms, self.qs

    def update_policy(self, X, A, Y):
        Z = np.concatenate([X, A], axis=1)
        self.fit(Z, Y)


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
        theta_hat = clf.fit(X, Y)

        self.assertTrue(np.allclose(theta_hat, theta, rtol=1e-01))

    def test_generate_action_scores(self):
        X = np.array([[1.0, 2.0, 3.0],
                      [2.0, 3.0, 4.0],
                      [1.5, 2.5, 3.5]])
        Aall = np.array([[0.1, 0.2, 0.3],
                         [0.4, 0.5, 0.6]])

        clf = RegularizedLogisticRegression(X.shape[1] + Aall.shape[1], greedy=True)
        clf.ms = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        actual_scores = clf.generate_action_scores(X, Aall)

        expected_scores = np.array([[1.72, 2.17],
                                    [2.32, 2.77],
                                    [2.02, 2.47]])
        self.assertEqual(actual_scores.shape, expected_scores.shape)
        self.assertTrue(np.allclose(actual_scores, expected_scores, rtol=1e-05))


if __name__ == '__main__':
    unittest.main()