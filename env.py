import numpy as np
from utils import sigmoid

class Simulator:
    """
    Simulator for Contextual Bandits.
    Two reward functions are implemented:
    - Linear reward: E[r | x, a] = sigmoid(theta * [x, a])
    - Nonlinear reward: E[r | x, a] = sigmoid(theta * [x, a] + W[a] * x)
    """
    def __init__(self, k=20, linear=True, d_x=5, d_a=3, seed=123):
        self.linear_reward = linear
        self.k = k # number of actions
        self.d_a = d_a # dimension of action features
        self.d_x = d_x # dimension of context features
        self.seed = seed

        # generate action features
        self.A = np.random.normal(size=(k, d_a))
        # generate theta for the global linear reward function
        self.theta = np.random.normal(size=(d_x + d_a,))
        # generate user-action interaction weights
        if not self.linear_reward:
            self.W = [np.random.normal(size=(self.d_x, self.d_a)) for _ in range(k)]
        else:
            self.W = None

        self.last_batch = None
        self.iter = 0

    def generate_reward(self, A):
        assert len(self.last_batch) == len(A)
        assert np.max(A) < self.k

        if self.linear_reward:
            true_scores = self._generate_reward_linear(self.last_batch)
        else:
            true_scores = self._generate_reward_nonlinear(self.last_batch)

        scores_observed = true_scores[np.arange(len(A)), A]
        Yobs = np.random.binomial(1, sigmoid(scores_observed + np.random.normal(size=(len(A),), scale=.5)))

        return Yobs, true_scores

    def get_action_features(self, chosen_actions):
        return self.A[chosen_actions]

    def _generate_reward_linear(self, X):
        """
        Simple linear reward function.

        :param X: set of contexts: 2d array (n, d_x)
        :return: scores: 2d array (n, k)
        """
        n, d_x = X.shape
        k, d_a = self.A.shape

        X_A = np.hstack([np.repeat(X, k, axis=0), np.tile(self.A, (n, 1))])
        scores = np.dot(X_A, self.theta)

        return scores.reshape(n, k)

    def _generate_reward_nonlinear(self, X):
        """
        Generate nonlinear reward - with Context-Action Interaction Weights

        :param X: set of contexts: 2d array (n, d_x)
        :return: scores: 2d array (n, k)
        """
        n, d_x = X.shape
        k, d_a = self.A.shape

        # adjust action features based on user features
        def get_adjusted_action_features(context_features, action_index):
            return self.A[action_index] + np.dot(context_features, self.W[action_index])

        # generate scores
        scores = np.zeros((n, k))
        for i in range(n):
            for j in range(k):
                adjusted_a = get_adjusted_action_features(X[i], j)
                x_a = np.concatenate([X[i], adjusted_a])
                scores[i, j] = np.dot(x_a, self.theta)

        return scores

    def generate_batch(self, n):
        # generate context features (x)
        X = np.random.normal(size=(n, self.d_x))
        self.last_batch = X

        return X

    def reset(self):
        self.__init__(k=self.k, linear=self.linear_reward, d_x=self.d_x,
                      d_a=self.d_a, seed=self.seed)
