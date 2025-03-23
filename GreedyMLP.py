import torch
import torch.nn as nn
import unittest
from utils import softmax_torch


class MLP(nn.Module):
    def __init__(self, d):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self,x):
        x = self.fc(x)
        return x


class IncrementalGreedyAgent:
    def __init__(self, d, learning_params={}):
        self.d = d
        self.Xbuffer = torch.empty((0, d), dtype=torch.float32)
        self.Ybuffer = torch.empty(0, dtype=torch.int32)
        self.lossbuffer = []

        self.n_epochs = learning_params.get('n_epochs', 10)
        self.batch_size = learning_params.get('batch_size', 32)
        self.lr = learning_params.get('lr', 0.0001)
        self.temperature = learning_params.get('temperature', 1.)

        self.policy = MLP(d)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr)

    def generate_action_scores(self, X, Aall):
        """
        Generate action scores for all actions given the context.

        :param X: context
        :param Aall: all possible actions
        :return: scores of shape (len(X) x len(Aall))
        """
        # Z = [X, A] for all possible actions
        #Z = np.hstack([np.repeat(X, len(Aall), axis=0), np.tile(Aall, (len(X), 1))])
        Z = torch.cat([X.repeat_interleave(len(Aall), dim=0), Aall.repeat(len(X), 1)], dim=1)
        with torch.no_grad():
            self.policy.eval()
            scores_pred = self.policy(Z)

        scores_pred = scores_pred.reshape(len(X), len(Aall))
        self.policy.train()

        return scores_pred

    def fit(self, X, Y):
        self.Xbuffer = torch.cat([self.Xbuffer, X])
        self.Ybuffer = torch.cat([self.Ybuffer, Y])

        lossi  = []
        for i in range(self.n_epochs):
            ix = torch.randint(0, self.Xbuffer.shape[0], (self.batch_size,))
            Xb, Yb = self.Xbuffer[ix], self.Ybuffer[ix]

            y_pred = self.policy(Xb)
            loss = torch.nn.BCEWithLogitsLoss()(y_pred.reshape(-1), Yb)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # track stats
            # if i % 10000 == 0:  # print every once in a while
            #     print(f'{i:7d}/{self.n_epochs:7d}: {loss.item():.4f}')
            lossi.append(loss.log10().item())
        self.lossbuffer += lossi

    def update_policy(self, X, A, Y):
        Z = torch.cat([X, A], dim=1)
        self.fit(Z, Y)

    def sample_actions(self, X, Aall):
        pred_scores = self.generate_action_scores(X, Aall)
        pred_probs = softmax_torch(pred_scores / self.temperature)
        Asampled = torch.multinomial(pred_probs, num_samples=1).squeeze(1)

        return Asampled, pred_scores


class TestIncrementalGreedyAgent(unittest.TestCase):
    def test_generate_action_scores(self):
        torch.manual_seed(1)
        d = 5
        policy = IncrementalGreedyAgent(d)

        class ToyModel(nn.Module):
            def __init__(self, d):
                super(ToyModel, self).__init__()
                self.fc = nn.Sequential(
                    nn.Linear(d, 1),
                )

            def forward(self, x):
                x = self.fc(x)
                return x
        policy.policy = ToyModel(d)
        #self.assertTrue(torch.allclose(policy.policy.fc[0].weight[0], torch.tensor([-0.1319, -0.3881,  0.0552, -0.1449, -0.2353]), rtol=1e-03))

        X = torch.tensor([[1.0, 2.0, 3.0],
                      [2.0, 3.0, 4.0],
                      [1.5, 2.5, 3.5]])
        Aall = torch.tensor([[0.1, 0.2],
                         [0.4, 0.5]])
        expected_scores = torch.tensor([
            [-0.7372821569442749, -0.8513319492340088],
            [-1.201964259147644, -1.316014051437378],
            [-0.9696231484413147, -1.0836730003356934]
        ])
        actual_scores = policy.generate_action_scores(X, Aall)
        self.assertTrue(torch.allclose(actual_scores, expected_scores, rtol=1e-08))

    def test_sample_actions(self):
        torch.manual_seed(1)
        d = 5
        X = torch.tensor([[1.0, 2.0, 3.0],
                      [2.0, 3.0, 4.0],
                      [1.5, 2.5, 3.5]])
        Aall = torch.tensor([[0.1, 0.2],
                         [0.4, 0.5]])
        policy = IncrementalGreedyAgent(d)
        Asampled = policy.sample_actions(X, Aall)
        self.assertTrue(torch.equal(torch.tensor([1, 0, 0]), Asampled))

    def test_fit(self):
        torch.manual_seed(1)
        d = 8
        n, n_te = 200000, 20000
        theta = torch.tensor([1.51067951, 0.26639065, -0.64699223, -0.56858549,
                              0.09642841, -0.17085906, -3.28359654, -1.22302803])
        learning_params = {'n_epochs': 100000, 'batch_size': 64, 'lr': 0.00001}

        # data
        X = torch.rand(n, d)
        true_scores = X @ theta
        Y = torch.bernoulli(torch.sigmoid(true_scores))
        Xtr, Xte = X[:n - n_te], X[n - n_te:]
        Ytr, Yte = Y[:n - n_te], Y[n - n_te:]

        # train model
        greedy_policy = IncrementalGreedyAgent(d, learning_params=learning_params)
        greedy_policy.fit(Xtr, Ytr)

        with torch.no_grad():
            greedy_policy.policy.eval()
            scores_pred = greedy_policy.policy(Xte).reshape(-1)
        mse_test = ((scores_pred - true_scores[n - n_te:]) ** 2).sum() / n_te
        self.assertTrue(mse_test < 0.004)


if __name__ == '__main__':
    # unittest.main()

    from envs.linear_bandit import SimulatorTorchWrapper

    k = 20
    batch_size = 4
    env = SimulatorTorchWrapper(k=k, linear=True)
    d = env.d_x + env.d_a
    policy = IncrementalGreedyAgent(d)

    chosen_actions = [0] * env.k
    optimal_actions = [0] * env.k
    for i in range(1):
        Xb = env.generate_batch(batch_size)

        # select actions and observe rewards
        pred_scores = policy.generate_action_scores(Xb, env.A)
        Ab = torch.argmax(pred_scores, dim=1)
        Y, true_scores = env.generate_reward(Ab)

        # log performance
        optimal_scores, Aopt = torch.max(true_scores, dim=1)
        Arand = torch.randint(0, env.k, (len(Ab),))
        chosen_score = true_scores[range(len(Ab)), Ab]
        random_score = true_scores[range(len(Ab)), Arand]
        for j in range(batch_size):
            chosen_actions[Ab[j]] += 1
            optimal_actions[Aopt[j]] += 1

        Afeat = env.A[Ab]
        policy.update_policy(Xb, Afeat, Y)
