import torch
from tqdm import tqdm

def safe_format(value):
    return f"{value:.4f}" if value is not None else "N/A"

def run_contextual_bandit(context_dim, actions, dataset, algos, batch_size, verbose=False):
    """Run a contextual bandit experiment on a set of algorithms.

    Args:
      context_dim: Dimension of the context.
      actions: Available actions (actions features or id)
      dataset: Matrix where every row is a context + num_actions rewards.
      algos: List of algorithms to use in the experiment.

    Returns:
      h_actions: Matrix with actions: size (num_context, num_algorithms).
      h_rewards: Matrix with rewards: size (num_context, num_algorithms).
    """

    num_contexts = dataset.shape[0]
    max_iters = int(num_contexts / batch_size)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create contextual bandit
    cmab = ContextualBandit(context_dim, actions)
    cmab.feed_data(dataset)

    h_actions = torch.empty((num_contexts, len(algos)), dtype=torch.long, device='cpu')
    h_rewards = torch.empty((num_contexts, len(algos)), dtype=torch.float, device='cpu')
    h_regrets = torch.empty((num_contexts, len(algos)), dtype=torch.float, device='cpu')
    row_idx = 0

    # Run the contextual bandit process
    for i in range(max_iters):
        context = cmab.context(batch_size).to(device)
        actions = []
        rewards = []

        for a in algos:
            action = a.action(context, cmab.actions()).to(device)
            reward = cmab.reward(action).to(device)

            actions.append(action.cpu())
            rewards.append(reward.cpu())

            # update the model
            # TODO: make sure that this update works for both algos with action features and without
            a.update(context, cmab.actions()[action], reward)
            if verbose and (i < 25 or i % 500 == 0):
                true_scores = cmab.true_scores()
                mse_scores, loss, variance = a.get_stats(true_scores)
                # print(f'{i:7d}/{max_iters:7d} - {a.name} - '
                #       f'loss: {loss:.4f}, ' if loss is not None else 'mse scores: N/A, '
                #       f'mse scores: {mse_scores:.4f}, ' if mse_scores is not None else 'mse scores: N/A, '
                #       f'variance: {variance:.4f}' if variance is not None else 'variance: N/A, ')
                print(f'{i:7d}/{max_iters:7d} - '
                      f'loss: {safe_format(loss)}, '
                      f'mse scores: {safe_format(mse_scores)}, '
                      f'variance: {safe_format(variance)}')

        #optimal_reward, _ = cmab.optimal()
        actions_tensor = torch.stack(actions, dim=1)
        rewards_tensor = torch.stack(rewards, dim=1)
        actual_batch_size = min(batch_size, context.shape[0])
        h_actions[row_idx:row_idx + actual_batch_size] = actions_tensor
        h_rewards[row_idx:row_idx + actual_batch_size] = rewards_tensor
        #h_regrets[row_idx:row_idx + actual_batch_size] = optimal_reward.reshape(actual_batch_size, 1) - rewards_tensor
        h_regrets[row_idx:row_idx + actual_batch_size] = cmab.regret(actions_tensor)
        row_idx += actual_batch_size

    return h_actions, h_rewards, h_regrets

class ContextualBandit(object):
    """Implements a Contextual Bandit with d-dimensional contexts and k arms."""

    def __init__(self, context_dim, actions):
        self._context_dim = context_dim
        self._actions = actions
        self._num_actions = len(actions)
        self.current_index = 0
        self.indices = None
        self.data = None

    def feed_data(self, data):
        """Feeds the data (contexts + rewards) to the bandit object.

        Args:
        data: Numpy array with shape [n, d+k], where n is the number of contexts,
          d is the dimension of each context, and k the number of arms (rewards).
        """

        if data.shape[1] != self.context_dim + self.num_actions + self.num_actions:
            raise ValueError('Data dimensions do not match.')

        self._number_contexts = data.shape[0]
        self.data = data
        self.order = torch.arange(self.number_contexts)

    def reset(self):
        """Randomly shuffle the order of the contexts to deliver."""
        self.order = torch.randperm(self.number_contexts)
        self.current_index = 0
        self.indices = None

    def context(self, batch_size):
        """
        Retrieves one batch of contexts
        We also update self.indices and generate rewards for the batch here.
        """
        if self.current_index >= self.number_contexts:
            raise StopIteration("No more data available. Please reset the bandit.")

        end_index = min(self.current_index + batch_size, self.number_contexts)
        self.indices = self.order[self.current_index:end_index]
        self.current_index = end_index

        return self.data[self.indices, :self.context_dim]

    def reward(self, action):
        """Returns the (noisy) rewards for the current batch of indices and given actions."""
        return self.data[self.indices, self.context_dim + action]

    # def optimal(self):
    #     """Returns the optimal scores and action (in hindsight) for the current batch of indices."""
    #     optimal_score, optimal_action = torch.max(self.data[self.indices, self.context_dim + self.num_actions:], dim=1)
    #     return optimal_score, optimal_action

    def regret(self, actions):
        """

        :param actions: torch tensor of shape batch_size x num_policies
        :return:
        """
        true_scores = self.data[self.indices, self.context_dim + self.num_actions:]
        optimal_score = torch.max(true_scores, dim=1, keepdim=True)[0] # (batch_size, 1)
        chosen_scores = torch.gather(true_scores, dim=1, index=actions)  # (batch_size, num_algos)
        return optimal_score - chosen_scores

    def true_scores(self):
        return self.data[self.indices, self.context_dim + self.num_actions:]

    @property
    def context_dim(self):
        return self._context_dim

    @property
    def num_actions(self):
        return self._num_actions

    def actions(self):
        return self._actions

    @property
    def number_contexts(self):
        return self._number_contexts
