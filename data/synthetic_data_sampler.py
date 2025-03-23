import torch

"""
All synthetic datasets are of the shape (num_contexts, dim_context + num_actions + num_actions).
The first dim_context columns are the context, the next num_actions columns are the (noisy) rewards for each action,
and the last num_actions columns are the true scores for each context (those are used for pseudo regret computation).
"""

def sample_dummy_data(num_contexts, dim_context, num_actions, dim_action):
    """
    There is nothing to learn here as the rewards do not depend on the context.
    """
    actions = torch.randn(num_actions, dim_action)
    data = torch.randn(num_contexts, dim_context + num_actions)
    opt_actions = data[:, dim_context].argmax(dim=1)
    opt_rewards = data[torch.arange(num_contexts), dim_context + opt_actions]
    data = torch.cat([data, data[:, dim_context:]], dim=1)
    assert data.shape[1] == dim_context + 2 * num_actions

    return data, actions, (opt_rewards, opt_actions)

def sample_linear_data(num_contexts, dim_context, num_actions, dim_action):
    actions = torch.randn(num_actions, dim_action)
    theta = torch.randn(size=(dim_context + dim_action,))
    contexts = torch.randn(num_contexts, dim_context)

    Z = torch.cat([contexts.repeat_interleave(len(actions), dim=0), actions.repeat(len(contexts), 1)], dim=1)
    true_scores = torch.matmul(Z, theta).reshape(num_contexts, num_actions)
    rewards = torch.bernoulli(torch.sigmoid(true_scores))

    opt_actions = true_scores.argmax(dim=1)
    opt_rewards = rewards[torch.arange(num_contexts), opt_actions]
    data = torch.cat([contexts, rewards, true_scores], dim=1)

    return data, actions, (opt_rewards, opt_actions)

def sample_semilinear_data(num_contexts, dim_context, num_actions, dim_action):
    # TODO: Implement this without for loops
    actions = torch.randn(num_actions, dim_action)
    theta = torch.randn(size=(dim_context + dim_action,))
    betas = [torch.randn(size=(dim_context, dim_action)) for _ in range(num_actions)]
    contexts = torch.randn(num_contexts, dim_context)

    def get_adjusted_action_features(context, action_index):
        return actions[action_index] + context @ betas[action_index]
    true_scores = torch.zeros((num_contexts, num_actions))

    for i in range(num_contexts):
        for j in range(num_actions):
            adjusted_a = get_adjusted_action_features(contexts[i], j)
            z = torch.cat([contexts[i], adjusted_a])
            true_scores[i, j] = z @ theta

    rewards = torch.bernoulli(torch.sigmoid(true_scores))
    opt_actions = true_scores.argmax(dim=1)
    opt_rewards = rewards[torch.arange(num_contexts), opt_actions]
    data = torch.cat([contexts, rewards, true_scores], dim=1)

    return data, actions, (opt_rewards, opt_actions)