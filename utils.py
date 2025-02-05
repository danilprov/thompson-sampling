from sklearn.metrics import mean_squared_error
import numpy as np


def sigmoid(z):
    # sigmoid function
    return 1 / (1 + np.exp(-z))

def argmax(q_values):
    """
    Takes in a list of q_values and returns the index of the item
    with the highest value. Breaks ties randomly.
    returns: int - the index of the highest value in q_values
    """
    top_value = float("-inf")
    ties = []

    for i in range(len(q_values)):
        if q_values[i] > top_value:
            ties = [i]
            top_value = q_values[i]
        elif q_values[i] == top_value:
            ties.append(i)

    return np.random.choice(ties)

def softmax(x):
    """
    A numerical stable version of the softmax function for 2d arrays.
    """
    exps = np.exp(x - x.max())
    return exps / np.sum(exps, keepdims=True)

def run_experiment(env, agent, n_iter=10000, batch_size=32, verbose=False):
    """
    Run an experiment on a contextual bandit environment.
    """
    logreg_regret = []
    chosen_actions = [0] * env.k
    optimal_actions = [0] * env.k
    random_regret = []

    for i in range(n_iter):
        # sample batch
        Xb = env.generate_batch(batch_size)

        # select actions and observe rewards
        pred_scores = agent.generate_action_scores(Xb, env.A)
        Ab = np.array([argmax(pred_scores[i]) for i in range(len(pred_scores))])
        Y, true_scores = env.generate_reward(Ab)

        # log performance
        Aopt = np.argmax(true_scores, axis=1)
        optimal_score = np.max(true_scores, axis=1)
        chosen_score = true_scores[range(len(Ab)), Ab]
        logreg_regret.append((optimal_score - chosen_score).mean())
        random_score = true_scores[range(len(Ab)), np.random.choice(range(env.k), size=len(Ab))]
        random_regret.append((optimal_score - random_score).mean())
        for j in range(batch_size):
            chosen_actions[Ab[j]] += 1
            optimal_actions[Aopt[j]] += 1

        # track stats
        if i < 50 or i % 500 == 0:
            mse_scores = mean_squared_error(true_scores, pred_scores)
            mse_params = mean_squared_error(env.theta, agent.policy.ms)
            if verbose:
                print(
                    f'{i:7d}/{n_iter:7d} - mse params {mse_params:.4f}, mse scores: {mse_scores:.4f}, variance: {(1 / agent.policy.qs).mean():.4f}')

        # update
        Afeat = env.get_action_features(Ab)
        agent.update_policy(Xb, Afeat, Y)

    return chosen_actions, optimal_actions, logreg_regret, random_regret
