"""Define the abstract class for contextual bandit algorithms."""
from abc import ABC, abstractmethod


class BanditAlgorithm(ABC):
    """A bandit algorithm must be able to do two basic operations.

    1. Choose an action given a context.
    2. Update its internal model given a triple (context, played action, reward).
    """

    @abstractmethod
    def action(self, context, all_actions):
        pass

    @abstractmethod
    def update(self, context, action, reward):
        pass

    @abstractmethod
    def get_stats(self, true_scores):
        pass