import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class NeuralBanditModel(nn.Module):
    """
    A policy underlying the bandit algorithm implemented as a neural network.

    Note: this implementation treats actions as classes and the network outputs the K logits for each action.
    Therefore, this policy ignores the action context (if it exists).
    """

    def __init__(self, hparams):
        super(NeuralBanditModel, self).__init__()

        self.hparams = hparams
        self.verbose = getattr(self.hparams, "verbose", True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # hidden layers
        layers = []
        input_dim = self.hparams.context_dim
        for num_units in self.hparams.layer_sizes:
            layers.append(nn.Linear(input_dim, num_units))
            layers.append(self.hparams.activation)
            if getattr(self.hparams, "use_dropout", False):
                layers.append(nn.Dropout(p=1 - self.hparams.keep_prob))
            input_dim = num_units

        # output layer
        self.net = nn.Sequential(*layers).to(self.device)
        self.output_layer = nn.Linear(input_dim, self.hparams.num_actions).to(self.device)

        # optimizer and learning rate scheduler
        self.optimizer = optim.RMSprop(self.parameters(), lr=self.hparams.initial_lr)
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=1, gamma=self.hparams.lr_decay_rate
        )

    def forward(self, x):
        nn_output = self.net(x)
        y_pred = self.output_layer(nn_output)
        return y_pred

    def train_model(self, data_loader, num_steps):
        """Trains the network for num_steps, using the provided data loader."""

        if self.verbose:
            print(f"Training for {num_steps} steps...")

        self.train()
        for step, (contexts, rewards, weights) in enumerate(data_loader):
            contexts = contexts.to(self.device)
            rewards = rewards.to(self.device)
            weights = weights.to(self.device)
            if step >= num_steps:
                break

            self.optimizer.zero_grad()

            # Forward pass
            y_pred = self.forward(contexts)

            # Compute weighted loss
            loss = torch.mean(weights * (y_pred - rewards) ** 2)

            # Backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), self.hparams.max_grad_norm)
            self.optimizer.step()

            if step % self.hparams.freq_summary == 0:
                print(f"Step {step}, Loss: {loss.item():.4f}")

            self.lr_scheduler.step()

    def predict(self, context):
        """Returns predictions for the given context x."""
        self.eval()
        with torch.no_grad():
            return self.forward(context)


# Example hyperparameters class
class HParams:
    def __init__(self):
        self.context_dim = 10
        self.num_actions = 5
        self.layer_sizes = [50, 50]
        self.activation = nn.ReLU()
        self.use_dropout = True
        self.keep_prob = 0.8
        self.initial_lr = 0.01
        self.lr_decay_rate = 0.95
        self.max_grad_norm = 5.0
        self.freq_summary = 100
        self.verbose = True


# Example usage
# hparams = HParams()
# model = NeuralBanditModel(hparams)
#
# # Create dummy data loader
# batch_size = 32
# x = torch.rand(batch_size, hparams.context_dim)
# y = torch.rand(batch_size, hparams.num_actions)
# w = torch.ones(batch_size, hparams.num_actions)  # Example weights
# data_loader = [(x, y, w)] * 500  # Dummy repeated dataset
#
# # Train model
# model.train_model(data_loader, num_steps=500)
#
# # Predict
# preds = model.predict(x)
# print(preds)
