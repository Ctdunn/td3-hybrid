import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class ImageEncoder(nn.Module):
    """Encodes images into a latent space."""

    def __init__(self, latent_dim=256):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=4, stride=2), nn.ReLU(),
            # Add adaptive average pooling to get fixed size output
            nn.AdaptiveAvgPool2d((2, 2))
        )
        # 256 * 2 * 2 = 1024
        self.fc = nn.Linear(1024, latent_dim)

    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        # Reshape to (batch_size * seq_len, 3, 480, 640)
        x = x.contiguous().view(-1, *x.shape[2:])

        x = self.convs(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        # Reshape back
        return x.view(batch_size, seq_len, -1)


class StateEncoder(nn.Module):
    """Encodes state vectors into the same latent space."""

    def __init__(self, state_dim, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )

    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        x = x.contiguous()
        x = x.view(-1, x.shape[-1])
        x = self.net(x)
        return x.view(batch_size, seq_len, -1)


class RecurrentStateSpaceModel(nn.Module):
    """Models the dynamics in latent space."""

    def __init__(self, latent_dim=256, action_dim=8, hidden_dim=200):
        super().__init__()
        self.rnn = nn.GRUCell(latent_dim + action_dim, hidden_dim)

        self.prior = nn.Sequential(
            nn.Linear(hidden_dim, 200), nn.ReLU(),
            nn.Linear(200, 2 * latent_dim)  # Mean and logvar
        )

        self.posterior = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 200), nn.ReLU(),
            nn.Linear(200, 2 * latent_dim)  # Mean and logvar
        )

    def forward(self, prev_latent, action, hidden_state, next_latent=None):
        """
        Args:
            prev_latent: [batch_size, 1, latent_dim]
            action: [batch_size, 1, action_dim]
            hidden_state: [batch_size, hidden_dim]
            next_latent: Optional [batch_size, 1, latent_dim]
        """
        # Remove sequence dimension for GRU input
        prev_latent = prev_latent.squeeze(1)
        action = action.squeeze(1)

        rnn_input = torch.cat([prev_latent, action], dim=1)

        next_hidden = self.rnn(rnn_input, hidden_state)

        # Compute prior
        prior_params = self.prior(next_hidden)
        prior_mean, prior_logvar = torch.chunk(prior_params, 2, dim=1)

        # Add sequence dimension back
        prior_mean = prior_mean.unsqueeze(1)

        # Compute posterior
        posterior_mean = None
        if next_latent is not None:
            next_latent = next_latent.squeeze(1)
            posterior_input = torch.cat([next_hidden, next_latent], dim=1)
            posterior_params = self.posterior(posterior_input)
            post_mean, post_logvar = torch.chunk(posterior_params, 2, dim=1)
            posterior_mean = post_mean.unsqueeze(1)

        return prior_mean, posterior_mean, next_hidden


class WorldModel(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim=256, hidden_dim=200):
        super().__init__()
        self.state_encoder = StateEncoder(state_dim, latent_dim)
        self.image_encoder = ImageEncoder(latent_dim)
        self.dynamics = RecurrentStateSpaceModel(latent_dim, action_dim, hidden_dim)

        # decoders for reconstruction
        self.state_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, state_dim)
        )

        # reward predictor
        self.reward_predictor = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

    def encode(self, state, image):
        state_latent = self.state_encoder(state)
        image_latent = self.image_encoder(image)
        return (state_latent + image_latent) / 2

    def decode_states(self, latents):
        batch_size, seq_len = latents.shape[:2]
        latents = latents.contiguous().view(-1, latents.shape[-1])
        states = self.state_decoder(latents)
        return states.view(batch_size, seq_len, -1)

    def predict_rewards(self, latents, actions):
        """Predict rewards from latents and actions."""
        # Handle both sequence and single-step predictions
        if isinstance(latents, torch.Tensor):
            if len(latents.shape) == 2:
                latents = latents.unsqueeze(1)
                actions = actions.unsqueeze(1)

            batch_size, seq_len = latents.shape[:2]
            latents = latents.contiguous().view(-1, latents.shape[-1])
            actions = actions.contiguous().view(-1, actions.shape[-1])

            combined = torch.cat([latents, actions], dim=-1)
            rewards = self.reward_predictor(combined)
            return rewards.view(batch_size, seq_len)
        else:
            latent_tensor = latents.mean if hasattr(latents, 'mean') else latents
            return self.predict_rewards(latent_tensor, actions)

    def compute_loss(self, states, images, actions, rewards, next_states, next_images):
        # Encode states and images
        latents = self.encode(states, images)
        next_latents = self.encode(next_states, next_images)

        # Initialize hidden state
        batch_size = states.size(0)
        hidden = torch.zeros(batch_size, self.hidden_dim).to(states.device)

        # Forward dynamics
        prior_dists, posterior_dists, _ = self.dynamics(
            latents, actions, hidden, next_latents
        )

        # Reconstruction loss for states using posterior
        decoded_states = self.decode_states(posterior_dists)
        state_loss = F.mse_loss(decoded_states, next_states)

        # Reward prediction loss
        predicted_rewards = self.predict_rewards(posterior_dists, actions)
        reward_loss = F.mse_loss(predicted_rewards, rewards)

        # KL divergence between prior and posterior (using simplified MSE for stability)
        kl_loss = F.mse_loss(prior_dists, posterior_dists)

        # Total loss with weightings
        total_loss = state_loss + reward_loss + 0.1 * kl_loss

        return {
            'total_loss': total_loss,
            'state_loss': state_loss,
            'reward_loss': reward_loss,
            'kl_loss': kl_loss
        }

    def imagine_trajectory(self, initial_latent, actions, hidden_state):
        """Generate an imagined trajectory given a sequence of actions."""
        with torch.no_grad():
            batch_size = actions.size(0)
            seq_len = actions.size(1)
            device = actions.device

            # Lists to store predictions
            latents = [initial_latent]  # [batch_size, 1, latent_dim]
            rewards = []
            current_hidden = hidden_state  # [batch_size, hidden_dim]

            # Generate trajectory
            for t in range(seq_len):
                # Get current latent and action
                curr_latent = latents[-1]  # [batch_size, 1, latent_dim]
                curr_action = actions[:, t:t + 1]  #[batch_size, 1, action_dim]

                # dynamics model to predict next state
                prior_dist, _, next_hidden = self.dynamics(
                    curr_latent,
                    curr_action,
                    current_hidden
                )

                current_hidden = next_hidden

                latents.append(prior_dist)  # Keep sequence dimension

                # predict reward using latent state
                reward = self.predict_rewards(prior_dist, curr_action)
                rewards.append(reward)

            latents = torch.cat(latents[1:], dim=1)  # [batch_size, seq_len, latent_dim]
            rewards = torch.cat(rewards, dim=1)  # [batch_size, seq_len]

            # decode states
            states = self.decode_states(latents)

            return {
                'predicted_states': states.cpu().numpy(),
                'predicted_rewards': rewards.cpu().numpy(),
                'latent_states': latents.cpu().numpy()
            }