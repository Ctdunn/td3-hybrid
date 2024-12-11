import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio
from torchvision import transforms


class TrajectoryGenerator:
    def __init__(self, world_model, device=None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.world_model = world_model.to(self.device)
        self.world_model.eval()  # Set to evaluation mode

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def initialize_latent_state(self, state, image):
        """Initialize latent state from a real state and image."""
        with torch.no_grad():
            # Debug prints removed
            if isinstance(state, dict):
                # GymWrapper uses 'robot0_proprio-state'
                state = state['robot0_proprio-state']

            state = torch.FloatTensor(state).unsqueeze(0).unsqueeze(0).to(self.device)

            # image processing
            if isinstance(image, np.ndarray):
                image = self.transform(image)
            image = image.unsqueeze(0).unsqueeze(0).to(self.device)

            # Encode initial state
            latent = self.world_model.encode(state, image)

            batch_size = 1
            hidden = torch.zeros(batch_size, self.world_model.hidden_dim).to(self.device)

            return latent, hidden

    def generate_trajectory(self, initial_state, initial_image, actions):
        """Generate an imagined trajectory from an initial state and sequence of actions."""
        with torch.no_grad():
            if isinstance(actions, np.ndarray):
                actions = torch.FloatTensor(actions)
            actions = actions.to(self.device)

            if len(actions.shape) == 2:
                actions = actions.unsqueeze(0)

            # Initialize latent and hidden states
            latent, hidden = self.initialize_latent_state(initial_state, initial_image)

            # Generate trajectory
            trajectory = self.world_model.imagine_trajectory(latent, actions, hidden)

            return trajectory

    def evaluate_trajectory_quality(self, real_trajectory, imagined_trajectory):
        """Compare imagined trajectory with real trajectory to evaluate prediction quality."""
        real_states = real_trajectory['states']
        real_rewards = real_trajectory['rewards']

        pred_states = imagined_trajectory['predicted_states'].squeeze()
        pred_rewards = imagined_trajectory['predicted_rewards'].squeeze()

        # Compute prediction errors
        state_mse = np.mean((real_states - pred_states) ** 2)
        reward_mse = np.mean((real_rewards - pred_rewards) ** 2)

        return {
            'state_mse': state_mse,
            'reward_mse': reward_mse
        }


def visualize_trajectories(real_trajectory, imagined_trajectory, save_path=None):
    """Visualize comparison between real and imagined trajectories."""
    # Plot state predictions
    num_dims = min(6, real_trajectory['states'].shape[1])  # Plot first 6 dimensions
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    axes = axes.flatten()

    real_states = real_trajectory['states']
    pred_states = imagined_trajectory['predicted_states'].squeeze()
    timesteps = np.arange(len(real_states))

    for i in range(num_dims):
        axes[i].plot(timesteps, real_states[:, i], 'b-', label='Real')
        axes[i].plot(timesteps, pred_states[:, i], 'r--', label='Predicted')
        axes[i].set_title(f'State Dimension {i}')
        axes[i].set_xlabel('Timesteps')
        axes[i].set_ylabel('Value')
        if i == 0:
            axes[i].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + '_states.png')
    plt.close()

    # Plot reward predictions
    plt.figure(figsize=(10, 5))
    real_rewards = real_trajectory['rewards']
    pred_rewards = imagined_trajectory['predicted_rewards'].squeeze()

    plt.plot(timesteps, real_rewards, 'b-', label='Real Rewards')
    plt.plot(timesteps, pred_rewards, 'r--', label='Predicted Rewards')
    plt.title('Reward Predictions')
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.legend()

    if save_path:
        plt.savefig(save_path + '_rewards.png')
    plt.close()