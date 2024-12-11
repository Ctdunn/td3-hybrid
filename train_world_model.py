import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms
import imageio


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
            nn.AdaptiveAvgPool2d((2, 2))
        )
        self.fc = nn.Linear(1024, latent_dim)

    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        x = x.contiguous().view(-1, *x.shape[2:])
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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
            nn.Linear(200, 2 * latent_dim)
        )
        self.posterior = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 200), nn.ReLU(),
            nn.Linear(200, 2 * latent_dim)
        )

    def forward(self, prev_latent, action, hidden_state, next_latent=None):
        prev_latent = prev_latent.contiguous()
        action = action.contiguous()
        batch_size = prev_latent.shape[0]
        seq_len = prev_latent.shape[1]

        prior_dists = []
        posterior_dists = []
        hidden_states = []
        current_hidden = hidden_state

        for t in range(seq_len):
            curr_latent = prev_latent[:, t]
            curr_action = action[:, t]
            rnn_input = torch.cat([curr_latent, curr_action], dim=1)
            current_hidden = self.rnn(rnn_input, current_hidden)
            hidden_states.append(current_hidden)

            prior_params = self.prior(current_hidden)
            prior_mean, prior_logvar = torch.chunk(prior_params, 2, dim=1)
            prior_std = torch.exp(0.5 * prior_logvar)
            prior_dist = Normal(prior_mean, prior_std)
            prior_dists.append(prior_dist.mean)

            if next_latent is not None:
                curr_next_latent = next_latent[:, t]
                posterior_input = torch.cat([current_hidden, curr_next_latent], dim=1)
                posterior_params = self.posterior(posterior_input)
                post_mean, post_logvar = torch.chunk(posterior_params, 2, dim=1)
                post_std = torch.exp(0.5 * post_logvar)
                posterior_dist = Normal(post_mean, post_std)
                posterior_dists.append(posterior_dist.mean)

        prior_dists = torch.stack(prior_dists, dim=1)
        hidden_states = torch.stack(hidden_states, dim=1)
        posterior_dists = torch.stack(posterior_dists, dim=1) if next_latent is not None else None

        return prior_dists, posterior_dists, hidden_states


class WorldModel(nn.Module):
    """Complete world model combining all components."""

    def __init__(self, state_dim, action_dim, latent_dim=256, hidden_dim=200):
        super().__init__()
        self.state_encoder = StateEncoder(state_dim, latent_dim)
        self.image_encoder = ImageEncoder(latent_dim)
        self.dynamics = RecurrentStateSpaceModel(latent_dim, action_dim, hidden_dim)

        self.state_decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, state_dim)
        )

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
        batch_size, seq_len = latents.shape[:2]
        latents = latents.contiguous().view(-1, latents.shape[-1])
        actions = actions.contiguous().view(-1, actions.shape[-1])
        combined = torch.cat([latents, actions], dim=-1)
        rewards = self.reward_predictor(combined)
        return rewards.view(batch_size, seq_len)

    def compute_loss(self, states, images, actions, rewards, next_states, next_images):
        latents = self.encode(states, images)
        next_latents = self.encode(next_states, next_images)
        batch_size = states.size(0)
        hidden = torch.zeros(batch_size, self.hidden_dim).to(states.device)

        prior_dists, posterior_dists, _ = self.dynamics(latents, actions, hidden, next_latents)
        decoded_states = self.decode_states(posterior_dists)
        state_loss = F.mse_loss(decoded_states, next_states)
        predicted_rewards = self.predict_rewards(posterior_dists, actions)
        reward_loss = F.mse_loss(predicted_rewards, rewards)
        kl_loss = F.mse_loss(prior_dists, posterior_dists)
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

            latents = [initial_latent]
            rewards = []
            current_hidden = hidden_state

            for t in range(seq_len):
                current_action = actions[:, t:t + 1]
                prior_dist, _, current_hidden = self.dynamics(latents[-1], current_action, current_hidden)
                next_latent = prior_dist.mean
                reward = self.predict_rewards(next_latent, current_action)
                latents.append(next_latent)
                rewards.append(reward)

            latents = torch.cat(latents[1:], dim=1)
            rewards = torch.cat(rewards, dim=1)
            states = self.decode_states(latents)

            return {
                'predicted_states': states.cpu().numpy(),
                'predicted_rewards': rewards.cpu().numpy(),
                'latent_states': latents.cpu().numpy()
            }


class RobosuiteWorldModelDataset(Dataset):
    """Dataset for loading and processing world model training data."""

    def __init__(self, states_dir, videos_dir, sequence_length=10):
        self.states_dir = states_dir
        self.videos_dir = videos_dir
        self.sequence_length = sequence_length
        self.episode_files = [f for f in os.listdir(states_dir) if f.endswith('_data.npy')]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.episode_lengths = []
        for episode_file in self.episode_files:
            episode_data = np.load(os.path.join(states_dir, episode_file), allow_pickle=True)
            self.episode_lengths.append(len(episode_data) - sequence_length + 1)

        self.cum_lengths = np.cumsum([0] + self.episode_lengths)

    def __len__(self):
        return self.cum_lengths[-1]

    def __getitem__(self, idx):
        episode_idx = np.searchsorted(self.cum_lengths[1:], idx, side='right')
        start_pos = idx - self.cum_lengths[episode_idx]

        episode_file = self.episode_files[episode_idx]
        episode_data = np.load(os.path.join(self.states_dir, episode_file), allow_pickle=True)
        sequence_data = episode_data[start_pos:start_pos + self.sequence_length]

        states = torch.FloatTensor(np.stack([s['observation'] for s in sequence_data]))
        actions = torch.FloatTensor(np.stack([s['action'] for s in sequence_data]))
        rewards = torch.FloatTensor(np.stack([s['reward'] for s in sequence_data]))

        episode_num = int(episode_file.split('_')[1])
        video_path = os.path.join(self.videos_dir, f'episode_{episode_num}.mp4')
        video = imageio.get_reader(video_path)

        frames = []
        for frame_idx in range(start_pos, start_pos + self.sequence_length):
            frame = video.get_data(frame_idx)
            frame = Image.fromarray(frame)
            frame = self.transform(frame)
            frames.append(frame)
        images = torch.stack(frames)

        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'images': images
        }


def train_world_model(world_model, states_dir, videos_dir, num_epochs=100,
                      batch_size=32, sequence_length=10, learning_rate=1e-4):
    """Train the world model using collected data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    world_model = world_model.to(device)
    dataset = RobosuiteWorldModelDataset(states_dir, videos_dir, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True)

    print(f"Number of batches per epoch: {len(dataloader)}")

    first_batch = next(iter(dataloader))
    print("\nBatch shapes:")
    print(f"States shape: {first_batch['states'].shape}")
    print(f"Actions shape: {first_batch['actions'].shape}")
    print(f"Images shape: {first_batch['images'].shape}")
    print(f"Rewards shape: {first_batch['rewards'].shape}\n")

    optimizer = torch.optim.Adam(world_model.parameters(), lr=learning_rate)
    best_loss = float('inf')

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        world_model.train()
        for batch in dataloader:
            states = batch['states'].to(device)
            actions = batch['actions'].to(device)
            rewards = batch['rewards'].to(device)
            images = batch['images'].to(device)

            current_states = states[:, :-1]
            current_images = images[:, :-1]
            next_states = states[:, 1:]
            next_images = images[:, 1:]
            current_actions = actions[:, :-1]
            current_rewards = rewards[:, :-1]

            losses = world_model.compute_loss(
                current_states, current_images,
                current_actions, current_rewards,
                next_states, next_images
            )

            optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(world_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += losses['total_loss'].item()
            num_batches += 1

            if num_batches % 10 == 0:
                print(f"Epoch {epoch}, Batch {num_batches}, "
                      f"Loss: {losses['total_loss'].item():.4f}, "
                      f"State Loss: {losses['state_loss'].item():.4f}, "
                      f"Reward Loss: {losses['reward_loss'].item():.4f}, "
                      f"KL Loss: {losses['kl_loss'].item():.4f}")

        avg_loss = total_loss / num_batches
        print(f"\nEpoch {epoch} complete, Average Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': world_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'world_model_best.pt')

        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': world_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'world_model_epoch_{epoch}.pt')
            print(f"Model checkpoint saved for epoch {epoch}.")


if __name__ == "__main__":
    # get dimensions
    dataset = RobosuiteWorldModelDataset('states', 'videos', sequence_length=10)
    first_item = dataset[0]
    state_dim = first_item['states'].shape[1]  # Get actual state dimension
    action_dim = first_item['actions'].shape[1]  # Get actual action dimension

    print(f"Detected dimensions from data:")
    print(f"State dimension: {state_dim}")
    print(f"Action dimension: {action_dim}")

    # Initialize world model
    world_model = WorldModel(state_dim=state_dim, action_dim=action_dim)

    # Train the model
    train_world_model(
        world_model=world_model,
        states_dir='states',
        videos_dir='videos',
        num_epochs=100,
        batch_size=32,
        sequence_length=5
    )