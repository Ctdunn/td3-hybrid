import torch
import numpy as np
import matplotlib.pyplot as plt
from world_model import WorldModel
from trajectory_generator import TrajectoryGenerator
import imageio


def test_world_model_predictions():
    state_dim = 42
    action_dim = 8
    world_model = WorldModel(state_dim=state_dim, action_dim=action_dim)


    print("Loading model...")
    checkpoint = torch.load('world_model_best.pt')
    world_model.load_state_dict(checkpoint['model_state_dict'])
    world_model.eval()

    generator = TrajectoryGenerator(world_model)

    # Load a test episode
    print("Loading test episode...")
    test_episode = np.load('states/episode_0_data.npy', allow_pickle=True)
    initial_state = test_episode[0]['observation']

    # Make sure to load the corresponding video frame
    video = imageio.get_reader('videos/episode_0.mp4')
    initial_image = video.get_data(0)

    # Get the sequence of actions from the episode
    actions = np.stack([step['action'] for step in test_episode[:-1]])  # exclude last step

    print("Generating predicted trajectory...")
    imagined_traj = generator.generate_trajectory(
        initial_state, initial_image, actions
    )

    # real trajectory for comparison
    real_traj = {
        'states': np.stack([step['observation'] for step in test_episode[1:]]),
        'rewards': np.stack([step['reward'] for step in test_episode[:-1]])
    }

    # remove after debugging
    print("\nTrajectory shapes:")
    print("Real states shape:", real_traj['states'].shape)
    print("Predicted states shape:", imagined_traj['predicted_states'].shape)
    print("Real rewards shape:", real_traj['rewards'].shape)
    print("Predicted rewards shape:", imagined_traj['predicted_rewards'].shape)

    errors = generator.evaluate_trajectory_quality(real_traj, imagined_traj)
    print("\nPrediction Errors:")
    print(f"State MSE: {errors['state_mse']:.6f}")
    print(f"Reward MSE: {errors['reward_mse']:.6f}")

    # Plot comparisons
    print("\nGenerating plots...")
    # Plot first 6 state dimensions
    plt.figure(figsize=(15, 10))
    for i in range(6):
        plt.subplot(3, 2, i + 1)
        real_states = real_traj['states'][:, i]
        pred_states = imagined_traj['predicted_states'].squeeze()[:, i]

        timesteps = np.arange(len(real_states))
        plt.plot(timesteps, real_states, 'b-', label='Real')
        plt.plot(timesteps, pred_states, 'r--', label='Predicted')

        plt.title(f'State Dimension {i}')
        plt.xlabel('Timesteps')
        plt.ylabel('Value')
        if i == 0:
            plt.legend()

    plt.tight_layout()
    plt.savefig('state_predictions.png')
    plt.close()

    # Plot rewards
    plt.figure(figsize=(10, 5))
    real_rewards = real_traj['rewards']
    pred_rewards = imagined_traj['predicted_rewards'].squeeze()

    timesteps = np.arange(len(real_rewards))
    plt.plot(timesteps, real_rewards, 'b-', label='Real Rewards')
    plt.plot(timesteps, pred_rewards, 'r--', label='Predicted Rewards')

    plt.title('Reward Predictions')
    plt.xlabel('Timesteps')
    plt.ylabel('Reward')
    plt.legend()
    plt.savefig('reward_predictions.png')
    plt.close()

    print("\nPlots saved as 'state_predictions.png' and 'reward_predictions.png'")


if __name__ == "__main__":
    test_world_model_predictions()