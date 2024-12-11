import os
import torch
import numpy as np
from trajectory_generator import TrajectoryGenerator
from robosuite.wrappers import GymWrapper

if not os.path.exists('/tmp'):
    os.makedirs('/tmp')


class WorldModelPlanner:
    def __init__(self, world_model, action_dim=8, planning_horizon=10,
                 num_trajectories=100, device=None):
        """Initialize the planner."""
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = TrajectoryGenerator(world_model, device=self.device)
        self.action_dim = action_dim
        self.planning_horizon = planning_horizon
        self.num_trajectories = num_trajectories

    def generate_action_sequences(self, mean_action=None, std=0.1):
        """Generate random action sequences for planning."""
        if mean_action is None:
            mean_action = np.zeros(self.action_dim)

        actions = np.random.normal(
            loc=mean_action,
            scale=std,
            size=(self.num_trajectories, self.planning_horizon, self.action_dim)
        )

        actions = np.clip(actions, -1, 1)
        return actions

    def evaluate_trajectories(self, trajectories, goal_state=None):
        """Evaluate trajectories based on rewards and optionally distance to goal."""
        rewards = trajectories['predicted_rewards']
        total_rewards = np.sum(rewards, axis=1)

        if goal_state is not None:
            final_states = trajectories['predicted_states'][:, -1]
            distances = np.linalg.norm(final_states - goal_state, axis=1)
            scores = total_rewards - 0.5 * distances
        else:
            scores = total_rewards

        return scores

    def plan_action_sequence(self, initial_state, initial_image, goal_state=None,
                             mean_action=None, std=0.1):
        """Plan a sequence of actions using random shooting."""
        action_sequences = self.generate_action_sequences(mean_action, std)

        best_reward = float('-inf')
        best_actions = None
        best_trajectory = None

        for actions in action_sequences:
            actions_batch = torch.FloatTensor(actions).unsqueeze(0).to(self.device)
            trajectory = self.generator.generate_trajectory(
                initial_state, initial_image, actions_batch
            )

            score = np.mean(self.evaluate_trajectories(trajectory, goal_state))

            if score > best_reward:
                best_reward = score
                best_actions = actions
                best_trajectory = trajectory

        return best_actions, best_trajectory, best_reward

    def plan_next_action(self, initial_state, initial_image, goal_state=None,
                         previous_actions=None):
        """Plan and return just the next action to take."""
        mean_action = None if previous_actions is None or len(previous_actions) == 0 else previous_actions[0]
        action_sequence, trajectory, reward = self.plan_action_sequence(
            initial_state, initial_image, goal_state, mean_action
        )
        return action_sequence[0], trajectory, reward

    def replan_online(self, env, num_steps=50, goal_state=None):
        """Execute online replanning in the environment."""
        observation = env.reset()
        total_reward = 0
        executed_actions = []

        for step in range(num_steps):
            # Get current image from environment
            current_image = env.sim.render(camera_name="agentview", height=480, width=640, depth=False)[:, :, :3]

            # Plan next action
            action, trajectory, predicted_reward = self.plan_next_action(
                observation, current_image, goal_state, executed_actions
            )

            # Execute action
            next_observation, reward, done, info = env.step(action)
            executed_actions.append(action)
            total_reward += reward

            print(f"Step {step}: Reward = {reward:.2f}, Predicted = {predicted_reward:.2f}")

            # Render the environment
            if env.has_renderer:
                env.render()

            if done:
                break

            observation = next_observation

        return total_reward, executed_actions



if __name__ == "__main__":
    try:
        import robosuite as suite
        from world_model import WorldModel

        print("Initializing world model and planner...")
        state_dim = 42
        action_dim = 8

        world_model = WorldModel(state_dim=state_dim, action_dim=action_dim)

        # remove this after test

        print("\nChecking saved training data:")
        test_episode = np.load('states/episode_0_data.npy', allow_pickle=True)
        print("Saved state shape:", test_episode[0]['observation'].shape)

        world_model.load_state_dict(
            torch.load('world_model_epoch_4.pt')['model_state_dict']
        )

        print("Creating environment...")

        env = suite.make(
            "Lift",
            robots=["Panda"],
            has_renderer=True,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            horizon=50,
            render_camera="agentview",
            control_freq=20,
        )

        env = GymWrapper(env)

        print("\nEnvironment information:")
        print("Observation space shape:", env.observation_space.shape)
        observation = env.reset()
        if isinstance(observation, dict):
            print("Observation keys:", observation.keys())
            print("Observation shapes:")
            for key, value in observation.items():
                print(f"{key}: {value.shape}")
        else:
            print("Observation direct shape:", observation.shape)

        print("\nInitializing planner...")

        planner = WorldModelPlanner(
            world_model=world_model,
            action_dim=action_dim,
            planning_horizon=10,
            num_trajectories=100
        )

        print("Starting online planning test...")
        total_reward, actions = planner.replan_online(env, num_steps=50)
        print(f"Planning test complete. Total reward: {total_reward}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise