import torch
import numpy as np
from td3_torch import Agent
from world_model import WorldModel
from model_based_planning import WorldModelPlanner


class HybridTD3Agent(Agent):
    def __init__(self, world_model, *args,
                 imagination_ratio=0.3,  # Ratio of imagined to real transitions
                 planning_steps=10,  # How many steps to plan ahead
                 **kwargs):
        super().__init__(*args, **kwargs)

        self.world_model = world_model
        self.planner = WorldModelPlanner(
            world_model=world_model,
            action_dim=kwargs['n_actions'],
            planning_horizon=planning_steps
        )
        self.imagination_ratio = imagination_ratio
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.world_model.to(self.device)

    def imagine_trajectories(self, state, image, num_trajectories=10):
        """Generate imagined trajectories using the world model."""
        imagined_transitions = []

        # Generate multiple trajectories
        for _ in range(num_trajectories):
            # Use current policy to generate actions
            actions = []
            current_state = state

            for _ in range(self.planner.planning_horizon):
                action = self.choose_action(current_state, validation=True)
                actions.append(action)

            actions = np.array(actions)

            # Debugging remove
            #print(f"Actions shape: {actions.shape}, Actions: {actions}")

            # Generate trajectory using world model
            trajectory = self.planner.generator.generate_trajectory(
                state, image, actions
            )

            # Debugging remove
            #print(f"Generated trajectory: {trajectory}")

            # Store transitions
            for t in range(len(actions)):
                if t + 1 < len(trajectory['predicted_states']):  # Safeguard
                    transition = {
                        'state': trajectory['predicted_states'][t],
                        'action': actions[t],
                        'next_state': trajectory['predicted_states'][t + 1],
                        'reward': trajectory['predicted_rewards'][t],
                        'done': False  # Assume non-terminal for imagined transitions
                    }
                    imagined_transitions.append(transition)
                else:
                    # Debugging remove
                    #print(f"Trajectory too short at step {t}: {trajectory}")
                    break

        return imagined_transitions

    def learn(self, real_state=None, real_image=None):
        """Enhanced learning step using both real and imagined data."""
        if self.memory.mem_ctr < self.batch_size * 10:
            return

        # Regular TD3 learning from real experience
        super().learn()

        # Add imagined experience if we have current state/image
        if real_state is not None and real_image is not None:
            # Generate imagined trajectories
            num_imagined = int(self.batch_size * self.imagination_ratio)
            imagined_transitions = self.imagine_trajectories(
                real_state, real_image,
                num_trajectories=max(1, num_imagined // self.planner.planning_horizon)
            )

            # Add imagined transitions to buffer
            for transition in imagined_transitions:
                self.memory.store_transition(
                    transition['state'],
                    transition['action'],
                    transition['reward'],
                    transition['next_state'],
                    transition['done']
                )

    def choose_action_with_planning(self, state, image, use_planning_prob=0.3):
        """Choose action using either planning or direct policy."""
        if np.random.random() < use_planning_prob:
            # Use planning
            action, _, _ = self.planner.plan_next_action(state, image)
            return action
        else:
            # Use regular TD3 policy
            return self.choose_action(state)


def train_hybrid_agent(env, num_episodes=1000, save_dir='tmp/hybrid_td3'):
    """
    Train the hybrid agent.
    """
    import os
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Helper function to preprocess observations
    def preprocess_observation(observation, selected_keys):
        # Concatenate values from the selected keys
        return np.concatenate([observation[key] for key in selected_keys if key in observation])

    # Initialize world model and hybrid agent
    obs_spec = env.observation_spec()
    print("Observation spec keys:", obs_spec.keys())

    # Filter keys to calculate state_dim
    selected_keys = [
        "robot0_joint_pos_cos", "robot0_joint_pos_sin", "robot0_joint_vel",
        "robot0_eef_pos", "robot0_eef_quat",
        "robot0_gripper_qpos", "robot0_gripper_qvel", "cube_pos", "cube_quat", "gripper_to_cube_pos"
    ]
    state_dim = sum(np.prod(obs_spec[key].shape) for key in selected_keys if key in obs_spec)
    print("Filtered State dimensions (state_dim):", state_dim)

    action_dim = np.prod(env.action_spec[0].shape)
    print("Action dimensions (action_dim):", action_dim)

    # Load pretrained world model
    world_model = WorldModel(state_dim=state_dim, action_dim=action_dim)
    world_model.load_state_dict(
        torch.load('world_model_epoch_4.pt')['model_state_dict']
    )

    # Create hybrid agent
    agent = HybridTD3Agent(
        world_model=world_model,
        actor_learning_rate=0.001,
        critic_learning_rate=0.001,
        input_dims=[state_dim],
        tau=0.005,
        env=env,
        gamma=0.99,
        update_actor_interval=2,
        warmup=1000,
        n_actions=action_dim,
        max_size=1000000,
        layer1_size=256,
        layer2_size=128,
        batch_size=128,
        noise=0.1
    )

    best_reward = float('-inf')
    reward_history = []

    for episode in range(num_episodes):
        observation = env.reset()
        observation = preprocess_observation(observation, selected_keys)
        done = False
        episode_reward = 0
        steps = 0

        while not done:
            # Get current image
            current_image = env.sim.render(
                width=640, height=480, camera_name="agentview"
            )

            # Choose action using hybrid approach
            action = agent.choose_action_with_planning(
                observation, current_image
            )

            # Take step in environment
            next_observation, reward, done, info = env.step(action)
            next_observation = preprocess_observation(next_observation, selected_keys)

            # Store transition
            agent.remember(observation, action, reward, next_observation, done)

            # Learn from both real and imagined experience
            agent.learn(real_state=observation, real_image=current_image)

            observation = next_observation
            steps += 1

        reward_history.append(episode_reward)
        avg_reward = np.mean(reward_history[-100:])

        print(f"Episode: {episode}, Steps: {steps}, "
              f"Reward: {episode_reward:.2f}, "
              f"100-episode Average: {avg_reward:.2f}")

        # Save if we have a new best model
        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save_models()
            print(f"New best reward: {best_reward:.2f}")

        # Periodic saving
        if episode % 100 == 0:
            save_path = os.path.join(save_dir, f'hybrid_td3_episode_{episode}')
            agent.save_models()
            np.save(os.path.join(save_dir, 'reward_history.npy'), reward_history)

    return agent, reward_history



if __name__ == "__main__":
    import robosuite as suite

    # Create environment
    env = suite.make(
        "Lift",
        robots=["Panda"],
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        use_camera_obs=False,
        control_freq=20,
    )

    # Train hybrid agent
    trained_agent, rewards = train_hybrid_agent(env)
    print("Training complete!")
