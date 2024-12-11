import time
import os
import numpy as np
import imageio
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3_torch import Agent


if not os.path.exists("videos"):
    os.makedirs("videos")

if not os.path.exists("states"):
    os.makedirs("states")


def save_video(frames, episode_idx, fps=20):
    """Saves the video of an episode."""
    video_filename = f"videos/episode_{episode_idx}.mp4"
    writer = imageio.get_writer(video_filename, fps=fps)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    print(f"Video for Episode {episode_idx} saved to {video_filename}.")


def save_data(episode_idx, data_buffer):
    """Saves the state-action-reward data of an episode."""
    data_filename = f"states/episode_{episode_idx}_data.npy"
    np.save(data_filename, np.array(data_buffer))
    print(f"Data for Episode {episode_idx} saved to {data_filename}.")


if __name__ == '__main__':
    if not os.path.exists("/tmp/td3"):
        os.makedirs("/tmp/td3")

    env_name = "Lift"

    # Initialize the Robosuite environment
    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=False,  # Changed to False to disable on-screen rendering
        use_camera_obs=False,
        horizon=50,
        render_camera="agentview",
        has_offscreen_renderer=True,  # Keep True for video recording
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(env)

    actor_learning_rate = 0.0010
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, tau=0.005,
                  input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0],
                  layer1_size=layer1_size,
                  layer2_size=layer2_size, batch_size=batch_size)

    agent.load_models()

    n_games = 1000
    fps = 20
    best_score = 0

    for episode_idx in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        frames = []
        data_buffer = []

        while not done:
            # Only render if saving frames
            frame = env.sim.render(camera_name="agentview", height=480, width=640, depth=False)
            frames.append(frame)

            action = agent.choose_action(observation, validation=True)
            next_observation, reward, done, info = env.step(action)

            data_buffer.append({
                "observation": observation,
                "action": action,
                "reward": reward,
                "next_observation": next_observation,
                "done": done
            })

            score += reward
            observation = next_observation

            if info.get("success", False):
                print(f"Task completed in Episode {episode_idx}!")
                done = True


        print(f"Episode: {episode_idx} Score: {score}")

        save_video(frames, episode_idx, fps)
        save_data(episode_idx, data_buffer)

    print("Testing complete. Videos and data saved in 'deletevid' and 'states' folders.")

