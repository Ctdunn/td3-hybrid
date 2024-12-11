import os
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3_torch import Agent

if __name__ == '__main__':

    if not os.path.exists("tmp/td3"):
        os.makedirs("tmp/td3")
    # "Door", "Lift", "Stack", "TwoArmLift"
    env_name = "Lift"

    env = suite.make(
        env_name,
        robots=["Panda"],
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=False,
        use_camera_obs=False,  # Disable camera observations
        horizon=300,
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(env)

    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128

    agent = Agent(actor_learning_rate=actor_learning_rate, critic_learning_rate=critic_learning_rate, tau=0.005,
                  input_dims=env.observation_space.shape, env=env, n_actions=env.action_space.shape[0], layer1_size=layer1_size,
                  layer2_size=layer2_size, batch_size=batch_size)

    writer = SummaryWriter('logs')
    n_games = 1000
    best_score = 0
    episode_identifier = (f"1 - actor_learning_rate={actor_learning_rate} critic_learning_rate={critic_learning_rate} "
                          f"layer1_size={layer1_size} layer2_size={layer2_size}")

    agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, next_observation, done)
            agent.learn()
            observation = next_observation

        writer.add_scalar(f"Score - {episode_identifier}", score, global_step=i)

        if score > best_score:
            best_score = score
            print(f"New best score: {best_score} at episode {i}")
            agent.save_models()

        if i % 10:
            agent.save_models()

        print(f"Episode: {i} Score: {score}")

