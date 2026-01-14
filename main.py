import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

env = Monitor(gym.make("CartPole-v1"))
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=5e-4,
    buffer_size=200000,
    learning_starts=5000,
    batch_size=64,
    gamma=0.99,
    train_freq=4,
    target_update_interval=1000,
    exploration_fraction=0.4,
    exploration_final_eps=0.05,
    verbose=1,
)


model.learn(total_timesteps=400000)
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
print("Baseline eval:", mean_reward, "+/-", std_reward)
model.save("dqn_cartpole_baseline")
env.close()


