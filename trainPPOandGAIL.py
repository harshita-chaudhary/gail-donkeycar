import gym

from stable_baselines import GAIL, SAC, PPO1
from stable_baselines.gail import ExpertDataset, generate_expert_traj
import gym_donkeycar

# Generate expert trajectories (train expert)
#model = PPO1('MlpPolicy', 'donkey-generated-track-v0', verbose=1)
model = PPO1('MlpPolicy', 'donkey-generated-track-v0', verbose=1)
generate_expert_traj(model, 'expert_donkeycar', n_timesteps=1000000, n_episodes=1000)

# Load the expert dataset
dataset = ExpertDataset(expert_path='expert_donkeycar.npz', traj_limitation=1, verbose=1)

model = GAIL('MlpPolicy', 'donkey-generated-track-v0', dataset, verbose=1)
# Note: in practice, you need to train for 1M steps to have a working policy
model.learn(total_timesteps=100000)
model.save("gail_donkeycar")


env = gym.make('donkey-generated-track-v0')
obs = env.reset()

while True:
  action, _states = model.predict(obs)
  obs, rewards, dones, info = env.step(action)
  env.render()
