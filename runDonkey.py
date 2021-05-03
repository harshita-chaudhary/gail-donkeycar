import gym

from stable_baselines import GAIL, SAC, PPO1
from stable_baselines.gail import ExpertDataset, generate_expert_traj
import gym_donkeycar

model = GAIL.load("gail_donkeycar")

env = gym.make('donkey-generated-track-v0')
obs = env.reset()

while True:
  action, _states = model.predict(obs)
  obs, rewards, dones, info = env.step(action)
  env.render()
