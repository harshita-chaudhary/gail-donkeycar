import gym

from stable_baselines import GAIL
from stable_baselines.gail import ExpertDataset
import gym_donkeycar
import tensorflow as tf


dataset = ExpertDataset(expert_path='expert_donkeycar_expert.npz', traj_limitation=10000000, verbose=1)
model = GAIL('CnnPolicy', 'donkey-generated-track-v0', dataset, verbose=1)

model.learn(total_timesteps=100000)
model.save("gail_donkeycar")

