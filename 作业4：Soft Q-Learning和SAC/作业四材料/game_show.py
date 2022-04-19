import DQN
import SAC
import Soft_QLearning
import torch
import gym


model = SAC
model.OBS_N = 2
model.ACT_N = 3
model.EPISODES = 300
model.TRAIN_EPOCHS = 500
model.LEARNING_RATE = 5e-3
model.LAMBDA = 10
model.ENV = "MountainCar-v0"
model.train(1)
# env = gym.make("CartPole-v0")
env = gym.make("MountainCar-v0")

obs = env.reset()
j = 0
for i in range(10000):
    action = model.policy(env,obs)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      print("存活了",i-j,"个step")
      j = i
      obs = env.reset()

env.close()