import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from game.flappy_bird import GameState
import os
import random
import sys
import time
import cv2
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()

        self.number_of_actions = 2
        self.gamma = 0.99
        self.final_epsilon = 0.0001
        self.initial_epsilon = 0.1
        self.number_of_iterations = 2000000
        self.replay_memory_size = 10000
        self.minibatch_size = 32

        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)
        self.relu3 = nn.ReLU(inplace=True)
        self.fc4 = nn.Linear(3136, 512)
        self.relu4 = nn.ReLU(inplace=True)
        self.fc5 = nn.Linear(512, self.number_of_actions)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.relu3(out)
        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu4(out)
        out = self.fc5(out)

        return out


def image_to_tensor(image):
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():  # put on GPU if CUDA is available
        image_tensor = image_tensor.cuda()
    return image_tensor


def resize_and_bgr2gray(image):
    image = image[0:288, 0:404]
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data


def test(model):
    game_state = GameState()

    # initial action is do nothing
    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat(
        (image_data, image_data, image_data, image_data)).unsqueeze(0)

    total_reward = 0

    while reward != -1 and total_reward <= 1000:
        # get output from the neural network
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action = action.cuda()

        # get action
        action_index = torch.argmax(output)
        if torch.cuda.is_available():  # put on GPU if CUDA is available
            action_index = action_index.cuda()
        action[action_index] = 1

        # get next state
        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat(
            (state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)
        total_reward += reward
        # print(total_reward)
        # set state to be state_1
        state = state_1
    return total_reward


cuda_is_available = torch.cuda.is_available()
plt.ion()
tmp_model = 25000
reward_list = []
model_list = []
while tmp_model <= 2000000:
    model = torch.load(
        'pretrained_model/current_model_'+str(tmp_model)+'.pth',
        map_location='cpu' if not cuda_is_available else None
    ).eval()
    if cuda_is_available:  # put on GPU if CUDA is available
        model = model.cuda()
    reward = 0
    for i in range(3):
        reward += test(model)
    reward /= 3
    print('model:','current_model_'+str(tmp_model)+'.pth','reward:',reward)
    reward_list.append(reward)
    model_list.append(tmp_model)
    tmp_model += 25000
    plt.xlabel('iteration')
    plt.ylabel('reward')
    plt.title('DQN with target-net')
    plt.plot(model_list, reward_list)
    plt.savefig('dqn_target.jpg')
    plt.show()