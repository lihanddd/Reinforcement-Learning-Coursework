import math

import numpy as np
import MDP

class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        reward = self.sampleReward(self.mdp.R[action,state]) #按照当前R值为均值根据高斯密度函数得到随机奖励
        cumProb = np.cumsum(self.mdp.T[action,state,:])#把
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        return [reward,nextState]

    def qLearning(self,s0,initialQ,nEpisodes,nSteps,epsilon=0,temperature=0):
        '''
        qLearning算法，需要将Epsilon exploration和 Boltzmann exploration 相结合。
        以epsilon的概率随机取一个动作，否则采用 Boltzmann exploration取动作。
        当epsilon和temperature都为0时，将不进行探索。

        Inputs:
        s0 -- 初始状态
        initialQ -- 初始化Q函数 (|A|x|S| array)
        nEpisodes -- 回合（episodes）的数量 (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- 每个回合的步数(steps)
        epsilon -- 随机选取一个动作的概率
        temperature -- 调节 Boltzmann exploration 的参数

        Outputs: 
        Q -- 最终的 Q函数 (|A|x|S| array)
        policy -- 最终的策略
        rewardList -- 每个episode的累计奖励（|nEpisodes| array）
        '''
        nActions, nStates = initialQ.shape
        Q = initialQ
        rewardList = np.zeros(nEpisodes)
        n = np.zeros([nActions, nStates])
        for episode in range(nEpisodes):
            total_reward = 0
            s = s0
            for step in range(nSteps):
                prob = np.random.uniform()
                # print(prob)
                if(prob < epsilon):
                    action = np.random.choice(nActions)
                    # print("epsilon")
                else:
                    if(temperature != 0):
                        action = np.argmax(np.exp(Q[:,s]/temperature))
                        # print("Boltzmann")
                    else:
                        action = np.argmax(Q[:,s])
                        # print("greedy")
                # print("action=",action)
                reward, next_state = self.sampleRewardAndNextState(s, action)
                n[action,s] += 1
                alpha = 1/n[action,s]
                Q[action,s] = Q[action,s] + alpha * (reward + self.mdp.discount * np.max(Q[:, next_state]) - Q[action, s])
                s = next_state
                total_reward += reward
                # policy = np.argmax(Q, axis=0)
                # print(policy)
            rewardList[episode] = total_reward
            # print(total_reward)
        policy = np.argmax(Q, axis=0)
        return [Q,policy,rewardList]