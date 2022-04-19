本次作业我们提供了DQN(带有Target Network)、Soft Q-Learning(SQL)和SAC算法在Cartpole任务上的代码。

**任务1：**同学们需要执行代码并分析结果，并且将DQN去除Target Network执行任务。最终分析DQN(没有Target Network)、DQN(带有Target Network)、Soft Q-Learning和SAC的模型在Cartpole任务上的性能。

**任务2：**同学们尝试将模型应用到MountainCar-v0任务上，并分析结果。如果修改后的模型能够完成MountainCar-v0任务，则可以获得额外加分。

Cartpole任务和MountainCar-v0任务的描述可以参考：https://gym.openai.com/envs/

**代码结构如下：**

utils文件夹：工具文件夹，文件夹下的代码文件会被模型调用

DQN.py：完成Cartpole任务的DQN模型

Soft_QLearning.py:完成Carpole任务的Soft Q-Learning模型

SAC.py：完成Carpole任务的Soft Actor Critic模型

test.py：测试对比DQN、SQL(LAMBDA=10)和SAC(LAMBDA=10)三个模型，并绘制累计奖励曲线。（除此以外，同学们需要加上不带有Target Network的DQN模型的运行结果）

test_SQL.py：测试超参数LAMBDA=1/10/100/1000时Soft Q-Learning模型的性能表现

test_SAC.py：测试超参数LAMBDA=1/10/100/1000时SAC模型的性能表现

game_show.py：可以调用DQN、SQL和SAC进行训练，然后看最终模型玩游戏的效果



**基础环境**：pytorch、tqdm、gym、matplotlib、numpy



**注意：**在修改模型完成MountainCar-v0任务时，状态空间和动作空间发生了变化，并且原始任务的奖励函数较为稀疏，重新设置奖励函数有助于模型训练。除此以外，模型的相关超参数也需要进行修改，以便更好地适用于新任务。



test.py的运行结果如下所示，最终提交时需要再添加不带有Target Network的DQN模型的曲线：

![img](https://lexue.bit.edu.cn/pluginfile.php/484293/mod_assign/intro/compare.png)

test_SQL.py的运行结果如下所示：

![img](https://lexue.bit.edu.cn/pluginfile.php/484293/mod_assign/intro/SQL_lambda.png)

test_SAC.py的运行结果如下所示：

![img](https://lexue.bit.edu.cn/pluginfile.php/484293/mod_assign/intro/SAC_lambda.png)