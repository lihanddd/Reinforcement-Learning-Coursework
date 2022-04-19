本次实验分为三个部分，**第一部分**：采用Q-learning算法实现与作业2相同的任务，**第二部分**：补全迷宫导航任务中的Q-learning算法和sarsa算法。**第三部分：**在原始DQN模型中加上Target Network，并进行训练，比较模型性能。

**第一部分**

本部分提供RL.py、TestRL.py和TestMaze.py三个代码文件，其中RL.py中需要同学们实现qLearning算法，TestRL.py是简单的测试，TestMaze.py中是复杂的测试。

并且在实现过程中会用到作业2中实现的MDP.py，请将MDP.py放在同一目录下。具体任务：

1.补全RL.py中Q-learning方法的代码，并解释实现思路

2.运行TestRL.py并提交运行截图

3.运行TestMaze.py并提交运行截图，并分析生成的Q-learning.jpg的含义和曲线的走势。样例图片如下所示，并且同学们提交时需要将XXX改为个人姓名

提示：所给样例图片不一定是最终结果

![Q-learning](https://lexue.bit.edu.cn/pluginfile.php/483111/mod_assign/intro/Q-learning.jpg?time=1639376837160)

**需要提交：**

1.word文档，包括代码实现截图和实现思路、TestRL的运行结果、TestMaze的运行结果图和分析

2.源代码



**第二部分**

本部分提供q-learning.py和sarsa.py两个文件，迷宫导航任务环境的相关设置已经在代码中给出，同学们需要补全Q-learning算法和sarsa算法

注意该任务是随机初始化四个障碍和目的地，如果出现路径不存在的情况，请重新运行。最终的运行结果如下所示：

![演示结果](https://lexue.bit.edu.cn/pluginfile.php/483111/mod_assign/intro/%E6%BC%94%E7%A4%BA%E7%BB%93%E6%9E%9C.PNG)

**结果解释**：左侧红色表示障碍，绿色表示目的地。右侧是V值的一个可视化界面，偏向绿色表示该状态离目标状态较为接近。

**需要提交：**

1.word文档(接着第一部分)，包括补全代码部分截图和实现思路、运行结果截图（两张）、算法分析（Q-learning和sarsa的区别）

2.源代码



**第三部分**

本部分需要使用DQN模型来玩Flappy bird游戏，我们给出tensorflow和Pytorch两个版本实现的原始DQN，以供参考。同学们需要修改代码，加上Target Network，比较改变前后模型的性能，具体体现为训练过程中奖励值的变化，如下所示：

![result](https://lexue.bit.edu.cn/pluginfile.php/483111/mod_assign/intro/result.jpg?time=1639565062981)

**文件说明：**

Flappybird-pytorch中：assets和game文件夹中是游戏环境的设置，pretrained_model中保存训练好的模型，dqn.py包括模型主体和游戏执行部分。

Flappybird-tensorflow中：assets和game文件夹中是游戏环境的设置，BrainDQN_NIPS.py是模型主体，FlappyBirdDQN.py是游戏执行部分，record.csv是记录每个episode持续时长和奖励的文件。