import numpy as np


class MDP:
    '''一个简单的MDP类，它包含如下成员'''

    def __init__(self, T, R, discount):
        '''构建MDP类

        输入:
        T -- 转移函数: |A| x |S| x |S'| array
        R -- 奖励函数: |A| x |S| array
        discount -- 折扣因子: scalar in [0,1)

        构造函数检验输入是否有效，并在MDP对象中构建相应变量'''

        assert T.ndim == 3, "转移函数无效，应该有3个维度"
        self.nActions = T.shape[0]
        self.nStates = T.shape[1]
        assert T.shape == (self.nActions, self.nStates, self.nStates), "无效的转换函数：它具有维度 " + \
            repr(T.shape) + ", 但它应该是(nActions,nStates,nStates)"
        assert (abs(T.sum(2)-1) < 1e-5).all(), "无效的转移函数：某些转移概率不等于1"
        self.T = T
        assert R.ndim == 2, "奖励功能无效：应该有2个维度"
        assert R.shape == (self.nActions, self.nStates), "奖励函数无效：它具有维度 " + \
            repr(R.shape) + ", 但它应该是 (nActions,nStates)"
        self.R = R
        assert 0 <= discount < 1, "折扣系数无效：它应该在[0,1]中"
        self.discount = discount

    def valueIteration(self, initialV, nIterations=np.inf, tolerance=0.01):
        '''值迭代法
        V <-- max_a R^a + gamma T^a V

        输入:
        initialV -- 初始的值函数: 大小为|S|的array
        nIterations -- 迭代次数的限制：标量 (默认值: infinity)
        tolerance -- ||V^n-V^n+1||_inf的阈值: 标量 (默认值: 0.01)

        Outputs: 
        V -- 值函数: 大小为|S|的array
        iterId -- 执行的迭代次数: 标量
        epsilon -- ||V^n-V^n+1||_inf: 标量'''

        iterId = 0
        epsilon = 0
        V = initialV
        # 当循环次数小于nIter
        while iterId < nIterations:
            # V_ <-- max_a R^a + gamma T^a V
            V_ = np.max(self.R + self.discount * (self.T.dot(V)), axis=0)
            # 计算前后iter的V的差
            epsilon = np.sum(np.fabs(V-V_))
            # epsilon小于阈值则跳出循环
            if(epsilon < tolerance):
                V = V_
                break
            # 将V_赋值给下次循环的V
            V = V_
            iterId += 1

        return [V, iterId, epsilon]

    def extractPolicy(self, V):
        '''从值函数中提取具体策略的程序
        pi <-- argmax_a R^a + gamma T^a V

        Inputs:
        V -- 值函数: 大小为|S|的array

        Output:
        policy -- 策略: 大小为|S|的array'''
        # pi <-- argmax_a R^a + gamma T^a V
        policy = np.argmax(self.R + self.discount * self.T.dot(V), axis=0)

        return policy

    def evaluatePolicy(self, policy):
        '''通过求解线性方程组来评估政策
        V^pi = R^pi + gamma T^pi V^pi

        Input:
        policy -- 策略: 大小为|S|的array

        Ouput:
        V -- 值函数: 大小为|S|的array'''
        # V = np.zeros(self.nStates)
        # for s in range(self.nStates):
        #     w = 1-self.discount*self.T[policy[s],s,s]
        #     b = self.R[policy[s],s]
        #     V[s] = b/w
        V = np.zeros(self.nStates)
        tolerance = 0.01
        nS = self.nStates
        while True:
            V_ = np.copy(V)
            # 对于s个状态
            for s in range(nS):
                a = policy[s]
                # 新的V(s)=从s开始的所有可能转移状态value的和
                V_[s] = sum([self.T[a, s, s_] * (self.R[a, s] + self.discount * V_[s_])
                             for s_ in range(nS)])
            epsilon = np.fabs((np.sum(V_-V)))
            if (epsilon < tolerance):
                V = V_
                break
            V = V_
        return V

    def policyIteration(self, initialPolicy, nIterations=np.inf):
        '''策略迭代程序:  在策略评估(solve V^pi = R^pi + gamma T^pi V^pi) 和
        策略改进 (pi <-- argmax_a R^a + gamma T^a V^pi)之间多次迭代

        Inputs:
        initialPolicy -- 初始策略: 大小为|S|的array
        nIterations -- 迭代数量的限制: 标量 (默认值: inf)

        Outputs: 
        policy -- 策略: 大小为|S|的array
        V -- 值函数: 大小为|S|的array
        iterId --策略迭代执行的次数 : 标量'''

        V = np.zeros(self.nStates)
        policy = initialPolicy
        iterId = 0
        # 当循环次数小于nIter
        while iterId < nIterations:
            # 从policy解析最合适的V
            V_ = self.evaluatePolicy(policy)
            # 从合适的V推出合适的policy
            policy_ = np.argmax(self.R + self.discount *
                                (self.T.dot(V_)), axis=0)
            # 当前后policy一样退出循环
            if((policy == policy_).all()):
                policy = policy_
                V = V_
                break
            policy = policy_
            V = V_
            iterId += 1

        return [policy, V, iterId]

    def evaluatePolicyPartially(self, policy, initialV, nIterations=np.inf, tolerance=0.01):
        '''部分的策略评估:
        Repeat V^pi <-- R^pi + gamma T^pi V^pi

        Inputs:
        policy -- 策略: 大小为|S|的array
        initialV -- 初始的值函数: 大小为|S|的array
        nIterations -- 迭代数量的限制: 标量 (默认值: infinity)
        tolerance --  ||V^n-V^n+1||_inf的阈值: 标量 (默认值: 0.01)

        Outputs: 
        V -- 值函数: 大小为|S|的array
        iterId -- 迭代执行的次数: 标量
        epsilon -- ||V^n-V^n+1||_inf: 标量'''

        V = initialV
        iterId = 0
        epsilon = 0
        # 当循环次数小于nIter
        while iterId < nIterations:
            # Repeat V^pi <-- R^pi + gamma T^pi V^pi
            V_ = self.R + self.discount*self.T.dot(V)
            V__ = np.array([V_[policy[s], s] for s in range(self.nStates)])
            epsilon = np.sum(np.fabs(V__ - V))
            if epsilon < tolerance:
                V = V__
                break
            V = V__
            iterId += 1

        return [V, iterId, epsilon]

    def modifiedPolicyIteration(self, initialPolicy, initialV, nEvalIterations=5, nIterations=np.inf, tolerance=0.01):
        '''修改的策略迭代程序: 在部分策略评估 (repeat a few times V^pi <-- R^pi + gamma T^pi V^pi)
        和策略改进(pi <-- argmax_a R^a + gamma T^a V^pi)之间多次迭代

        Inputs:
        initialPolicy -- 初始策略: 大小为|S|的array
        initialV -- 初始的值函数: 大小为|S|的array
        nEvalIterations -- 每次部分策略评估时迭代次数的限制: 标量 (默认值: 5)
        nIterations -- 修改的策略迭代中迭代次数的限制: 标量 (默认值: inf)
        tolerance -- ||V^n-V^n+1||_inf的阈值: scalar (默认值: 0.01)

        Outputs: 
        policy -- 策略: 大小为|S|的array
        V --值函数: 大小为|S|的array
        iterId -- 修改后策略迭代执行的迭代次数: 标量
        epsilon -- ||V^n-V^n+1||_inf: 标量'''

        iterId = 0
        epsilon = 0
        policy = initialPolicy
        V = initialV
        while iterId < nIterations:
            # 从policy和V解析最合适的V
            V_, _, _ = self.evaluatePolicyPartially(
                policy, V, nEvalIterations, tolerance)
            # 从V_获得最合适的policy
            policy_ = self.extractPolicy(V_)
            # 策略改进(pi <-- argmax_a R^a + gamma T^a V^pi)
            V__ = np.max(self.R + self.discount * (self.T.dot(V_)), axis=0)
            epsilon = np.sum(np.fabs(V-V__))
            policy = policy_
            if epsilon < tolerance:
                V = V__
                break
            V = V__
            iterId += 1

        return [policy, V, iterId, epsilon]
