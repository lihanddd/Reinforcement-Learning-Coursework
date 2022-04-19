import DQN
import DQN_without_target_net
import SAC
import Soft_QLearning
import numpy as np
import matplotlib.pyplot as plt

SEEDS = [1, 2, 3, 4, 5]
def plot_arrays(vars, color, label):
    mean = np.mean(vars, axis=0)
    std = np.std(vars, axis=0)
    plt.plot(range(len(mean)), mean, color=color, label=label)
    plt.fill_between(range(len(mean)), np.maximum(mean - std, 0), np.minimum(mean + std, 200), color=color, alpha=0.3)

if __name__ == "__main__":

    # Train for different seeds
    curves_dqn = []
    curves_dqn_without_target_net = []
    curves_sac = []
    curves_sql = []
    for seed in SEEDS:
        curves_dqn += [DQN.train(seed)]
        curves_dqn_without_target_net += [DQN_without_target_net.train(seed)]
        curves_sac += [SAC.train(seed)]
        curves_sql += [Soft_QLearning.train(seed)]

    # Plot the curve for the given seeds
    plot_arrays(curves_dqn, 'blue','dpn')
    plot_arrays(curves_dqn_without_target_net, 'yellow','dpn_no_target')
    plot_arrays(curves_sac, 'red','sac')
    plot_arrays(curves_sql, 'green','sql')
    plt.legend(loc='best')
    plt.savefig('compare.png')
    plt.show()