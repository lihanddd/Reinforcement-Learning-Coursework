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
    curves_sql_1 = []
    curves_sql_10 = []
    curves_sql_100 = []
    curves_sql_1000 = []
    Soft_QLearning.LAMBDA = 1
    for seed in SEEDS:
        curves_sql_1 += [Soft_QLearning.train(seed)]
    Soft_QLearning.LAMBDA = 10
    for seed in SEEDS:
        curves_sql_10 += [Soft_QLearning.train(seed)]
    Soft_QLearning.LAMBDA = 100
    for seed in SEEDS:
        curves_sql_100 += [Soft_QLearning.train(seed)]
    Soft_QLearning.LAMBDA = 1000
    for seed in SEEDS:
        curves_sql_1000 += [Soft_QLearning.train(seed)]

    # Plot the curve for the given seeds
    plot_arrays(curves_sql_1, 'blue','1')
    plot_arrays(curves_sql_10, 'red','10')
    plot_arrays(curves_sql_100, 'green','100')
    plot_arrays(curves_sql_1000, 'brown', '1000')
    plt.legend(loc='best')
    plt.savefig('SQL_lambda.png')
    plt.show()