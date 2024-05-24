import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os

folder = "results"
stop = 2050
dream_vals = [0, 1]
n_rep = [10, 10]
step_vals = [0, 0]
n_in = [0, 0]
labels = ['no dreaming', 'dreaming']

cmap = plt.get_cmap('Blues')
cmap2 = plt.get_cmap('RdPu')
colors = [cmap(0.8), cmap2(0.8)]

fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.5))

font_label = 8
font_tickle = 7
font_legend = 7

final_rew = []
final_rew_sem = []

for count, (if_dream, n_pred_steps) in enumerate(zip(dream_vals, step_vals)):
    print(n_pred_steps)

    REWARDS = []
    for repetitions in range(n_in[count], n_rep[count]):
        reward_file = os.path.join(folder, f"rewards_{repetitions}if_dream_{if_dream}.npy")
        reward = np.load(reward_file)
        REWARDS.append(savgol_filter(reward[:stop//50 - 1], 9, 3))

    mean_reward = np.mean(np.array(REWARDS), axis=0)
    sem = np.std(np.array(REWARDS), axis=0) / np.sqrt(n_rep[count])

    final_rew.append(mean_reward[-1])
    final_rew_sem.append(sem[-1])

    print(final_rew)

    # Plot mean and standard deviation
    ax.fill_between(np.arange(start=50, stop=stop, step=50), mean_reward+sem, mean_reward-sem, color=colors[count], alpha=0.3)
    ax.plot(np.arange(start=50, stop=stop, step=50), mean_reward, linestyle='dashed', linewidth=2.0, color=colors[count], label=labels[count])

    # Plot 80th percentile
    prct_80 = np.percentile(np.array(REWARDS), 80, axis=0)
    yhat = savgol_filter(prct_80, 9, 3)
    ax.plot(np.arange(start=50, stop=stop, step=50), yhat, linestyle='solid', linewidth=2.0, color=colors[count])

plt.plot([0, stop], [0, 0], 'k--', linewidth=1.0)

ax.spines["top"].set_visible(False)

ax.set_xlim(left=0, right=stop)
ax.set_ylim(bottom=-2.0, top=1)

ax.set_xlabel('frames (x100)', fontsize=font_label)
ax.set_ylabel('average return', fontsize=font_label)

ax.tick_params(axis='both', labelsize=font_tickle)

plt.tight_layout()
fig.savefig(folder + '/comparison_10.pdf', facecolor='w', edgecolor='w')
