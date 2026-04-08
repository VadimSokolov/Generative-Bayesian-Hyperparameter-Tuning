"""Generate MNIST 20-replicate summary figure."""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

files = sorted(glob.glob('code/results/mnist_rep*.csv'))
df = pd.concat([pd.read_csv(f) for f in files])
print(f"{len(df)} reps loaded")

fig, axes = plt.subplots(1, 2, figsize=(9, 4))

# Panel 1: Test accuracy (generator vs baseline)
ax = axes[0]
jitter = 0.15
ax.scatter(np.ones(len(df)) - jitter, df['test_acc_gen'], c='gray', alpha=0.5, s=40, zorder=2)
ax.scatter(np.ones(len(df))*2 + jitter, df['baseline_acc'], c='gray', alpha=0.5, s=40, marker='^', zorder=2)
ax.scatter(1 - jitter, df['test_acc_gen'].mean(), c='blue', s=120, zorder=3, label='Mean')
ax.scatter(2 + jitter, df['baseline_acc'].mean(), c='blue', s=120, marker='^', zorder=3)
ax.set_xticks([1, 2])
ax.set_xticklabels(['Generator', 'Baseline'])
ax.set_ylabel('Test accuracy')
ax.legend(loc='lower left')

# Panel 2: Selected lambda
ax = axes[1]
ax.scatter(range(1, len(df)+1), df['best_lambda'], c='black', s=40)
ax.axhline(df['best_lambda'].median(), color='blue', lw=2, label=f"Median = {df['best_lambda'].median():.1e}")
ax.set_xlabel('Replication')
ax.set_ylabel(r'Selected $\lambda$')
ax.set_yscale('log')
ax.legend()

plt.tight_layout()
plt.savefig('fig/mnist_replicates.pdf')
print("Wrote fig/mnist_replicates.pdf")
