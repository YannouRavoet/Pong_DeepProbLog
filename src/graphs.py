import pandas as pd
import matplotlib.pyplot as plt

# DATA
df = pd.read_csv('experiment_results/dataset_20000/training_loss_and_accuracy.csv')
df.DeepProblog_Accuracy = df.DeepProblog_Accuracy.interpolate()
df.PyTorch_Accuracy = df.PyTorch_Accuracy.interpolate()

# PLOTTING
ax = df.plot(x='Iterations', y=['DeepProblog_Loss', 'PyTorch_Loss'], legend=False, color=['orange', 'red'], alpha=0.8)
ax2 = ax.twinx()
df.plot(x='Iterations', y=['DeepProblog_Accuracy', 'PyTorch_Accuracy'], ax=ax2, legend=False, color=['blue', 'green'])

# LEGENDS AND TITLES
ax.set_xlabel('Iterations', fontdict={'size': 12})
ax.set_ylabel('Loss', fontdict={'size': 12})
ax2.set_ylabel('Accuracy', fontdict={'size': 12})
ax2.set_ylim(0, 1.025)


# STYLING
ax.grid(False)
ax2.grid(False)


# SHOW
plt.show()
