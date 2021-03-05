import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# DATA
df = pd.read_csv('experiment_results/CAPAI_Experiments.csv')
df.DeepProblog_Accuracy = df.DeepProblog_Accuracy.interpolate()
df.PyTorch_Accuracy = df.PyTorch_Accuracy.interpolate()

# PLOTTING
ax = df.plot(x='Iterations', y=['DeepProblog_Loss', 'PyTorch_Loss'], legend=False, color=['C1', 'C2'], alpha=0.8)
ax2 = ax.twinx()
df.plot(x='Iterations', y=['DeepProblog_Accuracy', 'PyTorch_Accuracy'], ax=ax2, legend=False, color=['C3', 'C4'])

# LEGENDS AND TITLES
ax.set_ylabel('Loss', fontdict={'size': 12})
ax2.set_ylabel('Accuracy', fontdict={'size': 12})


# STYLING
mpl.style.use('seaborn')
ax.set_facecolor('white')
ax2.set_facecolor('white')
ax.grid(False)
ax2.grid(False)


# SHOW
plt.show()
