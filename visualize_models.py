import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data from training results
models = ['LogReg', 'NB', 'SVM', 'Random Forest']
f1_scores = [1.0, 0.9584, 1.0, 1.0]
accuracy = [1.0, 0.9640, 1.0, 1.0]

# Set style
plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

# Colors for premium look
colors = ['#3498db', '#e74c3c', '#2ecc71', '#f1c40f']

# Plotting
x = np.arange(len(models))
width = 0.35

rects1 = ax.bar(x - width/2, accuracy, width, label='Accuracy', color='#5dade2', alpha=0.8)
rects2 = ax.bar(x + width/2, f1_scores, width, label='Macro F1', color='#58d68d', alpha=0.8)

# Add text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores (0.0 - 1.0)')
ax.set_title('Model Performance Comparison (Sentiment Analysis)', fontsize=16, pad=20, color='white')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.legend(loc='lower left')
ax.set_ylim(0.9, 1.02) # Focus on the high performance area

# Add labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('model_comparison.png', transparent=False)
print("Saved: model_comparison.png")
