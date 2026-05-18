import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Apply professional, clean styling suitable for academic papers
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'xtick.labelsize': 10, 'ytick.labelsize': 10, 'figure.titlesize': 14,
    'grid.alpha': 0.4, 'grid.linestyle': '--'
})

# ==========================================
# 1. PREPARE THE DATASETS
# ==========================================

# Historic timeline reflecting the Scaler introduction & correct dataset traits
timeline_data = pd.DataFrame({
    'Run': ['Run 1', 'Run 2', 'Run 3', 'Run 4', 'Run 5', 'Run 6', 'Run 7', 'Run 8\n(Uncapped)', 'Run 9\n(1M Cap)'],
    'F1-Score':  [0.6952, 0.8172, 0.7278, 0.7633, 0.9489, 0.9407, 0.9459, 0.9463, 0.9568],
    'MCC':       [0.6884, 0.7902, 0.6822, 0.6932, 0.9419, 0.9402, 0.9372, 0.9457, 0.9502],
    'Precision': [0.7602, 0.7353, 0.7111, 0.7948, 0.9879, 0.9927, 0.9548, 0.9925, 0.9790],
    'Recall':    [0.6404, 0.9197, 0.7454, 0.7341, 0.9129, 0.8939, 0.9372, 0.9042, 0.9355],
    'FP':        [80386, 131694, 72299, 45248, 4433, 2618, 8828, 2717, 3986],
    'FN':        [143048, 31941, 60776, 63457, 34649, 42212, 12499, 38102, 12820]
})

run9_grid = pd.DataFrame([
    {'loss': 'perceptron', 'alpha': 1e-06, 'MCC': 0.9375},
    {'loss': 'perceptron', 'alpha': 1e-05, 'MCC': 0.9357},
    {'loss': 'perceptron', 'alpha': 5e-05, 'MCC': 0.9420},
    {'loss': 'perceptron', 'alpha': 0.0001, 'MCC': 0.9356},
    {'loss': 'perceptron', 'alpha': 0.0002, 'MCC': 0.9399},
    {'loss': 'squared_hinge', 'alpha': 1e-06, 'MCC': 0.9305},
    {'loss': 'squared_hinge', 'alpha': 1e-05, 'MCC': 0.9502}, 
    {'loss': 'squared_hinge', 'alpha': 5e-05, 'MCC': 0.9201},
    {'loss': 'squared_hinge', 'alpha': 0.0001, 'MCC': 0.9306},
    {'loss': 'squared_hinge', 'alpha': 0.0002, 'MCC': 0.8897},
])

# ==========================================
# GRAPH 1: GLOBAL PERFORMANCE TRAJECTORY
# ==========================================
plt.figure(figsize=(10, 5.5))
plt.plot(timeline_data['Run'], timeline_data['F1-Score'], marker='o', linewidth=2.5, label='F1-Score', color='#1f77b4')
plt.plot(timeline_data['Run'], timeline_data['MCC'], marker='s', linewidth=2.5, label='MCC', color='#2ca02c')
plt.plot(timeline_data['Run'], timeline_data['Precision'], marker='^', linestyle=':', linewidth=1.5, label='Precision', color='#ff7f0e', alpha=0.8)

# Unified architectural shift boundary
plt.axvline(x=4, color='#d62728', linestyle='--', linewidth=1.5)
plt.text(3.85, 0.55, 'Architectural Pivot:\n+ Standard Scaler\n+ Metric Focus to MCC', color='#d62728', weight='bold', fontsize=10, ha='right')

plt.title('Global Performance Trajectory Across Engineering Iterations')
plt.xlabel('Experiment Sequence Sequence')
plt.ylabel('Score Value (0.0 - 1.0)')
plt.ylim(0.50, 1.02)
plt.legend(loc='lower right', frameon=True, facecolor='white')
plt.tight_layout()
plt.savefig('fig1_global_trajectory.pdf', dpi=300)
plt.show()

# ==========================================
# GRAPH 2: ERROR FOOTPRINT TRAJECTORY (LOG-SCALE)
# ==========================================
plt.figure(figsize=(10, 5))
x = np.arange(len(timeline_data['Run']))
width = 0.35

plt.bar(x - width/2, timeline_data['FP'], width, label='False Positives (FP)', color='#e377c2', alpha=0.85)
plt.bar(x + width/2, timeline_data['FN'], width, label='False Negatives (FN)', color='#9467bd', alpha=0.85)

plt.yscale('log')
plt.title('Classification Error Footprint Drop (Log Scale)')
plt.xlabel('Experiment Sequence Sequence')
plt.ylabel('Log Absolute Count of Errors')
plt.xticks(x, timeline_data['Run'])
plt.legend(frameon=True, facecolor='white')
plt.tight_layout()
plt.savefig('fig2_error_footprint.pdf', dpi=300)
plt.show()

# ==========================================
# GRAPH 3: PEAK PARAMETER SENSITIVITY HEATMAP
# ==========================================
plt.figure(figsize=(8, 4.5))
pivot_grid = run9_grid.pivot(index='loss', columns='alpha', values='MCC')
alpha_labels = [1e-06, 1e-05, 5e-05, 0.0001, 0.0002]
pivot_grid = pivot_grid[alpha_labels]

sns.heatmap(pivot_grid, annot=True, fmt=".4f", cmap="viridis", vmin=0.88, vmax=0.96,
            cbar_kws={'label': 'Matthews Correlation Coefficient (MCC)'})

plt.title('Hyperparameter Sensitivity Surface (Run 9 Grid Search)')
plt.xlabel(r'Regularization Strength Penalty ($\alpha$)')
plt.ylabel('Loss Objective Function')
plt.tight_layout()
plt.savefig('fig3_param_sensitivity.pdf', dpi=300)
plt.show()
