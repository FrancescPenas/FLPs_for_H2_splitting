import sys
import os
# Setup paths
sys.path.append(os.path.join(os.getcwd(), 'cdf francesc', 'FLPs', 'Python', 'code'))
os.chdir(os.path.join(os.getcwd(), 'cdf francesc', 'FLPs', 'Python', 'code'))
from utility_functions_v2 import *
import numpy as np
from pls_analysis_v15 import pls_analysis
import matplotlib.pyplot as plt

data = from_pkl('final_data_expanded.pkl')
names = data.iloc[:, 0]
X = normalize_dataframe(data.iloc[:, 5:])
Y = data.iloc[:, 1]

# Number of repetitions for random assignment
num_repetitions = 5

q2_list = []
time_list = []

for rep in range(num_repetitions):
    q2_rep = []
    time_rep = []
    cv_range = list(range(112, 10, -5))
    
    for cv in cv_range:
        # Randomly shuffle the data and split into k folds
        np.random.seed(rep)  # Ensure different splits for each repetition
        shuffled_indices = np.random.permutation(len(X))
        
        # Assuming the pls_analysis function can accept shuffled data
        z = pls_analysis(names.iloc[shuffled_indices], X.iloc[shuffled_indices], Y.iloc[shuffled_indices], 38, cv)
        
        q2_rep.append(z['score_cv_normalized'])
        time_rep.append(z['execution_time'])
    
    q2_list.append(q2_rep)
    time_list.append(time_rep)

# Averaging results across repetitions
avg_q2_list = np.mean(q2_list, axis=0)
avg_time_list = np.mean(time_list, axis=0)

# Plotting

fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot averaged R2 and Q2 on the primary y-axis
ax1.plot(cv_range, avg_q2_list, label='q$^2$', marker='s', color='orange')

ax1.set_xlabel('number of k-folds')
ax1.set_ylabel('q$^2$')
ax1.set_title('Averaged q$^2$ and Time over 5 repetitions')
ax1.legend(loc='upper left')
ax1.grid(True)

# Create secondary y-axis for Time
ax2 = ax1.twinx()
ax2.plot(cv_range, avg_time_list, label='Time', marker='^', color='g')
ax2.set_ylabel('Time (s)', color='g')
ax2.tick_params(axis='y', labelcolor='g')
ax2.legend(loc='upper right')

plt.show()