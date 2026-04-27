import numpy as np
from data_analysis.pls_analysis import pls_analysis
import pandas as pd
import matplotlib.pyplot as plt

def k_fold_optimization(X, Y, names, num_steps=20, num_repetitions=5, plot=False):
    # Performs k-fold cross-validation optimization for a range of k values and averages the results over multiple repetitions.

    q2_list = []
    time_list = []

    for rep in range(num_repetitions):
        q2_rep = []
        time_rep = []
        step = max(1, len(X) // num_steps)  # Ensure step is at least 1
        cv_range = list(range(len(X), 10, -step)) 
        
        for cv in cv_range:
            # Randomly shuffle the data and split into k folds
            np.random.seed(rep)  # Ensure different splits for each repetition
            shuffled_indices = np.random.permutation(len(X))
            
            # Assuming the pls_analysis function can accept shuffled data
            z = pls_analysis(names.iloc[shuffled_indices], X.iloc[shuffled_indices], Y.iloc[shuffled_indices], len(X.columns), cv, sig_figs=3, print_model=False)
            
            q2_rep.append(z['score_cv_normalized'])
            time_rep.append(z['execution_time'])
        
        q2_list.append(q2_rep)
        time_list.append(time_rep)

    # Averaging results across repetitions
    avg_q2_list = np.mean(q2_list, axis=0)
    avg_time_list = np.mean(time_list, axis=0)

    df = pd.DataFrame({
    'cv_range': cv_range,
    'q2': avg_q2_list,
    'time': avg_time_list
    })

    if plot:
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
    
    return df