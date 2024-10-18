import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from multiprocessing import Pool, cpu_count
import matplotlib.pyplot as plt
import time


class GrangerCausalityBasedBrainNetwork:

    def __init__(self, subject_data, sparsify=False, p_value_threshold=0.05):
        self.subject_data = subject_data.diff().dropna()
        self.max_lag = len(subject_data)
        self.variables = self.subject_data.columns
        self.sparsify = sparsify
        self.p_value_threshold = p_value_threshold
        self.n = len(self.variables)
        self.causality_matrix = np.zeros((self.n, self.n))

    def _compute_causality(self, indices):
        i, j = indices
        test_result = grangercausalitytests(self.subject_data[[self.variables[i], self.variables[j]]], 
                                            self.max_lag, 
                                            verbose=False)  # Set verbose to False

        # Find the lag with the best test result (lowest p-value)
        best_test_result_lag, best_test_result_description = min(test_result.items(), 
                                                                 key=lambda x: x[1][0]['ssr_ftest'][1])

        F_score = best_test_result_description[0]['ssr_ftest'][0]
        p_value = best_test_result_description[0]['ssr_ftest'][1]

        return i, j, (F_score, p_value)

    def generate_connectivity_matrix(self):
        indices = [(i, j) for i in range(self.n) for j in range(self.n)]

        with Pool(cpu_count()) as pool:
            results = pool.map(self._compute_causality, indices)

        for i, j, (F_score, p_value) in results:
            if self.sparsify:
                if p_value < self.p_value_threshold:
                    self.causality_matrix[i, j] = F_score
                else:
                    self.causality_matrix[i, j] = 0
            else:
                self.causality_matrix[i, j] = F_score

        self.causality_matrix = pd.DataFrame(self.causality_matrix, 
                                             index=self.variables, 
                                             columns=self.variables)

        return self.causality_matrix

    def plot_correlation_matrix(self, title='Correlation Matrix'):
        plt.figure(figsize=(10, 8))
        plt.imshow(self.causality_matrix, vmin=-1, vmax=1, cmap='coolwarm')
        plt.colorbar()
        plt.title(title)
        plt.show()