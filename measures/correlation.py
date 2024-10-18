import scipy.stats as statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CorrelationBasedBrainNetwork:

    def __init__(self, subject_data, sparsify=False, p_value_threshold=0.05):
        
        self.subject_dataframe = subject_data
        self.variables = subject_data.columns
        self.sparsify = sparsify
        self.p_value_threshold = p_value_threshold
        self.n = len(self.variables)

        self.correlation_matrix = np.empty((self.n, self.n))

    def __calculate_correlation(self, method):

        for i in range(self.n):
            for j in range(i, self.n):
                statistic, p_value = getattr(statistics, method)(self.subject_dataframe[self.variables[i]], 
                                                                 self.subject_dataframe[self.variables[j]])
                if self.sparsify:
                    if p_value < self.p_value_threshold:
                        self.correlation_matrix[i, j] = statistic
                        self.correlation_matrix[j, i] = statistic
                    else:
                        self.correlation_matrix[i, j] = 0
                        self.correlation_matrix[j, i] = 0
                else:
                    self.correlation_matrix[i, j] = statistic
                    self.correlation_matrix[j, i] = statistic

        return self.correlation_matrix
    
    def generate_connectivity_matrix(self, method='pearsonr'):
        self.__calculate_correlation(method)
        self.correlation_matrix = pd.DataFrame(self.correlation_matrix, 
                                               index=self.variables, 
                                               columns=self.variables)
        return self.correlation_matrix
    
    def plot_correlation_matrix(self, title='Correlation Matrix'):
        plt.figure(figsize=(10, 8))
        plt.imshow(self.correlation_matrix, vmin=-1, vmax=1, cmap='coolwarm')
        plt.colorbar()
        plt.title(title)
        plt.show()
