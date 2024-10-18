from loader import Loader
from parcellation import AAL
from measures.correlation import CorrelationBasedBrainNetwork
from measures.covariance import CovarianceBasedBrainNetwork
from statsmodels.tsa.stattools import grangercausalitytests
from itertools import combinations
import random
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from IPython.display import clear_output
import numpy as np
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import time


class GrangerCausalityBasedBrainNetwork:

    def __init__(self, subject_data, max_lag=5, sparsify=False, p_value_threshold=0.05):

        self.subject_data = subject_data.diff().dropna()
        self.max_lag = max_lag
        self.variables = subject_data.columns
        self.sparsify = sparsify
        self.p_value_threshold = p_value_threshold
        self.n = len(self.variables)

        self.causality_matrix = np.zeros((self.n, self.n))

    def _compute_causality(self, indices):

        i, j = indices
        test_result = grangercausalitytests(self.subject_data[[self.variables[i], self.variables[j]]], 
                                            self.max_lag, 
                                            verbose=False)
        
        best_test_results = min([(key, value) for key, value in test_result.items()], 
                                key=lambda x: x[1][0]['ssr_ftest'][1])
        
        best_test_result_lag, best_test_result_description = best_test_results[0], best_test_results[1]

        F_score = best_test_result_description[0]['ssr_ftest'][0]
        p_value = best_test_results[0]['ssrf_test'][1]

        return i, j, (F_score, p_value)

    def generate_connectivity_matrix(self):

        indices = [(i, j) for i in range(self.n) for j in range(self.n)]
        print(indices)

        results = []
        with ThreadPoolExecutor() as executor:
            future_to_index = {executor.submit(self._compute_causality, index): index for index in indices}
            for future in as_completed(future_to_index):
                result = future.result()
                results.append(result)

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


group = 'SCZ'
modality = 'BOLD'
task = 'task001_run001'
kind = 'bold_mcf_brain'

agent = Loader()
data = agent.filter_data(group=group, modality=modality, task=task, kind=kind)
agent = AAL(data)
parcellation_results = agent.parcellate()

subject = parcellation_results['sub001'][0]

causation = GrangerCausalityBasedBrainNetwork(subject).generate_connectivity_matrix()
