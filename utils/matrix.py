import numpy as np
import pandas as pd

class MatrixManipulator:

    def __init__(self, association_matrix):

        self.matrix = association_matrix.values
        self.variables = association_matrix.columns
    
    def separate_positive_negative(self):
        positive_matrix = np.where(self.matrix > 0, self.matrix, 0)
        negative_matrix = np.where(self.matrix < 0, self.matrix, 0)

        positive_matrix = pd.DataFrame(positive_matrix, 
                                       columns=self.variables, 
                                       index=self.variables)
        negative_matrix = pd.DataFrame(negative_matrix, 
                                       columns=self.variables, 
                                       index=self.variables)

        return positive_matrix, negative_matrix
    
    def binarize(self, threshold):
        binary_matrix = np.where(self.matrix >= threshold, 1, 0)
        binary_matrix = pd.DataFrame(binary_matrix,
                                     columns=self.variables,
                                     index=self.variables)
        return binary_matrix
    