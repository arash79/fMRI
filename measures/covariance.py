import sklearn
from nilearn.connectome import ConnectivityMeasure
import pandas as pd
import matplotlib.pyplot as plt


class CovarianceBasedBrainNetwork:

    def __init__(self, subject_data):
        
        self.subject_data = subject_data
        self.subject_dataframe = subject_data

        self.covariance_matrix = None
        self.brain_network = None
    
    @staticmethod
    def __select_covariance_estimators(estimator_name='LedoitWolf'):

        """
        Allowed Estimators:

        EmpiricalCovariance: Maximum likelihood covariance estimator.

        GraphicalLasso: Sparse inverse covariance estimation with an l1-penalized estimator.

        GraphicalLassoCV: Sparse inverse covariance w/ cross-validated choice of the l1 penalty.

        LedoitWolf: LedoitWolf Estimator.

        MinCovDet: Minimum Covariance Determinant (MCD): robust estimator of covariance.

        OAS: Oracle Approximating Shrinkage Estimator.

        ShrunkCovariance: Shrunk covariance estimator.

        EllipticEnvelope: An object for detecting outliers in a Gaussian distributed dataset.
        
        """

        return getattr(sklearn.covariance, estimator_name)()
    
    @staticmethod
    def __connectivity_measure(kind='partial correlation'):

        """

        Allowed connectivity measures:

        "covariance", "correlation", "partial correlation", "tangent", "precision"

        """

        return ConnectivityMeasure(kind=kind)

    def generate_covariance_matrix(self, 
                                     connectivity_kind='partial correlation', 
                                     estimator_name='LedoitWolf'):

        connectivity_measure = self.__connectivity_measure(connectivity_kind)
        estimator = self.__select_covariance_estimators(estimator_name)
        connectivity_measure.cov_estimator = estimator

        self.covariance_matrix = connectivity_measure.fit_transform([self.subject_dataframe.values])[0]

        return self.covariance_matrix

    def plot_covariance_matrix(self, title='Correlation Matrix'):
        plt.figure(figsize=(10, 8))
        plt.imshow(self.covariance_matrix, vmin=-1, vmax=1, cmap='coolwarm')
        plt.colorbar()
        plt.title(title)
        plt.show()
