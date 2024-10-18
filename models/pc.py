from causalai.models.time_series.pc import PC
from causalai.models.common.CI_tests.partial_correlation import PartialCorrelation
from causalai.models.common.CI_tests.kci import KCI
from causalai.data.time_series import TimeSeriesData
from causalai.data.transforms.time_series import StandardizeTransform
from causalai.misc.misc import plot_graph
import time
import yaml
from rich.console import Console


console = Console()

with open('config.yaml', 'r', encoding='utf-8-sig') as file:
    config = yaml.safe_load(file)
    
    
class PCCausality:
    
    def __init__(self, data, max_lag=10, p_value_threshold=0.05, multiprocess=True) -> None:
        
        self.data = data
        self.max_lag = max_lag
        self.p_value = p_value_threshold
        self.multiprocess = multiprocess
        self.estimates = None
    
    @staticmethod
    def __preprocessing(data):
        
        StandardizeTransform_ = StandardizeTransform()
        StandardizeTransform_.fit(data.values)

        transformed_data = StandardizeTransform_.transform(data.values)
        data_object = TimeSeriesData(transformed_data, 
                                     var_names=list(data.columns))
        return data_object
    
    def fit(self, max_iter=1000, expected_causal_relations='linear', chunk_size=100):
        
        data_object = self.__preprocessing(self.data)
        
        if expected_causal_relations == 'linear':
            CI_test = PartialCorrelation()
        else:
            CI_test = KCI(chunk_size=chunk_size)
        
        pc = PC(
            data=data_object,
            prior_knowledge=None,
            max_iter=max_iter,
            CI_test=CI_test,
            use_multiprocessing=self.multiprocess
        )
        
        start = time.time()
        result = pc.run(pvalue_thres=self.p_value, 
                        max_lag=self.max_lag)
        end = time.time()
        
        console.log(f'Time taken: {end - start:.2f}s\n')
        
        self.estimates = dict(zip(result.keys(), [item['parents'] for item in result.values()]))
        
        return self.estimates
    
    def plot_GCM(self, filename='pc_gcm.png'):
        plot_graph(self.estimates, filename=filename, node_size=1000)
        