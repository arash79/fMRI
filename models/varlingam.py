from causalai.models.time_series.var_lingam import VARLINGAM
from causalai.data.time_series import TimeSeriesData
from causalai.data.transforms.time_series import StandardizeTransform
from causalai.misc.misc import plot_graph
import time
import yaml
from rich.console import Console


console = Console()

with open('config.yaml', 'r', encoding='utf-8-sig') as file:
    config = yaml.safe_load(file)
    
    
class VARLINGAMCausality:
    
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
    
    def fit(self, max_iter=1000):
        
        data_object = self.__preprocessing(self.data)
        varlingam = VARLINGAM(
            data=data_object,
            prior_knowledge=None,
            max_iter=max_iter,
            use_multiprocessing=self.multiprocess
        )
        
        start = time.time()
        result = varlingam.run(pvalue_thres=self.p_value, 
                               max_lag=self.max_lag)
        end = time.time()
        
        console.log(f'Time taken: {end - start:.2f}s\n')
        
        self.estimates = dict(zip(result.keys(), [item['parents'] for item in result.values()]))
        
        return self.estimates
    
    def plot_GCM(self, filename='varlingam_gcm.png'):
        plot_graph(self.estimates, filename=filename, node_size=1000)
        