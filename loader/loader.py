import os
import glob
import yaml
import pandas as pd


with open('config.yaml', 'r', encoding='utf-8-sig') as file:
    config = yaml.safe_load(file)


class Loader:

    def __init__(self) -> None:
        
        self.study_groups = self.__metadata()
        self.available_data = self.__load_all_nifti_files()

        self.groups_names = [
            'CON', 
            'CON-SIB', 
            'SCZ', 
            'SCZ-SIB'
            ]

    @staticmethod
    def __metadata():
        metadata = pd.read_csv(config['PATHs']['METADATA'], delimiter='\t')
        study_groups = {group[0]: group[1]['subcode'].values.tolist() for group in metadata.groupby('condit')}
        # TODO: include gender and race into groupings
        return study_groups
    
    @staticmethod
    def __load_all_nifti_files(extension='*.nii.gz'):
        
        nifti_files = [
            file
            for path, subdir, files in os.walk(config['PATHs']['BASE_PATH'])
            for file in glob.glob(os.path.join(path, extension))
            ]
        
        return nifti_files
    
    def filter_data(self, group, **kwargs):
        subjects = self.study_groups[group]

        results = list(filter(lambda data: data.split('/')[6] in subjects, self.available_data))

        if kwargs.get('modality') is not None:
            results = list(filter(lambda data: data.split('/')[7] == kwargs['modality'], results))

        if kwargs.get('task') is not None:
            results = list(filter(lambda data: data.split('/')[8] == kwargs['task'], results))

        if kwargs.get('kind') is not None:

            kind = kwargs.get('kind')

            if kind == 'bold':
                results = list(filter(lambda data: 'bold.nii.gz' in data.split('/')[-1], results))   
            if kind == 'bold_mcf':
                results = list(filter(lambda data: 'bold_mcf.nii.gz' in data.split('/')[-1], results))
            if kind == 'bold_mcf_brain':
                results = list(filter(lambda data: 'bold_mcf_brain.nii.gz' in data.split('/')[-1], results))
            if kind == 'bold_mcf_brain_mask': 
                results = list(filter(lambda data: 'bold_mcf_brain_mask.nii.gz' in data.split('/')[-1], results))
                
        return results
    