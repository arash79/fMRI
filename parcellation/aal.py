import nilearn.datasets as datasets
from nilearn import image
from nilearn.input_data import NiftiLabelsMasker
from nilearn import datasets, plotting, image, surface
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd


class AAL:

    def __init__(self, data_paths):

        self.data_paths = data_paths
        self.atlas = datasets.fetch_atlas_aal()
    
        self.data = self.__load_data()

    def __load_data(self):
        return {file_path.split('/')[6]: image.load_img(file_path) for file_path in self.data_paths}
    
    def parcellate(self):
        
        labels = self.atlas['labels']
        maps = self.atlas['maps']

        masker = NiftiLabelsMasker(labels_img=maps, standardize=True, detrend=True)
        time_series = {subject: dict(zip(labels, masker.fit_transform(MRI).T)) for subject, MRI in self.data.items()}
        
        coordinates = plotting.find_parcellation_cut_coords(labels_img=maps)

        return {subject: (pd.DataFrame.from_dict(MRI), coordinates) for subject, MRI in time_series.items()}

    def plot_parcellations_using_matplotlib(self, view='lateral'):

        """
        view: can be one of ('anterior', 'posterior', 'medial', 'lateral', 'dorsal', 'ventral').
        """

        atlas_filename = self.atlas['maps']

        fsaverage = datasets.fetch_surf_fsaverage()

        texture_left = surface.vol_to_surf(atlas_filename, fsaverage.pial_left)
        texture_right = surface.vol_to_surf(atlas_filename, fsaverage.pial_right)

        title = f'AAL Atlas - Parcellated Regions'

            
        fig, axes = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(12, 6))

        hemispheres = ['left', 'right']
        textures = [texture_left, texture_right]
        cmap = 'tab20'

        for ax, hemi, texture in zip(axes, hemispheres, textures):

            mesh = fsaverage.infl_left if hemi == 'left' else fsaverage.infl_right
            bg_map = fsaverage.sulc_left if hemi == 'left' else fsaverage.sulc_right

            plotting.plot_surf_roi(
                mesh,
                roi_map=texture,
                hemi=hemi,
                view=view,
                cmap=cmap,
                colorbar=False,
                bg_map=bg_map,
                darkness=0.6,
                axes=ax,
                title=f'{hemi.capitalize()} Hemisphere - {view.capitalize()} View'
                )
            
            ax.dist = 7

        plt.suptitle(title, fontsize=20)

        plt.close()
        
        return fig
    
    def plot_parcellations_using_plotly(self):

        atlas_filename = self.atlas['maps']

        fsaverage = datasets.fetch_surf_fsaverage()
        surf_mesh_left = fsaverage['infl_left']
        surf_mesh_right = fsaverage['infl_right']

        atlas_img_left = surface.vol_to_surf(atlas_filename, surf_mesh_left)
        atlas_img_right = surface.vol_to_surf(atlas_filename, surf_mesh_right)

        title = 'AAL Atlas - Parcellated Regions'

        fig = go.Figure()

        hemispheres = ['left', 'right']
        meshes = [surf_mesh_left, surf_mesh_right]
        atlas_imgs = [atlas_img_left, atlas_img_right]
        colorscale = 'turbo'

        for hemi, mesh, atlas_img in zip(hemispheres, meshes, atlas_imgs):
            mesh_data = surface.load_surf_mesh(mesh)
            x, y, z = mesh_data[0].T
            faces = mesh_data[1]

            if hemi == 'left':
                x = x - np.max(x) - 10  
            else:
                x = x + 10 

            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z, 
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                intensity=atlas_img,
                colorscale=colorscale,
                name=f'{hemi.capitalize()} Hemisphere',
                showlegend=True,
                showscale=hemi == 'left',  # Show the color bar for the left hemisphere only
                colorbar=dict(title='Intensity', len=0.75, x=1) if hemi == 'left' else None
            ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis=dict(showbackground=False, visible=False),
                yaxis=dict(showbackground=False, visible=False),
                zaxis=dict(showbackground=False, visible=False),
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            scene_aspectmode='data'
        )

        return fig
    