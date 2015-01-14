from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset
from pylearn2.space import CompositeSpace, Conv2DSpace, VectorSpace
import numpy as np
import os

class CompositeDataset(VectorSpacesDataset):
    def __init__(self, data_path, topo_view, topo_view_2, y, rng=(17, 2, 946),
                 preprocessor=None, fit_preprocessor=False):

        topo1 = data_path + [topo_view]
        topo1 = np.load(os.path.join(*topo1))

        topo2 = data_path + [topo_view_2]
        topo2 = np.load(os.path.join(*topo2))

        Y = data_path + [y]
        Y = np.load(os.path.join(*Y))
        data = (topo1, topo2, Y)
        
        shape1 = topo1.shape[1:3]
        shape2 = topo2.shape[1:3]
        space = CompositeSpace(components=(
            Conv2DSpace(shape=topo1.shape[1:3],
                        num_channels=topo1.shape[-1]),
            Conv2DSpace(shape=topo2.shape[1:3],
                        num_channels=topo2.shape[-1]),
            VectorSpace(dim=Y.shape[-1])))
        source = ('features', 'extra_features', 'targets')
        data_specs = (space, source)

        super(CompositeDataset, self).__init__(data, data_specs, rng,
                                               preprocessor, fit_preprocessor)

        
