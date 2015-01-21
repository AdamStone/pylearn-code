from pylearn2.config import yaml_parse
import numpy as np
import os

outputs_path = 'validate_outputs'

variables = ['learning_rate', 'conv1_kernel', 'conv2_kernel']
ranges = [[0.03, 0.01],
          [7, 5, 3],
          [5, 3]]
          

if __name__ == '__main__':
    
    trainfile = open('validate_template.yaml', 'r').read()
    
    base_params = { 'trainfile': '64px_rot_mirr_crop.train.hdf5',
                    'validfile': '64px_rot_mirr_crop.valid.hdf5',
                    
                    'irange': .005,
                    'istdev': .05,
                    
                    'max_kernel_norm': 1.,
                    'max_norm': 3.,
                    
                    'learning_rate': .01,
                    'init_momentum': 0.5,
                    
                    'conv1_channels': 96,
                    'conv1_kernel': 5,
                    'conv1_pshape': 2,
                    'conv1_pstride': 2,

                    'conv2_channels': 128,
                    'conv2_kernel': 3,
                    'conv2_pshape': 2,
                    'conv2_pstride': 2,

                    'conv3_channels': 256,
                    'conv3_kernel': 3,
                    'conv3_pshape': 1,
                    'conv3_pstride': 1,

                    'conv4_channels': 512,
                    'conv4_kernel': 3,
                    'conv4_pshape': 2,
                    'conv4_pstride': 2,

                    'fc1_dim': 2048,
                    'fc2_dim': 1024,

                    'fc_dropout': 0.5,

                    'lr_shrink': 0.95,
                    'lr_grow': 1.04,

                    'max_epochs': 5
    }

    meshgrid = np.meshgrid(*ranges)
    meshgrid = [coord.flatten() for coord in meshgrid]
    for i in range(len(meshgrid[0])):
        params = base_params.copy()
        save_path = []
        for j, var in enumerate(variables):
            params[var] = meshgrid[j][i]
            save_path.append('{} {}'.format(var, meshgrid[j][i]))
        params['savepath'] = os.path.join(outputs_path, '  '.join(save_path))

        train = trainfile % (params)

        train = yaml_parse.load(train)
        try:
            train.main_loop()
        except Exception:
            print Exception
