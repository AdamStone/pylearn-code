import os
import gc
from pylearn2.utils import serial
import matplotlib.pyplot as plt


metrics = ['learning_rate', 'conv1_kernel', 'conv2_kernel']
plot_channel = 'valid_y_nll'
outputs_path = 'validate_outputs'


if __name__ == '__main__':
        
    best_results = {}

    for root, folders, files in os.walk(outputs_path):
        for filename in files:
            with open(os.path.join(outputs_path, filename), 'r') as readfile:
                params = [param.split(' ') for param in filename.split('  ')]
                results = readfile.read()
                results = [line.split(':\t\t') for line in results.split('\n')]
                for i, (channel, value) in enumerate(results):
                    if channel in best_results:
                        if float(best_results[channel][1][i][1]) > float(value):
                            best_results[channel] = (params, results)
                    else:
                        best_results[channel] = (params, results)

    for channel, (params, results) in best_results.items():
        print 'minimum {}:'.format(channel)
        for result in results:
            print ' '.join(result)    
        for param in params:
            print ' '.join(param)
        print '\n'
                    
    def get_scatter(parameter, channel):
        coords = []
        for root, folders, files in os.walk(outputs_path):
            for filename in files:
                with open(os.path.join(outputs_path, filename), 'r') as readfile:
                    params = [param.split(' ') for param in filename.split('  ')]
                    results = readfile.read()
                    results = [line.split(':\t\t') for line in results.split('\n')]
                    for param, pval in params:
                        if parameter == param:
                            for ch, cval in results:
                                if ch == channel:
                                    coords.append((pval, cval))
        return coords

    if plot_channel:
        all_coords = [get_scatter(metric, plot_channel) for metric in metrics]

        fig, ax = plt.subplots(1, len(all_coords), figsize=(16,5))

        for i, coords in enumerate(all_coords):
            x, y = zip(*coords)
            ax[i].scatter(x, y, alpha=0.2)
            ax[i].set_title(metrics[i])

        plt.show()
