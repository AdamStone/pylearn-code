from pylearn2.train_extensions.best_params import MonitorBasedSaveBest
from pylearn2.utils.timing import log_timing
import os

class MonitorBasedSaveSummary(MonitorBasedSaveBest):
    def __init__(self, monitor_channel, extra_channels=[], save_path=None,
    store_best_model=False, higher_is_better=False, tag_key=None):

        self.summary_channels = [monitor_channel] + extra_channels

        super(MonitorBasedSaveSummary, self).__init__(monitor_channel, save_path,
            store_best_model, higher_is_better, tag_key)

    def on_monitor(self, model, dataset, algorithm):
        """
        Looks whether the model performs better than earlier. If it's the
        case, saves the value of monitor_channel and extra_channels.

        Parameters
        ----------
        model : pylearn2.models.model.Model
            model.monitor must contain a channel with name given by
            self.channel_name
        dataset : pylearn2.datasets.dataset.Dataset
            Not used
        algorithm : TrainingAlgorithm
            Not used
        """
        monitor = model.monitor
        channels = monitor.channels
        channel = channels[self.channel_name]
        val_record = channel.val_record
        new_cost = val_record[-1]

        if self.coeff * new_cost < self.coeff * self.best_cost:
            self.best_cost = new_cost
            self._update_tag(model)
            if self.store_best_model:
                self.best_model = deepcopy(model)
            if self.save_path is not None:
                print 'Saving to ' + self.save_path
                with open(self.save_path, 'w') as outfile:
                    writeline = []
                    for channel in self.summary_channels:
                        writeline.append('{}:\t\t{}'.format(channel,
                            monitor.channels[channel].val_record[-1]))
                    outfile.write('\n'.join(writeline))
