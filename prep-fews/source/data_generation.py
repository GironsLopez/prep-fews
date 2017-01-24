# -*- coding: utf-8 -*-
###############################################################################
# PREP-FEWS
# Social Preparedness in Early Warning Systems model
#
# Synthetic hydrological data generator
#
# Marc Girons Lopez, Giuliano di Baldassarre, Jan Seibert
###############################################################################

import os
import numpy as np
import pandas as pd

import funcitons as fun


class HydrologicalSystem(object):
    """Define the characteristics of the climatic variability of events that
    can potentially produce floods.
    """
    def __init__(self, file_conf='config_data.yml'):

        params = fun.load_parameters(file_conf)

        # set parameters
        self.shape = params['Gamma']['shape']
        self.scale = params['Gamma']['scale']
        self.length = params['length']
        self.ts = None

    def generate_event(self):
        """Generate a random event based on a gamma distribution function.

        Returns a float with the event magnitude.
        """
        return np.random.gamma(shape=self.shape, scale=self.scale)

    def generate_time_series(self):
        """Generate a set of random events based on a gamma distribution
        function.

        Returns a list of floats with size=length representing the magnitude
        of the events.
        """
        data = np.random.gamma(shape=self.shape, scale=self.scale,
                               size=self.length)

        index = range(self.length)
        self.ts = pd.DataFrame(data=data, index=index, columns=['Runoff'])

    def write_time_series(self, filename_data='time_series.txt'):
        path_data = os.getcwd() + '\\..\\config\\'
        if not os.path.exists(path_data):
            os.makedirs(path_data)

        self.ts.to_csv(path_data + filename_data, index_label='Index')

if __name__ == '__main__':

    flows = HydrologicalSystem(file_conf='config_data.yml')
    flows.generate_time_series()
    flows.write_time_series(filename_data='time_series.txt')
