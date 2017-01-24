# -*- coding: utf-8 -*-
###############################################################################
# PREP-FEWS
# Social Preparedness in Early Warning Systems model
#
# Various functions
#
# Marc Girons Lopez, Giuliano di Baldassarre, Jan Seibert
###############################################################################

import os
import yaml
import pandas as pd


def load_data(datafile):
    """Load a txt data file
    """
    filename = os.getcwd() + '\\..\\config\\' + datafile
    with open(filename, 'r') as txtfile:
        return pd.read_csv(txtfile, index_col=0)


def load_parameters(configfile):
    """Read a Yaml configuration file
    """
    filename = os.getcwd() + '\\..\\config\\' + configfile
    with open(filename, 'r') as ymlfile:
        return yaml.load(ymlfile)


def progress_bar(iteration, total, prefix='Progress', suffix='Complete',
                 decimals=1, length=40, fill='█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r')
    # Print New Line on Complete
    if iteration == total:
        print()
