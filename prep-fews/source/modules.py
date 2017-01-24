# -*- coding: utf-8 -*-
###############################################################################
# PREP-FEWS
# Social Preparedness in Early Warning Systems model
#
# Model modules
#
# Marc Girons Lopez, Giuliano di Baldassarre, Jan Seibert
###############################################################################

import os
import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter

###############################################################################
# FORECASTING SYSTEM
###############################################################################


class ForecastingSystem(object):
    """Define the technical capabilities, in terms of precision and accuracy,
    of a flood forecasting system.
    """
    def __init__(self, acc_mean, acc_sd, prc_shape, prc_scale):
        """ Initialize the class with the parameter values
        """
        self.mean = acc_mean
        self.sd = acc_sd
        self.shape = prc_shape
        self.scale = prc_scale

    def issue_forecast(self, event):
        """Generate a random probabilistic flood forecast based on the
        following conceptualization: F=N(event+N(mu,sigma),gamma(shape,scale)).

        Returns a tuple with the mean and sd of the forecast.
        """
        deviation = np.random.normal(loc=self.mean, scale=self.sd)
        mean = event + deviation

        sd = np.random.gamma(shape=self.shape, scale=self.scale)

        return (mean, sd)

###############################################################################
# WARNING ASSESSMENT
###############################################################################


class WarningAssessment(object):
    """Assess the outcome of the flood warning system based on a simple
    contingency table drawing from decisions on the flood magnitude at and
    from which warnings are to be issued (mag_thr) and the required
    likelihood of the forecast for warning to be issued (prob_thr).
    """
    def __init__(self, mag_thr, prob_thr):
        """ Initialize the class with the parameter values
        """
        self.mag_thr = mag_thr
        self.prob_thr = prob_thr

    def assess_warning_outcome(self, event, forecast):
        """Define the contingency table used for evaluating the warnings.

        Returns a string describing the warning outcome
        """
        cdf = stats.norm(loc=forecast[0], scale=forecast[1]).cdf(self.mag_thr)
        aep = 1. - cdf  # annual exceedance probability

        if event < self.mag_thr and aep < self.prob_thr:
            return 'true negative'  # all clear

        elif event < self.mag_thr and aep >= self.prob_thr:
            return 'false positive'  # false alarm

        elif event >= self.mag_thr and aep < self.prob_thr:
            return 'false negative'  # missed event

        elif event >= self.mag_thr and aep >= self.mag_thr:
            return 'true positive'  # hit, successful alarm

###############################################################################
# LOSS ESTIMATION
###############################################################################


class LossEstimation(object):
    """Define how the different warning outcomes and preparedness level affect
    the losses incurred by the system
    """
    def __init__(self, dmg_thr, dmg_shape, res_dmg, mit_cst):
        """initialize the class with the parameters values
        """
        self.dmg_thr = dmg_thr
        self.dmg_shape = dmg_shape
        self.res_dmg = res_dmg
        self.mit_cst = mit_cst

    def estimate_damage(self, event):
        """Calculate the damage produced by a flood event of a
        specified mangitude.

        Returns a float representing the damage magnitude.
        """
        if event < self.dmg_thr:
            return 0.

        else:
            return 1. - np.exp(-((event - self.dmg_thr) / self.dmg_shape))

    def estimate_residual_damage(self, damage, preparedness):
        """Calculate the residual damage given a specific disaster damage
        magnitude and preparedness level.

        Returns a float representing the residual damage magnitude
        """
        dmg_fun = np.log(1 / self.res_dmg)

        return damage * np.exp(- dmg_fun * preparedness)

    def get_warning_damage(self, event, warning_outcome, preparedness):
        """Calculate the flood-related damages for the different warning
        outcomes after a flood event has taken place.

        Returns a float representing the damage magnitude.
        """
        if warning_outcome == 'true positive':
            damage = self.estimate_damage(event)
            return self.estimate_residual_damage(damage, preparedness)

        else:
            return self.estimate_damage(event)

    def get_warning_loss(self, event, warning_outcome, preparedness):
        """Calculate the flood-related losses (damage + costs) for the
        different warning outcomes after a flood event has taken place.

        Returns a float representing the loss magnitude.
        """
        damage = self.estimate_damage(event)

        if warning_outcome == 'true positive':
            residual_damage = self.estimate_residual_damage(damage, preparedness)
            return residual_damage + event * self.mit_cst

        elif warning_outcome == 'false positive':
            return damage + event * self.mit_cst

        else:
            return damage

###############################################################################
# DISASTER PREPAREDNESS
###############################################################################


class SocialPreparedness(object):
    """Assess the disaster preparedness level as a function of
    the recency of flood events.
    """
    def __init__(self, shock, half_life):

        # set parameters
        self.shock = shock
        # calculate the decay constants as a function of the half-life
        self.decay_ct = np.log(2) / half_life

    def update_preparedness(self, damage, preparedness):
        """Calculate the impact of the different warning outcomes on the
        disaster preparedness level.

        Returns a float (0,1] representing the degree of preparedness.
        """
        if damage == 0.:
            prep_tmp = preparedness - self.decay_ct * preparedness

        else:
            prep_tmp = preparedness + self.shock * damage

        # minimum preparedness value is 1%, set it as a variable?
        return np.clip(prep_tmp, 0.01, 1.)

###############################################################################
# MODEL EVALUATION
###############################################################################


class ModelEvaluation(object):
    """Assess the efficiency of the model according to different
    metrics.
    """
    def __init__(self):
        self.tn = None
        self.fn = None
        self.fp = None
        self.tp = None

    def count_warning_outcomes(self, outcome_ls):
        """Calculate the number of occurrences of each warning outcome in the
        time series.

        Returns a float for each warning outcome.
        """
        count = Counter(outcome_ls)

        self.tn = count['true negative']
        self.fn = count['false negative']
        self.fp = count['false positive']
        self.tp = count['true positive']

    def flood_frequency(self):
        """Calculate the frequency of occurrence of flood events.

        Returns a float representing the disaster frequency [0, 1]
        """
        return (self.tp + self.fn) / (self.tn + self.fn + self.fp + self.tp)

    def return_period(self):
        """Calculate the return period of flood events.

        Returns a float representing the return period in years.
        """
        if self.tp + self.fn == 0.:
            # raise ValueError('No flood events recorded in the time series.')
            return None

        else:
            return (self.tn + self.fn + self.fp + self.tp + 1) / (self.tp + self.fn)

    def hit_rate(self):
        """Calculate the hit rate (probability of detection)
        of the warning system.

        Returns a float representing the hit rate [0, 1]
        """
        if self.fn + self.tp == 0.:
            # raise ValueError('No flood events recorded in the time series.')
            return None

        else:
            return self.tp / (self.fn + self.tp)

    def false_alarm_rate(self):
        """Calculate the false alarm rate (probability of false detection)
        for the warning system.

        Returns a float representing the false alarm rate [0, 1]
        """
        if self.tn + self.fp == 0.:
            # raise ValueError('No normal events recorded in the time series.')
            return None

        else:
            return self.fp / (self.tn + self.fp)

    def false_alarm_ratio(self):
        """Calculate the false alarm ratio for the warning system.

        Returns a float representing the false alarm ratio [0, 1]
        """
        if self.fp + self.tp == 0.:
            # raise ValueError('No alarms were raised during these period.')
            return None

        else:
            return self.fp / (self.fp + self.tp)

    def relative_loss(self, climate_loss_ls, warning_loss_ls):
        """Calculate the relative loss produced by the mitigation measures.

        Returns a float representing the relative loss [0, 1]
        """
        # check if there is any non-null element in the list
        if any(climate_loss_ls) is False:
            # raise ValueError('No flood events produced losses to the system.')
            return None

        else:
            return sum(warning_loss_ls) / sum(climate_loss_ls)

    def calculate_statistics(self, outcome_ls, climate_loss_ls,
                             warning_loss_ls, preparedness_ls):
        """Calculate the statistics for the model output
        """
        self.count_warning_outcomes(outcome_ls)

        self.stats = {'Probability of occurrence': self.flood_frequency(),
                      'Return period (yrs)': self.return_period(),
                      'Hit rate': self.hit_rate(),
                      'False alarm rate': self.false_alarm_rate(),
                      'False alarm ratio': self.false_alarm_ratio(),
                      'Relative loss': self.relative_loss(climate_loss_ls,
                                                          warning_loss_ls),
                      'Average preparedness': np.average(preparedness_ls)
                      }

    def save_statistics(self, statfile, mode='single_run'):
        """write the statistics to the output file
        """
        path_out = os.getcwd() + '\\..\\output\\' + mode + '\\'
        if not os.path.exists(path_out):
            os.makedirs(path_out)

        if mode == 'single_run':
            df = pd.DataFrame.from_dict(self.stats, orient='index')
            df.to_csv(path_out + statfile, sep='\t', header=False)

        elif mode == 'monte_carlo':
            self.mc_stats.to_csv(path_out + statfile, index=False)