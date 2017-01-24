# -*- coding: utf-8 -*-
###############################################################################
# PREP-FEWS
# Social Preparedness in Early Warning Systems model
#
# Main model script
#
# Marc Girons Lopez, Giuliano di Baldassarre, Jan Seibert
###############################################################################

import numpy as np
import pandas as pd
from math import isclose

import modules as mod
import functions as fun


###############################################################################
# PREP-FEWS MODEL
###############################################################################


class PrepFews(mod.ForecastingSystem, mod.WarningAssessment,
               mod.SocialPreparedness, mod.LossEstimation,
               mod.ModelEvaluation):
    """prep-fews model to ...
    """
    def __init__(self, mean, sd, shape, scale, mag_thr, prob_thr, dmg_thr,
                 dmg_shape, res_dmg, mit_cst, init_prep, shock, half_life,
                 ct_prep=False):
        """
        """
        # set parameters and variables
        self.ct_prep = ct_prep
        self.preparedness = init_prep

        # initialize super-classes
        mod.ForecastingSystem.__init__(self, mean, sd, shape, scale)
        mod.WarningAssessment.__init__(self, mag_thr, prob_thr)
        mod.LossEstimation.__init__(self, dmg_thr, dmg_shape, res_dmg, mit_cst)
        mod.SocialPreparedness.__init__(self, shock, half_life)
        mod.ModelEvaluation.__init__(self)

        # Initialize lists to keep track of the states of the system
        self.forecast_ls = []
        self.outcome_ls = []
        self.climate_loss_ls = []
        self.warning_loss_ls = []
        self.warning_damage_ls = []
        self.preparedness_ls = []

        # keep track of the model results
        self.stats = None

    def run_model(self, datafile):
        """Run the prep-fews model

        Returns a dict witht the different performance measures
        """
        time_series = fun.load_data(datafile)

        for event in time_series['Runoff']:
            # calculate losses if no forecasting system is operational
            climate_loss = self.estimate_damage(event)

            # calculate losses if a forecasting system is operational
            forecast = self.issue_forecast(event)
            warning_outcome = self.assess_warning_outcome(event, forecast)
            warning_damage = self.get_warning_damage(event, warning_outcome,
                                                     self.preparedness)
            warning_loss = self.get_warning_loss(event, warning_outcome,
                                                 self.preparedness)

            # keep track of the states of the model
            self.forecast_ls.append(forecast)
            self.outcome_ls.append(warning_outcome)
            self.warning_loss_ls.append(warning_loss)
            self.climate_loss_ls.append(climate_loss)
            self.warning_damage_ls.append(warning_damage)
            self.preparedness_ls.append(self.preparedness)

            # update the variables
            if self.ct_prep is False:
                self.preparedness = self.update_preparedness(warning_damage,
                                                             self.preparedness)

###############################################################################
# DEFINE THE SINGLE-RUN MODE
###############################################################################


class SingleRun(PrepFews):

    def __init__(self, configfile, ct_prep=False):
        """
        """
        params = fun.load_parameters(configfile)

        # set parameters and variables
        mean = params['Forecast']['acc_mean']
        sd = params['Forecast']['acc_sd']
        shape = params['Forecast']['prc_shape']
        scale = params['Forecast']['prc_scale']

        mag_thr = params['Warning']['mag_thr']
        prob_thr = params['Warning']['prob_thr']

        dmg_thr = params['Loss']['dmg_thr']
        dmg_shape = params['Loss']['dmg_shape']
        res_dmg = params['Loss']['res_dmg']
        mit_cst = params['Loss']['mit_cst']

        ct_prep = ct_prep
        init_prep = params['Preparedness']['init_prep']
        shock = params['Preparedness']['shock']
        half_life = params['Preparedness']['half_life']

        # initialize the model
        PrepFews.__init__(self, mean, sd, shape, scale, mag_thr, prob_thr,
                          dmg_thr, dmg_shape, res_dmg, mit_cst, init_prep,
                          shock, half_life, ct_prep)

    def run(self, datafile, statfile):
        """Perform a single run of the MEWS model
        """
        # run the model
        self.run_model(datafile)

        # save the model results
        self.calculate_statistics(self.outcome_ls, self.climate_loss_ls,
                                  self.warning_loss_ls, self.preparedness_ls)
        self.save_statistics(statfile, mode='single_run')

###############################################################################
# DEFINE THE MONTE CARLO MODE
###############################################################################


class MonteCarloRun(PrepFews):

    def __init__(self, configfile, ct_prep=False):
        """
        """
        params = fun.load_parameters(configfile)

        # check if the config file is the correct one
        if len(params['Forecast']) != 8:
            raise ValueError('Are you sure you used a MonteCarlo config file?')

        # set parameters and variables
        self.iterations = params['MC']['iterations']

        self.mean_low = params['Forecast']['acc_mean_low']
        self.mean_high = params['Forecast']['acc_mean_high']
        self.sd_low = params['Forecast']['acc_sd_low']
        self.sd_high = params['Forecast']['acc_sd_high']
        self.shape_low = params['Forecast']['prc_shape_low']
        self.shape_high = params['Forecast']['prc_shape_high']
        self.scale_low = params['Forecast']['prc_scale_low']
        self.scale_high = params['Forecast']['prc_scale_high']

        self.mag_thr_low = params['Warning']['mag_thr_low']
        self.mag_thr_high = params['Warning']['mag_thr_high']
        self.prob_thr_low = params['Warning']['prob_thr_low']
        self.prob_thr_high = params['Warning']['prob_thr_high']

        self.dmg_thr_low = params['Loss']['dmg_thr_low']
        self.dmg_thr_high = params['Loss']['dmg_thr_high']
        self.dmg_shape_low = params['Loss']['dmg_shape_low']
        self.dmg_shape_high = params['Loss']['dmg_shape_high']
        self.res_dmg_low = params['Loss']['res_dmg_low']
        self.res_dmg_high = params['Loss']['res_dmg_high']
        self.mit_cst_low = params['Loss']['mit_cst_low']
        self.mit_cst_high = params['Loss']['mit_cst_high']

        self.ct_prep = ct_prep
        self.init_prep_low = params['Preparedness']['init_prep_low']
        self.init_prep_high = params['Preparedness']['init_prep_high']
        self.shock_low = params['Preparedness']['shock_low']
        self.shock_high = params['Preparedness']['shock_high']
        self.half_life_low = params['Preparedness']['half_life_low']
        self.half_life_high = params['Preparedness']['half_life_high']

        columns = ['index', 'mean', 'sd', 'shape', 'scale', 'mag_thr',
                   'prob_thr', 'dmg_thr', 'dmg_shape', 'res_dmg', 'mit_cst',
                   'init_prep', 'shock', 'half_life', 'flood_freq', 'yrp',
                   'hr', 'far', 'fao', 'rel_loss', 'av_prep']
        self.mc_stats = pd.DataFrame(index=range(self.iterations),
                                     columns=columns)

    def mc_value(self, val_low, val_high):
        """
        """
        # if no range is given (val_low = val_high), pick that value
        if isclose(val_low, val_high) is True:
            return val_low
        # return a random value between val_low and val_high
        else:
            return val_low + np.random.random() * (val_high - val_low)

    def run(self, datafile, statfile):
        """
        """
        fun.progress_bar(0, self.iterations)
        for i in range(self.iterations):

            # set the random values
            mean = self.mc_value(self.mean_low, self.mean_high)
            sd = self.mc_value(self.sd_low, self.sd_high)
            shape = self.mc_value(self.shape_low, self.shape_high)
            scale = self.mc_value(self.scale_low, self.scale_high)
            mag_thr = self.mc_value(self.mag_thr_low, self.mag_thr_high)
            prob_thr = self.mc_value(self.prob_thr_low, self.prob_thr_high)
            dmg_thr = self.mc_value(self.dmg_thr_low, self.dmg_thr_high)
            dmg_shape = self.mc_value(self.dmg_shape_low, self.dmg_shape_high)
            res_dmg = self.mc_value(self.res_dmg_low, self.res_dmg_high)
            mit_cst = self.mc_value(self.mit_cst_low, self.mit_cst_high)
            init_prep = self.mc_value(self.init_prep_low, self.init_prep_high)
            shock = self.mc_value(self.shock_low, self.shock_high)
            half_life = self.mc_value(self.half_life_low, self.half_life_high)

            # set the model
            model = PrepFews(mean, sd, shape, scale, mag_thr, prob_thr,
                             dmg_thr, dmg_shape, res_dmg, mit_cst, init_prep,
                             shock, half_life, self.ct_prep)

            # run the model
            model.run_model(datafile)

            # calculate the model statistics
            model.calculate_statistics(model.outcome_ls,
                                       model.climate_loss_ls,
                                       model.warning_loss_ls,
                                       model.preparedness_ls)

            # store the results
            self.mc_stats.loc[i] = [i, mean, sd, shape, scale, mag_thr,
                                    prob_thr, dmg_thr, dmg_shape, res_dmg,
                                    mit_cst, init_prep, shock, half_life,
                                    model.stats['Probability of occurrence'],
                                    model.stats['Return period (yrs)'],
                                    model.stats['Hit rate'],
                                    model.stats['False alarm rate'],
                                    model.stats['False alarm ratio'],
                                    model.stats['Relative loss'],
                                    model.stats['Average preparedness']]

            # update progress bar
            fun.progress_bar(i, self.iterations)

        # write the results to the output file
        self.save_statistics(statfile, mode='monte_carlo')

        # finalize progress bar
        fun.progress_bar(self.iterations, self.iterations)

###############################################################################
# RUN THE PREP-FEWS MODEL
###############################################################################

if __name__ == '__main__':

    model_type = 'monte_carlo'

    if model_type == 'single_run':
        single_run = SingleRun(configfile='config.yml')
        single_run.run(datafile='time_series.txt',
                       statfile='results_single.txt')

    elif model_type == 'monte_carlo':
        monte_carlo = MonteCarloRun(configfile='config_mc.yml')
        monte_carlo.run(datafile='time_series.txt',
                        statfile='results_mc.txt')
