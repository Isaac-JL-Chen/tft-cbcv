# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Default configs for TFT experiments.

Contains the default output paths for data, serialised models and predictions
for the main experiments used in the publication.
"""

import os

import data_formatters.electricity
import data_formatters.favorita
import data_formatters.traffic
import data_formatters.volatility

import data_formatters.acq
import data_formatters.init_aov
import data_formatters.init_spend
import data_formatters.censored_spend
import data_formatters.rpt_orders
import data_formatters.rpt_orders_search
import data_formatters.rpt_aov


class ExperimentConfig(object):
  """Defines experiment configs and paths to data/model/outputs.

  Attributes:
    root_folder: Root folder to contain all experimental outputs.
    experiment: Name of experiment to run.
    data_folder: Folder to store data for experiment.
    model_folder: Folder to store serialised models.
    results_folder: Folder to store results.
    data_csv_path: Path to primary data csv file used in experiment.
    hyperparam_iterations: Default number of random search iterations for
      experiment.
  """

  default_experiments = ['volatility', 'electricity', 'traffic', 'favorita', 
                         
                         'acq_10', 'acq_100', 'acq_1000', 
                         'init_aov_10', 'init_aov_100', 'init_aov_1000', 
                         'init_spend_10', 'init_spend_100', 'init_spend_1000', 
                         'censored_spend_10', 'censored_spend_100', 'censored_spend_1000', 
                         
                         'rpt_orders_10', 'rpt_orders_100', 'rpt_orders_1000',
                         'rpt_orders_s1', 'rpt_orders_s2', 'rpt_orders_s3', 'rpt_orders_s4',
                         'rpt_orders_s5', 'rpt_orders_s6', 'rpt_orders_s7', 'rpt_orders_s8',
                         'rpt_orders_s9', 
                         'rpt_aov_10', 'rpt_aov_100', 'rpt_aov_1000',                      
                         ]

  def __init__(self, experiment='acq_10', root_folder=None):
    """Creates configs based on default experiment chosen.

    Args:
      experiment: Name of experiment.
      root_folder: Root folder to save all outputs of training.
    """

    if experiment not in self.default_experiments:
      raise ValueError('Unrecognised experiment={}'.format(experiment))

    # Defines all relevant paths
    if root_folder is None:
      root_folder = os.path.join(
          os.path.dirname(os.path.realpath(__file__)), '..', 'outputs')
      print('Using root folder {}'.format(root_folder))

    self.root_folder = root_folder
    self.experiment = experiment
    # self.data_folder = os.path.join(root_folder, 'data', experiment)
    self.data_folder = os.path.join(root_folder, '../data/preprocessed_data/tft_google')
    self.model_folder = os.path.join(root_folder, 'saved_models', experiment)
    self.results_folder = os.path.join(root_folder, 'results', experiment)

    # Creates folders if they don't exist
    for relevant_directory in [
        self.root_folder, self.data_folder, self.model_folder,
        self.results_folder
    ]:
      if not os.path.exists(relevant_directory):
        os.makedirs(relevant_directory)

  @property
  def data_csv_path(self):
    csv_map = {
        'volatility': 'formatted_omi_vol.csv',
        'electricity': 'hourly_electricity.csv',
        'traffic': 'hourly_data.csv',
        'favorita': 'favorita_consolidated.csv',
        
        'acq_10': 'company_0_10_acq_initaov.csv',
        'acq_100': 'company_0_100_acq_initaov.csv',
        'acq_1000': 'company_0_1000_acq_initaov.csv',

        'init_aov_10': 'company_0_10_acq_initaov.csv',
        'init_aov_100': 'company_0_100_acq_initaov.csv',
        'init_aov_1000': 'company_0_1000_acq_initaov.csv',
                
        'init_spend_10': 'company_0_10_acq_initaov.csv',
        'init_spend_100': 'company_0_100_acq_initaov.csv',
        'init_spend_1000': 'company_0_1000_acq_initaov.csv',
        
        'censored_spend_10': 'company_0_10_censoredspend.csv', 
        'censored_spend_100': 'company_0_100_censoredspend.csv',
        'censored_spend_1000': 'company_0_1000_censoredspend.csv',
        
        'rpt_orders_10': 'company_0_10.csv',
        'rpt_orders_100': 'company_0_100.csv',
        'rpt_orders_1000': 'company_0_1000.csv',
        
        'rpt_orders_s1': 'company_0_1000.csv',
        'rpt_orders_s2': 'company_0_1000.csv',
        'rpt_orders_s3': 'company_0_1000.csv',
        'rpt_orders_s4': 'company_0_1000.csv',
        'rpt_orders_s5': 'company_0_1000.csv',
        'rpt_orders_s6': 'company_0_1000.csv',
        'rpt_orders_s7': 'company_0_1000.csv',
        'rpt_orders_s8': 'company_0_1000.csv',
        'rpt_orders_s9': 'company_0_1000.csv',
        
        'rpt_aov_10': 'company_0_10.csv',
        'rpt_aov_100': 'company_0_100.csv', 
        'rpt_aov_1000': 'company_0_1000.csv',
    }

    return os.path.join(self.data_folder, csv_map[self.experiment])

  @property
  def hyperparam_iterations(self):

    return 240 if self.experiment == 'volatility' else 60

  def make_data_formatter(self):
    """Gets a data formatter object for experiment.

    Returns:
      Default DataFormatter per experiment.
    """

    data_formatter_class = {
        'volatility': data_formatters.volatility.VolatilityFormatter,
        'electricity': data_formatters.electricity.ElectricityFormatter,
        'traffic': data_formatters.traffic.TrafficFormatter,
        'favorita': data_formatters.favorita.FavoritaFormatter,
        
        'acq_10': data_formatters.acq.AcqTenFormatter,
        'acq_100': data_formatters.acq.AcqHundredFormatter,
        'acq_1000': data_formatters.acq.AcqThousandFormatter,
        
        'init_aov_10': data_formatters.init_aov.InitAOVTenFormatter, 
        'init_aov_100': data_formatters.init_aov.InitAOVHundredFormatter, 
        'init_aov_1000': data_formatters.init_aov.InitAOVThousandFormatter, 
        
        'init_spend_10': data_formatters.init_spend.InitSpendTenFormatter, 
        'init_spend_100': data_formatters.init_spend.InitSpendHundredFormatter, 
        'init_spend_1000': data_formatters.init_spend.InitSpendThousandFormatter, 
        
        'censored_spend_10': data_formatters.censored_spend.CensoredSpendTenFormatter, 
        'censored_spend_100': data_formatters.censored_spend.CensoredSpendHundredFormatter, 
        'censored_spend_1000': data_formatters.censored_spend.CensoredSpendThousandFormatter, 
        
        'rpt_orders_10': data_formatters.rpt_orders.RptOrdersTenFormatter,
        'rpt_orders_100': data_formatters.rpt_orders.RptOrdersHundredFormatter,
        'rpt_orders_1000': data_formatters.rpt_orders.RptOrdersThousandFormatter,
  
        'rpt_orders_s1': data_formatters.rpt_orders_search.RptOrdersDR0_2HL160LR1e3GR1Formatter,
        'rpt_orders_s2': data_formatters.rpt_orders_search.RptOrdersDR0_5HL160LR1e3GR1Formatter,
        'rpt_orders_s3': data_formatters.rpt_orders_search.RptOrdersDR0_2HL160LR1e4GR1Formatter,
        'rpt_orders_s4': data_formatters.rpt_orders_search.RptOrdersDR0_2HL240LR1e3GR1Formatter,
        
        'rpt_orders_s5': data_formatters.rpt_orders_search.RptOrdersDR0_5HL160LR1e43GR1Formatter,
        'rpt_orders_s6': data_formatters.rpt_orders_search.RptOrdersDR0_5HL240LR1e43GR1Formatter,
        'rpt_orders_s7': data_formatters.rpt_orders_search.RptOrdersDR0_5HL160LR1e43GR100Formatter,
        'rpt_orders_s8': data_formatters.rpt_orders_search.RptOrdersDR0_2HL240LR1e43GR1Formatter,
        'rpt_orders_s9': data_formatters.rpt_orders_search.RptOrdersDR0_2HL160LR5e43GR1Formatter,                
        
        'rpt_aov_10': data_formatters.rpt_aov.RptAOVTenFormatter, 
        'rpt_aov_100': data_formatters.rpt_aov.RptAOVHundredFormatter, 
        'rpt_aov_1000': data_formatters.rpt_aov.RptAOVThousandFormatter,
    }

    return data_formatter_class[self.experiment]()
