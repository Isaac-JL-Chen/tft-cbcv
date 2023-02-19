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

"""Trains TFT based on a defined set of parameters.

Uses default parameters supplied from the configs file to train a TFT model from scratch.

Usage:
  default)
  python3 script_train_fixed_params {expt_name} {output_folder}
  
  specifying experiment and gpu, and saving printed output on bash into txt file)
  e.g. python3 -m script_train_fixed_params --expt_name rpt_aov_1000 --gpu_num 6 | tee txt/rpt_aov_1000_s3_best.txt

Command line args:
  expt_name: Name of dataset/experiment to train.
  output_folder: Root folder in which experiment is saved
  gpu_num: gpu number (default is 0 for single GPU machine)

"""

import argparse
import datetime
from datetime import date
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

## disable all warning messages
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # INFO and WARNING messages are not printed

## load local files
import data_formatters.base
import expt_settings.configs
import libs.hyperparam_opt
import libs.tft_model
import libs.utils as utils

## version compatibility
import tensorflow.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR) # deprecation warning is not printed

ExperimentConfig = expt_settings.configs.ExperimentConfig # defines experiment configs and paths to outputs. experiment config detail is in data_formatter.
ModelClass = libs.tft_model.TemporalFusionTransformer # full TFT architecture with training, evaluation and prediction using Pandas Dataframe inputs
HyperparamOptManager = libs.hyperparam_opt.HyperparamOptManager # classes used for hyperparameter optimisation on a single machine/GPU
# DistributedHyperparamOptManager = libs.hyperparam_opt.DistributedHyperparamOptManager # for multi GPU --- many errors here, not using at this moment



def main(expt_name,
         use_gpu,
         gpu_number,
         data_csv_path,
         data_formatter,
         num_repeats=1,
         use_testing_mode=False):
  
  """Trains tft based on defined model params.

  Args:
    expt_name: Name of experiment e.g. 'acq_1000' specified in expt_settings.configs.ExperimentConfig
    use_gpu: [yes/no] Whether to run tensorflow with GPU operations
    gpu_number: GPU device number if use_gpu = TRUE and multiple GPUs available
    data_csv_path: Path to csv file containing data
    data_formatter: Dataset-specific data formatter instance. inherit from data_formatters.base.GenericDataFormatter
    use_testing_mode: Uses a smaller models and data sizes for testing purposes only -- switch to False to use original default settings      
  """

  ## check whether imported data formatter is a correct class instance.
  if not isinstance(data_formatter, data_formatters.base.GenericDataFormatter):
    raise ValueError(
        "Data formatters should inherit from" +
        "AbstractDataFormatter! Type={}".format(type(data_formatter)))


  ## Tensorflow setup and
  ## specifies whether to run graph on gpu or cpu and which GPU ID to use for multi GPU machines.
  print("\n\n*** Tensorflow setup ***")
  if use_gpu:
        if gpu_number == 'all':
              default_keras_session = tf.keras.backend.get_session() # use all available GPUs
              tf_config = utils.get_default_tensorflow_config(tf_device="gpu")
        else:
              tf_config = utils.get_default_tensorflow_config(tf_device="gpu", gpu_id=int(gpu_number))
              tf_config.gpu_options.allow_growth = True # single GPU
              default_keras_session = tf.keras.backend.get_session() # use selected GPU
  else:
    tf_config = utils.get_default_tensorflow_config(tf_device="cpu")


  print("\n\n*** Training from defined parameters for experiment: {} ***".format(expt_name))
  print("Loading & splitting data...") ## data_csv_path is in configs file
  raw_data = pd.read_csv(data_csv_path, index_col=0) # first column of raw data is index column Unnamed:0
  print(" - Data is located in: ", data_csv_path)
  train, valid, test = data_formatter.split_data(raw_data) # set validation start, test start, test end in data_formatter file
  train_samples, valid_samples = data_formatter.get_num_samples_for_calibration() # if subsampling data


  ## Sets up default params
  # data specific data formatter should define these 5 fixed_params: total time steps of TFT, num LSTM encoder steps, max num epochs, early stopping patience, CPU multiprocessing workers
  fixed_params = data_formatter.get_experiment_params() # Returns fixed model parameters for experiments.
  # data specific data formatter can flexibly have model_params:
  params = data_formatter.get_default_model_params() # Returns default optimised model parameters.
  
  
  ## Folder path where models are serialized
  model_folder = os.path.join(config.model_folder, datetime.datetime.now().strftime("%Y-%m-%d_%H_%M"))
  params["model_folder"] = model_folder
  print("Model will be saved in: ", model_folder)


  ## Parameter overrides for testing only! Small sizes used to speed up script.
  if use_testing_mode:
    fixed_params["num_epochs"] = 1
    params["hidden_layer_size"] = 5
    train_samples, valid_samples = 100, 10

  ## Sets up hyperparam manager
  print("\n*** Loading hyperparm manager ***")
  
  # opt_manager = DistributedHyperparamOptManager({k: [params[k]] for k in params}, fixed_params, model_folder) ## Error- nont using distributed optimization
  opt_manager = HyperparamOptManager({k: [params[k]] for k in params}, fixed_params, model_folder)



  #====================================================
  print("\n\n*** Running calibration ***")
  ## For each iteration, we start with different initialization to find best parameters set
  ## that provides smallest local minimum of validation loss.
  num_repeats = num_repeats # Training -- one iteration only
  best_loss = np.Inf
  
  for _ in range(num_repeats):

    tf.reset_default_graph()
    with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
      # set session on keras
      tf.keras.backend.set_session(sess)

      # set initial parameter, model, training data
      params = opt_manager.get_next_parameters() # get initialized parameters from random search in new iteration
      model = ModelClass(params, use_cudnn=use_gpu)
      if not model.training_data_cached():
        # Data to batch and cache & Maximum number of samples to extract (-1 to use all data)
        # model.cache_batched_data(train, "train", num_samples=train_samples)
        # model.cache_batched_data(valid, "valid", num_samples=valid_samples)
        model.cache_batched_data(train, "train", num_samples=-1)
        model.cache_batched_data(valid, "valid", num_samples=-1)

      # run session with initialization
      sess.run(tf.global_variables_initializer())

      model.fit()
    
      val_loss = model.evaluate()

      if val_loss < best_loss:
        opt_manager.update_score(params, val_loss, model)
        best_loss = val_loss

      tf.keras.backend.set_session(default_keras_session)
      
      print('\n* Iteration ' + str(_) + ' is done.')



  #====================================================
  print("\n\n*** Running tests ***")
  
  tf.reset_default_graph()
  with tf.Graph().as_default(), tf.Session(config=tf_config) as sess:
    # set session on keras
    tf.keras.backend.set_session(sess)
    
    best_params = opt_manager.get_best_params()
    model = ModelClass(best_params, use_cudnn=use_gpu)

    model.load(opt_manager.hyperparam_folder)

    print("\nComputing best validation loss")
    val_loss = model.evaluate(valid)


    print("\nSaving results")
    ## prepare valid_fortest data
    test_start = model.predict(test, return_targets=True)["p50"]['forecast_time'].iloc[0]
    cal_end = model.predict(pd.concat([train,valid]), return_targets=True)["p50"]['forecast_time'].iloc[-1]
    gap = int(str((pd.to_datetime(test_start) - pd.to_datetime(cal_end))/7).split()[0])-1
    
    if expt_name in ['censored_spend_10', 'censored_spend_100', 'censored_spend_1000']:
      valid_fortest = valid[valid['week'] > str(pd.to_datetime(valid['week'].iloc[-1]) - datetime.timedelta(weeks=gap)).split()[0]]
    else: 
      valid_fortest = valid[valid['acq_week'] > str(pd.to_datetime(valid['acq_week'].iloc[-1]) - datetime.timedelta(weeks=gap)).split()[0]]


    print("\nComputing test loss")
    output_map = model.predict(pd.concat([valid_fortest,test]), return_targets=True)
    targets = data_formatter.format_predictions(output_map["targets"])
    p10_forecast = data_formatter.format_predictions(output_map["p10"])
    p50_forecast = data_formatter.format_predictions(output_map["p50"])
    p90_forecast = data_formatter.format_predictions(output_map["p90"])

    output_withtrain_map = model.predict(pd.concat([train,valid]), return_targets=True)
    targets_cal = data_formatter.format_predictions(output_withtrain_map["targets"])
    p10_forecast_cal = data_formatter.format_predictions(output_withtrain_map["p10"])
    p50_forecast_cal = data_formatter.format_predictions(output_withtrain_map["p50"])
    p90_forecast_cal = data_formatter.format_predictions(output_withtrain_map["p90"])

    ### save files
    if not os.path.exists(f'{config.results_folder}/{version_name}'): os.makedirs(f'{config.results_folder}/{version_name}')
    targets.to_csv(f'{config.results_folder}/{version_name}/target.csv')
    targets_cal.to_csv(f'{config.results_folder}/{version_name}/target_cal.csv')
    p90_forecast.to_csv(f'{config.results_folder}/{version_name}/pred_q90.csv')
    p90_forecast_cal.to_csv(f'{config.results_folder}/{version_name}/pred_q90_cal.csv')
    p50_forecast.to_csv(f'{config.results_folder}/{version_name}/pred_q50.csv')
    p50_forecast_cal.to_csv(f'{config.results_folder}/{version_name}/pred_q50_cal.csv') 
    p10_forecast.to_csv(f'{config.results_folder}/{version_name}/pred_q10.csv')
    p10_forecast_cal.to_csv(f'{config.results_folder}/{version_name}/pred_q10_cal.csv')


   

    def extract_numerical_data(data):
      """Strips out forecast time and identifier columns."""
      return data[[
          col for col in data.columns
          if col not in {"forecast_time", "identifier"}
      ]]

    p10_loss = utils.numpy_normalised_quantile_loss(
        extract_numerical_data(targets), extract_numerical_data(p10_forecast),
        0.1)
    p50_loss = utils.numpy_normalised_quantile_loss(
        extract_numerical_data(targets), extract_numerical_data(p50_forecast),
        0.5)
    p90_loss = utils.numpy_normalised_quantile_loss(
        extract_numerical_data(targets), extract_numerical_data(p90_forecast),
        0.9)

    tf.keras.backend.set_session(default_keras_session)



  print("Training completed @ {}".format(datetime.datetime.now()))
  print("Best validation loss = {x:.5f}".format(x=val_loss))
  print("Params:")

  for k in best_params:
    print(k, " = ", best_params[k])
  print()
  print("Normalised Quantile Loss for Test Data: P50={x1:.5f}, P90={x2:.5f}, P10={x3:.5f}".format(
      x1=p50_loss.mean(), x2=p90_loss.mean(), x3=p10_loss.mean()))







if __name__ == "__main__":
  # file_writer = tf2.summary.create_file_writer("logs")
  
  def get_args():
    """Gets settings from command line."""

    experiment_names = ExperimentConfig.default_experiments

    parser = argparse.ArgumentParser(description="Data download configs")
    parser.add_argument(
        "--expt_name",
        metavar="e",
        type=str,
        nargs="?",
        default="rpt_order",
        choices=experiment_names,
        help="Experiment Name. Default={}".format(",".join(experiment_names)))
    parser.add_argument(
        "--output_folder",
        metavar="f",
        type=str,
        nargs="?",
        default=str(date.today()), 
        help="Path to folder for output data. Default={}".format(str(date.today())))
    parser.add_argument(
        "--use_gpu",
        metavar="g",
        type=str,
        nargs="?",
        choices=["yes", "no"],
        default="yes",
        help="Whether to use gpu for training.")
    parser.add_argument(
        "--gpu_num",
        type=str,
        default='all',
        help="GPU device number. Default={}. Or set GPU id as an integer.".format("all"))    
    parser.add_argument(
        "--num_iter",
        type=int,
        nargs="?",
        default=1,
        help="number of iterations. Default={}".format(1))

    args = parser.parse_known_args()[0]

    root_folder = None if args.output_folder == "." else args.output_folder

    return args.expt_name, args.gpu_num, root_folder, args.use_gpu == "yes", args.num_iter

  name, gpu_num, output_folder, use_tensorflow_with_gpu, num_repeats = get_args()

  print("Using output folder {}".format(output_folder))

  config = ExperimentConfig(experiment=name, root_folder=output_folder)
  formatter = config.make_data_formatter()



  # Customise inputs to main() for new datasets.
  main(
      expt_name=name,
      use_gpu=use_tensorflow_with_gpu,
      gpu_number=gpu_num,
      data_csv_path=config.data_csv_path,
      data_formatter=formatter,
      num_repeats=num_repeats,
      use_testing_mode=False)  # Change to false to use original default params



### check tensorboard: tensorboard --logdir {output_folder}/saved_models/{experiment}/{%Y-%m-%d_%H_%M}/logs
### TensorBoard 2.9.1 at http://localhost:6006/
