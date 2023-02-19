Add new experiment
1. generate data_formatters/{rpt_orders_search}
2. modify expt_settings/configs
- import data_formatters.rpt_orders_search
- add {experiment names} in default_experiments 
- update data_csv_path(self): csv_map= {experiment names} : datafile
- update data_formatter_class= {experiment names} : {formatter class}
3. put csv file on outputs/data/{experiment names}
4. launch on tmux
- tmux new -s myname
- tmux a -t myname
- [ctrl + b] > d # detach
- tmux kill-session -t myname

==================================================
## 12/6/2022

conda activate google-tft-tf2
cd jianlin-tft

python3 -m script_train_fixed_params --expt_name acq_1000 --ver_name ver1 --output_folder output_2022_12 --gpu_num 1 | tee txt/output_2022_12_acq_1000_ver1.txt

python3 -m script_train_fixed_params --expt_name init_aov_1000 --ver_name ver1 --output_folder output_2022_12 --gpu_num 1 | tee txt/output_2022_12_init_aov_1000_ver1.txt

python3 -m script_train_fixed_params --expt_name censored_spend_1000 --ver_name ver1 --output_folder output_2022_12 --gpu_num 2 | tee txt/output_2022_12_censored_spend_1000_ver1.txt









# KB's command

conda activate google-tft-tf2
cd jianlin-tft

python3 -m script_hyperparam_opt --expt_name acq_10 --gpu_num 0 | tee txt/acq_10_hyp_optim.txt

python3 -m script_hyperparam_opt --expt_name acq_100 --gpu_num 1 | tee txt/acq_100_hyp_optim.txt

python3 -m script_hyperparam_opt --expt_name acq_1000 --gpu_num 7 | tee txt/acq_1000_hyp_optim.txt

==========================================

python3 -m script_train_fixed_params --expt_name acq_10 --gpu_num 0 | tee txt/acq_10_fixed.txt

python3 -m script_train_fixed_params --expt_name acq_100 --gpu_num 1 | tee txt/acq_100_fixed.txt

python3 -m script_train_fixed_params --expt_name acq_1000 --gpu_num 7 | tee txt/acq_1000_fixed.txt

==========================================
## 11/1/2022 1:30PM

python3 -m script_hyperparam_opt --expt_name rpt_orders_10 --gpu_num 0 | tee txt/rpt_orders_10_hyp_optim.txt
>> tmux attach -t 3

python3 -m script_hyperparam_opt --expt_name rpt_orders_100 --gpu_num 1 | tee txt/rpt_orders_100_hyp_optim.txt
>> tmux attach -t 1

python3 -m script_hyperparam_opt --expt_name rpt_orders_1000 --gpu_num 4 | tee txt/rpt_orders_1000_hyp_optim.txt



==========================================

  @classmethod
  def get_hyperparm_choices(cls):
    """Returns hyperparameter ranges for random search."""
    return {
        'dropout_rate': [0.2, 0.5],
        'hidden_layer_size': [160, 240],
        'learning_rate': [1e-4, 5e-4, 1e-3],
        'minibatch_size': [256],
        'max_gradient_norm': [1.0, 100.0],
        'num_heads': [4],
        'stack_size': [1],
    }



=======================================
## 11/1/2022 11:30PM

python3 -m script_train_fixed_params --expt_name rpt_orders_s1 --gpu_num 7 | tee txt/rpt_orders_s1.txt

python3 -m script_train_fixed_params --expt_name rpt_orders_s2 --gpu_num 4 | tee txt/rpt_orders_s2.txt

python3 -m script_train_fixed_params --expt_name rpt_orders_s3 --gpu_num 0 | tee txt/rpt_orders_s3.txt

python3 -m script_train_fixed_params --expt_name rpt_orders_s4 --gpu_num 7 | tee txt/rpt_orders_s4.txt

python3 -m script_hyperparam_opt --expt_name rpt_orders_1000 --gpu_num 4 | tee txt/rpt_orders_1000_hyperparam_shorter.txt

=======================================
## 11/2/2022 8:30AM

python3 -m script_train_fixed_params --expt_name rpt_orders_s5 --gpu_num 1 | tee txt/rpt_orders_s5.txt

python3 -m script_train_fixed_params --expt_name rpt_orders_s6 --gpu_num 5 | tee txt/rpt_orders_s6.txt

python3 -m script_train_fixed_params --expt_name rpt_orders_s7 --gpu_num 7 | tee txt/rpt_orders_s7.txt

python3 -m script_train_fixed_params --expt_name rpt_orders_s8 --gpu_num 1 | tee txt/rpt_orders_s8.txt

python3 -m script_train_fixed_params --expt_name rpt_orders_s9 --gpu_num 5 | tee txt/rpt_orders_s9.txt

=======================================
## 11/2/2022 12:45PM

python3 -m script_hyperparam_opt --expt_name rpt_aov_1000 --gpu_num 5 | tee txt/rpt_aov_1000_hyperparam_shorter.txt

python3 -m script_hyperparam_opt --expt_name init_spend_1000 --gpu_num 7 | tee txt/init_spend_1000_hyperparam_shorter.txt

python3 -m script_hyperparam_opt --expt_name acq_1000 --gpu_num 5 | tee txt/acq_1000_hyperparam_shorter.txt

=======================================
## 11/2/2022 5:00PM

python3 -m script_hyperparam_opt --expt_name censored_spend_1000 --gpu_num 0 | tee txt/censored_spend_1000_hyperparam_shorter.txt



========================================

## 11/6/2022 6:00PM

tmux new -s XXX
tmux a -t XXX


python3 -m script_train_fixed_params --expt_name rpt_orders_100 --gpu_num 2 | tee txt/rpt_orders_100_fixed_new.txt
tmux a -t rpt_order100_fix

python3 -m script_train_fixed_params --expt_name censored_spend_100 --gpu_num 2 | tee txt/censored_spend_100_fixed.txt
tmux a -t censored100_fix

python3 -m script_train_fixed_params --expt_name init_spend_100 --gpu_num 2 | tee txt/init_spend_100_fixed.txt
tmux a -t initspend100_fix

python3 -m script_train_fixed_params --expt_name acq_100 --gpu_num 2 | tee txt/acq_100_fixed_new.txt

python3 -m script_train_fixed_params --expt_name rpt_aov_100 --gpu_num 7 | tee txt/rpt_aov_100_fixed.txt
tmux a -t rpt_aov100_fix





=================================
11/1/2022

tmux a -t rptaov1000_best
python3 -m script_train_fixed_params --expt_name rpt_aov_1000 --gpu_num 6 | tee txt/rpt_aov_1000_s3_best.txt

tmux a -t censored1000_best
python3 -m script_train_fixed_params --expt_name censored_spend_1000 --gpu_num 6 | tee txt/censored_spend_1000_best_time.txt

tmux a -t censored1000_best
python3 -m script_train_fixed_params --expt_name censored_spend_1000 --gpu_num 6 | tee txt/censored_spend_1000_best_week.txt

tmux a -t initspend1000_best
python3 -m script_train_fixed_params --expt_name init_spend_1000 --gpu_num 3 | tee txt/init_spend_1000_best.txt

tmux a -t initspend1000_best
python3 -m script_train_fixed_params --expt_name acq_1000 --gpu_num 3 | tee txt/acq_1000_best.txt















# Jianlin Workspace Introduction

## Key file description

There are some trash in this workspace, so user only need to know the following important tips.

The google tft training code is saved in `/autodl-tmp/google-research/tft`. The data formatters are contained in `/autodl-tmp/google-research/tft/data_formatters`. You can ignore `rpt_order_onecohort` since it is just for debug. (ignore it in all of the following part)  

The codes of training data generation for google tft are saved in `/autodl-tmp/Boston/acq_gendata.ipynb` and `/autodl-tmp/Boston/rpt_order_gendata.ipynb`. In the notebook, I have also marked the cohort-wise, company-wise data generation part.

The code of Darts model training and the combination of the result of tft_one and tft_allco and the generation of comparison figures can be found in `/root/autodl-tmp/Boston/acquisition prediction TFT ver2.ipynb` and `/root/autodl-tmp/Boston/cohort_rpt_order_tft_dart.ipynb`. 

The raw data saved at `/autodl-tmp/Boston/data/weekly_cohort_sample100_zeromasked`

The generated training data (`company.csv`) and prediction result are saved in `/autodl-tmp/data/tft/data`.

The google tft models are saved in `/autodl-tmp/data/tft/saved_models`.

The result of acq prediction for tft_one, tft_allco and Darts are saved in `/autodl-tmp/Boston/output/tft_20220912/acq`.

The figures of comparison of those models on acq are saved in `/autodl-tmp/Boston/output/final_result/acq`.

The model of Darts is saved in `/root/autodl-tmp/Boston/model/tft_20220912`.

## Quick Start

1. Run `/autodl-tmp/Boston/acq_gendata.ipynb` and `/autodl-tmp/Boston/rpt_order_gendata.ipynb` first to generate the training data for google tft.

2. Train the google tft model with fixed params or optimizing the hyper-params together. Run:
    ```bash
    python3 -m script_train_fixed_params $EXPT $OUTPUT_FOLDER $USE_GPU 
    ```
    or
    ```bash
    python3 -m script_hyperparam_opt $EXPT $OUTPUT_FOLDER $USE_GPU yes
    ```
    In my machine, I only need to run:
    ```bash
    python3 -m script_train_fixed_params [rpt_order, rpt_order_allco, acq, acq_allco]
    ```
    or
    ```bash
    python3 -m script_hyperparam_opt [rpt_order, rpt_order_allco, acq, acq_allco]
    ```
    Therefore, you can change the default arguments in `/autodl-tmp/google-research/tft/script_train_fixed_params.py` or `/autodl-tmp/google-research/tft/script_hyperparam_opt.py` to achieve this goal.

3. Train the Darts model and generate the comparison figures of all three models. You can run `/root/autodl-tmp/Boston/acquisition prediction TFT ver2.ipynb` and `/root/autodl-tmp/Boston/cohort_rpt_order_tft_dart.ipynb`. 















# Original README:

## Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting

Authors: Bryan Lim, Sercan Arik, Nicolas Loeff and Tomas Pfister

Paper link: https://arxiv.org/pdf/1912.09363.pdf

### Abstract
> Multi-horizon forecasting problems often contain a complex mix of inputs -- including static (i.e. time-invariant) covariates, known future inputs, and other exogenous time series that are only observed historically -- without any prior information on how they interact with the target. While several deep learning models have been proposed for multi-step prediction, they typically comprise black-box models which do not account for the full range of inputs present in common scenarios. In this paper, we introduce the Temporal Fusion Transformer (TFT) -- a novel attention-based architecture which combines high-performance multi-horizon forecasting with interpretable insights into temporal dynamics. To learn temporal relationships at different scales, the TFT utilizes recurrent layers for local processing and interpretable self-attention layers for learning long-term dependencies. The TFT also uses specialized components for the judicious selection of relevant features and a series of gating layers to suppress unnecessary components, enabling high performance in a wide range of regimes. On a variety of real-world datasets, we demonstrate significant performance improvements over existing benchmarks, and showcase three practical interpretability use-cases of TFT.


## Code Organisation
This repository contains the source code for the Temporal Fusion Transformer, along with the training and evaluation routines for the experiments described in the paper.

The key modules for experiments are organised as:

* **data\_formatters**: Stores the main dataset-specific column definitions, along with functions for data transformation and normalization. For compatibility with the TFT, new experiments should implement a unique ``GenericDataFormatter`` (see **base.py**), with examples for the default experiments shown in the other python files.
* **expt\_settings**: Holds the folder paths and configurations for the default experiments,
* **libs**: Contains the main libraries, including classes to manage hyperparameter optimisation (**hyperparam\_opt.py**), the main TFT network class (**tft\_model.py**), and general helper functions (**utils.py**)

Scripts are all saved in the main folder, with descriptions below:

* **run.sh**: Simple shell script to ensure correct environmental setup.
* **script\_download\_data.py**: Downloads data for the main experiment and processes them into csv files ready for training/evaluation.
* **script\_train\_fixed\_params.py**: Calibrates the TFT using a predefined set of hyperparameters, and evaluates for a given experiment.
* **script\_hyperparameter\_optimisation.py**: Runs full hyperparameter optimization using the default random search ranges defined for the TFT.

## Running Default Experiements
Our four default experiments are divided into ``volatility``, ``electricity``, ``traffic``, and``favorita``. To run these experiments, first download the data, and then run the relevant training routine.

### Step 1: Download data for default experiments
To download the experiment data, run the following script:
```bash
python3 -m script_download_data $EXPT $OUTPUT_FOLDER
```
where ``$EXPT`` can be any of {``volatility``, ``electricity``, ``traffic``, ``favorita``}, and ``$OUTPUT_FOLDER`` denotes the root folder in which experiment outputs are saved.

### Step 2: Train and evaluate network
To train the network with the optimal default parameters, run:
```bash
python3 -m script_train_fixed_params $EXPT $OUTPUT_FOLDER $USE_GPU 
```
where ``$EXPT`` and ``$OUTPUT_FOLDER`` are as above, ``$GPU`` denotes whether to run with GPU support (options are {``'yes'`` or``'no'``}).

For full hyperparameter optimization, run:
```bash
python3 -m script_hyperparam_opt $EXPT $OUTPUT_FOLDER $USE_GPU yes
```
where options are as above.

## Customising Scripts for New Datasets
To re-use the hyperparameter optimization scripts for new datasets, we need to add a new experiment -- which involves the creation of a new data formatter and config updates.

### Step 1: Implement custom data formatter
First, create a new python file in ``data_formatters`` (e.g. example.py) which contains a data formatter class (e.g. ``ExampleFormatter``). This should inherit ``base.GenericDataFormatter`` and provide implementations of all abstract functions. An implementation example can be found in volatility.py.

### Step 2: Update configs.py
Add a name for your new experiement to the ``default_experiments`` attribute in ``expt_settings.configs.ExperimentConfig`` (e.g. ``example``).
```python
default_experiments = ['volatility', 'electricity', 'traffic', 'favorita', 'example']
```


Next, add an entry in ``data_csv_path`` mapping the experiment name to name of the csv file containing the data:

```python
@property
  def data_csv_path(self):
    csv_map = {
        'volatility': 'formatted_omi_vol.csv',
        'electricity': 'hourly_electricity.csv',
        'traffic': 'hourly_data.csv',
        'favorita': 'favorita_consolidated.csv',
        'example': 'mydata.csv'  # new entry here!
    }

    return os.path.join(self.data_folder, csv_map[self.experiment])
```

Lastly, add your custom data formatter to the factory function:

```python
def make_data_formatter(self):
    """Gets a data formatter object for experiment.

    Returns:
      Default DataFormatter per experiment.
    """

    data_formatter_class = {
        'volatility': data_formatters.volatility.VolatilityFormatter,
        'electricity': data_formatters.electricity.ElectricityFormatter,
        'traffic': data_formatters.traffic.TrafficFormatter,
        'example': data_formatters.example.ExampleFormatter, # new entry here!
    }
```

As an optional step, change the number of random search iterations if required:
```python
@property
  def hyperparam_iterations(self):
    
    my_search_iterations=1000
    
    if self.experiment == 'example':
      return my_search_iterations
    else:
      return 240 if self.experiment == 'volatility' else 60
```


### Step 3: Run training script
Full hyperparameter optimization can then be run as per the previous section, e.g.:
```bash
python3 -m script_hyperparam_opt example . yes yes

```













tmux new -s z_initspend_lstm_01
tmux new -s z_initspend_lstm_02

tmux new -s z_initspend_transformer_01
tmux new -s z_initspend_transformer_02
tmux new -s z_initspend_transformer_03

conda activate darts
cd jianlin-tft

python3 initspend_lstm_copy.py --gpu 0 --mainfolder MDCconference --start_K 0 --end_K 500
python3 initspend_lstm_copy.py --gpu 4 --mainfolder MDCconference --start_K 500 --end_K 1076

python3 initspend_transformer_copy.py --gpu 3 --mainfolder MDCconference --start_K 0 --end_K 350
python3 initspend_transformer_copy.py --gpu 3 --mainfolder MDCconference --start_K 350 --end_K 700
python3 initspend_transformer_copy.py --gpu 3 --mainfolder MDCconference --start_K 700 --end_K 1076


========================================
### 11/27/2022

tmux new -s z_rptorder_lstm_01
tmux new -s z_rptorder_lstm_01_1
tmux new -s z_rptorder_lstm_02
tmux new -s z_rptorder_lstm_03
tmux new -s z_rptorder_lstm_04
tmux new -s z_rptorder_lstm_04_1

tmux a -t z_rptorder_lstm_01
tmux a -t z_rptorder_lstm_02
tmux a -t z_rptorder_lstm_03
tmux a -t z_rptorder_lstm_04

conda activate darts
cd jianlin-tft

python3 rptorder_lstm.py --gpu 3 --mainfolder MDCconference --start_K 0 --end_K 150
149 DONE
python3 rptorder_lstm.py --gpu 3 --mainfolder MDCconference --start_K 150 --end_K 250
249 DONE
python3 rptorder_lstm.py --gpu 1 --mainfolder MDCconference --start_K 250 --end_K 500
499 DONE
python3 rptorder_lstm.py --gpu 1 --mainfolder MDCconference --start_K 500 --end_K 750
749 DONE
python3 rptorder_lstm.py --gpu 1 --mainfolder MDCconference --start_K 750 --end_K 900
899 DONE
python3 rptorder_lstm.py --gpu 3 --mainfolder MDCconference --start_K 900 --end_K 1076
1074 DONE




tmux a -t temp

python3 rptaov_lstm.py --gpu 3 --mainfolder MDCconference
498





tmux new -s rptorder_transformer_01
tmux new -s rptorder_transformer_02
tmux new -s rptorder_transformer_03
tmux new -s rptorder_transformer_04
tmux new -s rptorder_transformer_05
tmux new -s rptorder_transformer_06
tmux new -s rptorder_transformer_07
tmux new -s rptorder_transformer_08
tmux new -s rptorder_transformer_09

conda activate darts
cd jianlin-tft



python3 rptorder_transformer.py --gpu 1 --mainfolder MDCconference --start_K 28 --end_K 125
28 DK -- 124 DONE

python3 rptorder_transformer.py --gpu 1 --mainfolder MDCconference --start_K 180 --end_K 250
184 DK -- 249 DONE

python3 rptorder_transformer.py --gpu 3 --mainfolder MDCconference --start_K 300 --end_K 375
313 DK -- 374 DONE

python3 rptorder_transformer.py --gpu 3 --mainfolder MDCconference --start_K 430 --end_K 500
435 DK -- 499 DONE

python3 rptorder_transformer.py --gpu 7 --mainfolder MDCconference --start_K 550 --end_K 625
562 DK -- 624 DONE

python3 rptorder_transformer.py --gpu 0 --mainfolder MDCconference --start_K 680 --end_K 750
688 DK -- 749 DONE

python3 rptorder_transformer.py --gpu 4 --mainfolder MDCconference --start_K 800 --end_K 875
804 DK -- 874 DONE

python3 rptorder_transformer.py --gpu 3 --mainfolder MDCconference --start_K 930 --end_K 1000
937 DK -- 999 DONE



python3 rptorder_transformer.py --gpu 0 --mainfolder MDCconference --start_K 1000 --end_K 1010
1009 DONE

python3 rptorder_transformer.py --gpu 1 --mainfolder MDCconference --start_K 1010 --end_K 1020
1019 DONE

python3 rptorder_transformer.py --gpu 2 --mainfolder MDCconference --start_K 1020 --end_K 1030
1029 DONE

python3 rptorder_transformer.py --gpu 3 --mainfolder MDCconference --start_K 1030 --end_K 1040
1039 DONE

python3 rptorder_transformer.py --gpu 0 --mainfolder MDCconference --start_K 1040 --end_K 1050
1049 DONE

python3 rptorder_transformer.py --gpu 1 --mainfolder MDCconference --start_K 1050 --end_K 1060
1059 DONE

python3 rptorder_transformer.py --gpu 2 --mainfolder MDCconference --start_K 1060 --end_K 1076
1074 DONE





========================================
# 12/2/2022 3AM

tmux new -s rptaov_cond_transf_01
tmux new -s rptaov_cond_transf_02
tmux new -s rptaov_cond_transf_03
tmux new -s rptaov_cond_transf_04
tmux new -s rptaov_cond_transf_05
tmux new -s rptaov_cond_transf_06
tmux new -s rptaov_cond_transf_07
tmux new -s rptaov_cond_transf_08

conda activate darts
cd jianlin-tft

python3 rptaov_transformer.py --gpu 1 --mainfolder MDCconference --start_K 0 --end_K 125
124 DONE

python3 rptaov_transformer.py --gpu 1 --mainfolder MDCconference --start_K 125 --end_K 250
249 DONE

python3 rptaov_transformer.py --gpu 1 --mainfolder MDCconference --start_K 250 --end_K 375
374 DONE

python3 rptaov_transformer.py --gpu 1 --mainfolder MDCconference --start_K 375 --end_K 500
499 DONE

python3 rptaov_transformer.py --gpu 0 --mainfolder MDCconference --start_K 500 --end_K 625
python3 rptaov_transformer.py --gpu 4 --mainfolder MDCconference --start_K 580 --end_K 625
python3 rptaov_transformer.py --gpu 1 --mainfolder MDCconference --start_K 611 --end_K 625
596 - 610
611 - 624

python3 rptaov_transformer.py --gpu 4 --mainfolder MDCconference --start_K 625 --end_K 750
749 DONE

python3 rptaov_transformer.py --gpu 7 --mainfolder MDCconference --start_K 750 --end_K 875
874 DONE

python3 rptaov_transformer.py --gpu 7 --mainfolder MDCconference --start_K 875 --end_K 1076
1074 DONE

=======================================


568 - 579
574 - 579 DONE
607 - 624 DONE
861 - 875 DONE










python3 rptaov_transformer.py --gpu 1 --mainfolder MDCconference --start_K 574 --end_K 579


