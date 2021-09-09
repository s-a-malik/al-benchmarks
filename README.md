# al-benchmarks

A repository for active learning benchmarks. This is specifically set up to evaluate the effect of model transfer where data acquired by one model is used to train another.

The goal of this repository is to maintain an up to date evaluation of different active
learning setups and be able to iterate over them quickly.

We can compare how active learning performs for:

* Different active learning strategies
* Different datasets
* Different model types and hyperparameters

The experiments run on the UCL Cluster.

## Usage

First install dependancies and login to wandb `pip install -r requirements.txt; wandb login <API_KEY>`. Then to run an experiment, run `python cluster_experiment_runner.py --config your_experiment.yaml`. For cluster experiments, see example submission script `run_experiments.sh`, and hyperparameter sweep `run_sweep.sh`. The experiment will output (logged to wandb):

1. Accuracy, Precision, Recall and F1 scores as a function of dataset size for each data-set and each active learning strategy on test/train/dev averaged across num_repetitions with uncertainty bars
2. Training curves on test/train/dev for each experiment run will be logged
3. Logs of all hyper-paramters are saved alongside the experimental results and the models are checkpointed every epoch (saving the best peforming model and the most recent)

### Hyperparamters you can change

Experimental settings are adjusted via a .yaml config file. See experiments folder for examples, and an example shell script for cluster submission.

Each experiment has one Acquisition Model, which is used to inform data-selection.

Each experiment has one or more Evaluation Model's which are trained on the acquired data.

Datasets:

* Classification:
  * AG News
  * DBPedia
  * YRF

Class Balance of datasets:

* Balanced
* Imbalanced
* Class imbalance of the seed data

Model types:

* BERT
* RNN based models
* MLP
* Logistic Regression

Active learning methods:

* Random
* Uncertainty-based
  * entropy acquistion
  * Least confidence
  * Margin
  * MC Dropout - same as above but use MC estimate of posterior instead of max likelihood
  * BALD
* Diversity-based
  * Core-set (k-center greedy)
* Uncertainty + Diversity-based
  * BatchBALD 

## Other Benchmark Papers

* [Ein-Dor 2020](https://www.aclweb.org/anthology/2020.emnlp-main.638.pdf) ([code](https://github.com/IBM/low-resource-text-classification-framework))
* [Munjal 2020](https://arxiv.org/pdf/2002.09564.pdf) (vision-centric, code available offline)
* [Prabhu 2019](https://arxiv.org/pdf/1909.09389.pdf) ([code](https://github.com/drimpossible/Sampling-Bias-Active-Learning))
* [Siddhant 2018](https://arxiv.org/pdf/1808.05697.pdf) ([code](https://github.com/asiddhant/Active-NLP))
* [Lipton 2019](https://arxiv.org/pdf/1807.04801.pdf) (model bias focussed)
* [Atighehchian 2020](https://arxiv.org/pdf/2006.09916.pdf) ([library](https://baal.readthedocs.io/en/stable/index.html))
