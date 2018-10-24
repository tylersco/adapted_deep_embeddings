# Adapted Deep Embeddings: A Synthesis of Methods for k-Shot Inductive Transfer Learning (NIPS 2018)

[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)

Code associated with "Adapted Deep Embeddings: A Synthesis of Methods for k-Shot Inductive Transfer Learning" (https://arxiv.org/abs/1805.08402)

## Installation

`pip install -r requirements.txt`

Note: the code was tested with Tensorflow v1.11.0

## Datasets

1. [MNIST](http://yann.lecun.com/exdb/mnist/)
    * Make sure to uncompress the `.gz` files
2. [Isolet](https://archive.ics.uci.edu/ml/datasets/isolet)
    * Make sure to uncompress the `.Z` files 
3. [Omniglot](https://github.com/brendenlake/omniglot)
    * I used Jake Snell's script to process the Omniglot dataset (https://github.com/jakesnell/prototypical-networks/blob/master/download_omniglot.sh)
4. [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/)
    * Feel free to remove the test set. It isn't used in the code. (only train & val are needed)

## Running Experiments

The command-line arguments for a given experiment are specified in `opts.txt` files. Some of the arguments will need to be changed to accomodate your system, such as the path to a given dataset. The `opts.txt` files for the experiments in the paper are specified at:

`trained_models/<dataset_name>/<dataset_name>_<k>_<n>/<model_name>/opts.txt`

where the dataset names are [mnist, isolet, omniglot, tiny_imagenet] and the model names are [baseline, weight_transfer, hist_loss, adap_hist_loss, proto_net, adap_proto_net]. To run an experiment:

`python3 main.py @trained_models/<dataset_name>/<dataset_name>_<k>_<n>/<model_name>/opts.txt`

For example, to run MNIST weight transfer for k = 10, n = 5:

`python3 main.py @trained_models/mnist/mnist_10_5/weight_transfer/opts.txt`

Debugging information as well as the results for the experiment will be stored in a `log.txt` file in the same directory as the associated `opts.txt` file.

Some of the models, specifically with Tiny ImageNet, require large amounts of memory and can run for several weeks (if you are doing 10 replications). I tested these models with an Nvidia GTX 1080 TI graphics card and 64 GB RAM. 
