import argparse
import logging
import os

from run_model import run

def get_opts():
    class MyArgumentParser(argparse.ArgumentParser):
        def convert_arg_line_to_args(self, arg_line):
            return arg_line.split()

    parser = MyArgumentParser(fromfile_prefix_chars='@')

    # Dataset parameters
    parser.add_argument('-d', '--dataset', type=str, required=True, choices=('mnist', 'omniglot', 'tiny_imagenet', 'isolet'),
        dest='dataset', help='Name of dataset to be used in lowercase')
    parser.add_argument('-dp', '--data_path', type=str, required=True,
        help='Path to dataset files')

    # Training parameters
    parser.add_argument('--t1_train', type=int, default=None,
        help='Number of training examples from task 1')
    parser.add_argument('--t1_valid', type=int, default=None,
        help='Number of validation examples from task 1')
    parser.add_argument('-k', type=int, required=True,
        help='Number of examples per class for training on task 2')
    parser.add_argument('-n', type=int, required=True,
        help='Number of classes for training on task 1 & task 2')
    parser.add_argument('--t2_test', type=int, default=None,
        help='Number of test examples from task 2')

    # Model parameters
    parser.add_argument('-e', '--epochs', type=int, default=10,
        help='Number of passes through the data for training the model')
    parser.add_argument('-bs', '--batch_size', type=int, default=128,
        help='Number of data examples in each batch')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
        help='Float value for network learning rate')
    parser.add_argument('-p', '--patience', type=int, default=10,
        help='Number of epochs to wait for improvement before early stopping')
    parser.add_argument('-esr', '--percentage_es', type=float, default=0.01,
        help='Percentage increase that must be attained to continue training')

    # Other parameters
    parser.add_argument('-r', '--random_seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--replications', type=int, default=10,
        help='Number of replications to run the model')
    parser.add_argument('-g', '--gpu', type=str,
        help='Index of GPU to use for training')
    parser.add_argument('-ctl', '--controller', type=str, default='/cpu:0',
        help='Default controller device for handling non-gpu operations')
    parser.add_argument('-sd', '--save_dir', type=str, default='./',
        help='Directory to save model')
    parser.add_argument('-lf', '--log_file', type=str, default='./log.txt',
        help='Path to log file created for training')

    subparsers = parser.add_subparsers(dest='command')

    # Histogram Loss subparser
    parser_hist_loss = subparsers.add_parser('hist_loss', help='Histogram loss model')

    parser_hist_loss.add_argument('--dist', type=str, choices=['cos'], default='cos',
        help='Distance function to use with histogram loss')
    parser_hist_loss.add_argument('--kappa', type=int, default=1,
        help='Kappa value used when computing Recall@kappa')
    parser_hist_loss.add_argument('--adaptive', action='store_true',
        help='Adapt embedding using task 2 training set')

    # Prototypical Networks subparser
    parser_proto = subparsers.add_parser('proto', help='Prototypical networks model')

    parser_proto.add_argument('--classes_per_episode', type=int, required=True,
        help='Number of classes per episode of training')
    parser_proto.add_argument('--query_train_per_class', type=int, required=True,
        help='Number of query examples per class for each episode of training')
    parser_proto.add_argument('--training_episodes', type=int, required=True,
        help='Number of episodes when training the model')
    parser_proto.add_argument('--evaluation_episodes', type=int, required=True,
        help='Number of episodes when evaluating the model')
    parser_proto.add_argument('--adaptive', action='store_true',
        help='Adapt embedding using task 2 training set')
    parser_proto.add_argument('--query_batch_size', type=int, default=0,
        help='Size of batch for query points for prototypical networks')

    # Weight Transfer subparser
    parser_weight_transfer = subparsers.add_parser('weight_transfer', help='Weight transfer model')

    # Non Transfer subparser
    parser_non_transfer = subparsers.add_parser('baseline', help='Non-transfer baseline model')

    return parser.parse_args()

def main():
    params = get_opts()

    if params.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(params.gpu)

    logging.basicConfig(filename=params.log_file, level=logging.DEBUG)

    run(params)

if __name__ == '__main__':
    main()
