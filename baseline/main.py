import torch
import torch.nn as nn

import argparse
import os
from os import path
from experiment import Experiment

def main():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()

    # Add arguments to parser
    parser.add_argument('--data_dir', '-data_dir',
                        default='/home/shtoshni/Research/entnet_pytorch/data',
                        help='Root directory of data', type=str)
    parser.add_argument('--task_id', '-task_id', default=1,
                        help='BABI task ID', type=int)
    parser.add_argument('--joint', '-joint', default=False, action="store_true")
    parser.add_argument('--rand_seed', '-seed', default=0,
                        help='Random seed to get different runs', type=int)
    parser.add_argument('--base_model_dir',
                        default='/home/shtoshni/Research/entnet_pytorch/models',
                        help='Root folder storing model runs', type=str)
    parser.add_argument('--memory_slots', '-memslots',
                        help='Number of memory cells', default=20, type=int)
    parser.add_argument('--hidden_size', '-hsize',
                        help='Hidden/Embedding size used in the model',
                        default=100, type=int)
    parser.add_argument('--batch_size', '-bsize',
                        help='Batch size', default=32, type=int)
    parser.add_argument('--max_epochs', '-mepochs',
                        help='Maximum number of epochs',
                        default=200, type=int)
    parser.add_argument('--bow_encoding', '-bow', default=False,
                        action='store_true',
                        help='Whether to use BoW encoding for sentences.')
    parser.add_argument('--init_lr', help="Initial learning rate",
                        default=0.005, type=float)
    parser.add_argument('--learning_rate_decay_epochs', '-lr_decay',
                        default=-1, help='Learning rate decay rate', type=int)
    parser.add_argument('--oneK', '-oneK',
                        help="If true, use 1K training samples",
                        default=False, action="store_true")
    parser.add_argument('--eval', '-eval', help="Evaluate model",
                        default=False, action="store_true")
    args = parser.parse_args()

    # Get model directory name
    model_name = ""
    if not args.joint:
        model_name = "qa" + str(args.task_id) + "_"
    else:
        model_name = "joint_"

    model_name += "seed_" + str(args.rand_seed) + "_"
    if args.memory_slots != 20:
        model_name += "mem_" + str(args.memory_slots) + "_"
    if args.hidden_size != 100:
        model_name += "hsize_" + str(args.hidden_size) + "_"
    if args.batch_size != 32:
        model_name += "bsize_" + str(args.batch_size) + "_"
    if args.bow_encoding:
        model_name += 'bow_'
    if args.oneK:
        model_name += 'onek_'
    if args.learning_rate_decay_epochs != -1:
        model_name += 'lrd_' + str(args.learning_rate_decay_epochs) + '_'
    if args.init_lr != 0.005:
        model_name += 'init_lr_' + str(args.init_lr) + '_'

    # Strip any _ endings in model_name
    if model_name[-1] == "_":
        model_name = model_name[:-1]

    model_path = path.join(args.base_model_dir, model_name)
    best_model_path = path.join(
        path.join(args.base_model_dir, 'best_models'), model_name)
    if not path.exists(model_path):
        os.makedirs(model_path)
    if not path.exists(best_model_path):
        os.makedirs(best_model_path)

    Experiment(data_dir=args.data_dir, model_dir=model_path,
                      best_model_dir=best_model_path,
                      task_id=args.task_id, joint=args.joint,
                      memory_slots=args.memory_slots,
                      hidden_size=args.hidden_size,
                      bow_encoding=args.bow_encoding,
                      rand_seed=args.rand_seed,
                      init_lr=args.init_lr,
                      lr_decay = args.learning_rate_decay_epochs,
                      max_epochs=args.max_epochs,
                      oneK=args.oneK,
                      batch_size=args.batch_size,
                      eval=args.eval)


if __name__=="__main__":
    main()
