from os import path
import torch
import torch.nn as nn

import numpy as np
import matplotlib.pyplot as plt
import argparse

import seaborn as sns


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_dir', '-mdir',
                        help="Model directory", type=str, required=True)
    parser.add_argument('--plot_dir',
                        default='/home/shtoshni/Research/entnet_pytorch/analysis/plots', type=str)

    args = parser.parse_args()

    return args


def analyze_model(model_dir, plot_dir):
    # Load the best model
    model_path = path.join(model_dir, 'model_best.pth')
    model_name = path.basename(path.normpath(model_dir))
    if not path.exists(model_path):
        print ('Model NOT FOUND!')
        return

    print ('Loading previous model')
    checkpoint = torch.load(model_path)
    model = checkpoint['model_state_dict']
    itos = checkpoint['itos']

    # print (type(model['R']))
    # Get the key matrix and the embedding matrix
    key_mat = None
    emb_mat = None
    for param_name in model.keys():
        if param_name == 'memory_net.keys':
            key_mat = model[param_name].data
        elif param_name == 'embedding.weight':
            emb_mat = model[param_name].data

    dot_mat = torch.mm(key_mat, torch.transpose(emb_mat, 0, 1))
    key_sim = torch.mm(key_mat, torch.transpose(key_mat, 0, 1))

    num_keys, vocab_size = list(dot_mat.size())
    dot_mat = dot_mat.cpu().numpy()
    key_sim = key_sim.cpu().numpy()

    sns_plot = sns.heatmap(dot_mat.T, cmap="GnBu", yticklabels=itos, annot=True)
    plt_file = path.join(plot_dir, model_name + '.png')
    sns_plot.figure.savefig(plt_file, dpi=300, format='png', bbox_inches='tight')
    sns_plot.clear()

    sns_plot_2 = sns.heatmap(key_sim, cmap="GnBu", cbar=False, annot=True)
    plt_file = path.join(plot_dir, 'key_' + model_name + '.png')
    sns_plot_2.figure.savefig(plt_file, dpi=300, format='png', bbox_inches='tight')

    # plt.show()

def main():
    args = parse_args()
    analyze_model(args.model_dir, args.plot_dir)


if __name__=='__main__':
    main()
