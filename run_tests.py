import sys
import os

import torch

from utils import HParams, get_data, get_model
from test import *

def main():
    out_dir = './saved_models'

    hparams = HParams()
    _, val_loader = get_data(hparams, training=False)

    if len(sys.argv) < 2:
        print('Missing test type')
        print('Exiting...')
        exit()
    test_type = sys.argv[1]
    model_fnames = sys.argv[2:]
    if len(sys.argv) < 3:
        model_fnames = input().split(' ')

    plural = 's' if len(sys.argv) > 2 else ''
    print(f'Testing model{plural} {", ".join(model_fnames)}...')

    models = list()

    for model_fname in model_fnames:
        model_name = '_'.join(model_fname.split('_')[:-1])
        model = get_model(model_name)

        print(f'Loading model from {model_fname}...')
        path = os.path.join(out_dir, model_fname)
        model.load_state_dict(torch.load(path))
        models.append(model)

    if test_type == 's':
        acc, ece, aupr, auc = test_cycle(models[0], hparams, val_loader)
    elif test_type == 'de':
        acc, ece, aupr, auc = test_cycle_DE(models, hparams, val_loader)
    elif test_type == 'pe':
        acc, ece, aupr, auc = test_cycle_PE(models[0], hparams, val_loader)
    else:
        raise ValueError(f'Unknown test type: {test_type}')

    print(f'Accuracy: {acc}')
    print(f'ECE: {ece}')
    print(f'AUPR: {aupr}')
    print(f'AUC: {auc}')

if __name__ == '__main__':
    main()