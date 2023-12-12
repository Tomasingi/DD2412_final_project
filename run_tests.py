import sys
import os

import torch

from utils import HParams, get_data, get_model
from test import test_cycle

def main():
    out_dir = './saved_models'

    hparams = HParams()
    train_loader, val_loader = get_data(hparams)

    pretrained_models = sys.argv[1:]

    if len(pretrained_models) == 0:
        print('No pretrained models specified. Testing all models...')
        pretrained_models = os.listdir(out_dir)
    else:
        plural = 's' if len(sys.argv) > 2 else ''
        print(f'Testing model{plural} {", ".join(pretrained_models)}...')

    for pretrained_model in pretrained_models:
        model_name = '_'.join(pretrained_model.split('_')[:-1])
        model = get_model(model_name)

        print(f'Loading model from {pretrained_model}...')
        path = os.path.join(out_dir, pretrained_model)
        model.load_state_dict(torch.load(path))

        acc, ece, aupr, auc = test_cycle(model, hparams, val_loader)

        print(f'Accuracy: {acc}')
        print(f'ECE: {ece}')
        print(f'AUPR: {aupr}')
        print(f'AUC: {auc}')

if __name__ == '__main__':
    main()