import os
import numpy as np

import torch
import torch.nn as nn

from utils import HParams, get_data
from train import train_cycle
from test import test_cycle
import baseline_models
import packed_models

def main():
    out_dir = './saved_models'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    hparams = HParams()
    train_loader, val_loader = get_data(batch_size=hparams.batch_size, data_transforms=hparams.data_transforms)

    baseline_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    baseline_model.fc = nn.Linear(512, 10)
    baseline_model_scratch = baseline_models.ResNet18()
    packed_model = packed_models.PackedResNet18(alpha=2, gamma=2, n_estimators=4)

    models = {
        '1': baseline_model,
        '2': baseline_model_scratch,
        '3': packed_model,
    }

    model_names = {
        '1': 'baseline',
        '2': 'baseline_scratch',
        '3': 'packed',
    }

    query_str = '\n'.join([
        '✨Welcome to the CIFAR-10 training and testing script!✨',
        '===================================================',
        'Which model should be trained?',
        'Choose one or more from the following options:',
        '',
        '1. Baseline model',
        '2. Baseline model from scratch',
        '3. Packed model',
        '===================================================',
        '',
    ])

    model_idxs = input(query_str)

    for idx in model_idxs:
        if not idx in models:
            print(f'Invalid model index: {idx}')
            print('Exiting...')
            exit()
    else:
        plural = 's' if len(model_idxs) > 1 else ''
        print(f'Excellent choice{plural}! Training model{plural} {", ".join(model_idxs)}...')

        for idx in model_idxs:
            model = models[idx]

            print(f'Training {model_names[idx]}...')
            train_cycle(model, hparams, train_loader, val_loader)

            unique_id = np.random.randint(0, 100000)
            path = os.path.join(out_dir, f'{model_names[idx]}_{unique_id}.pt')
            while os.path.exists(path):
                unique_id = np.random.randint(0, 100000)

            print(f'Saving model to {path}...')
            torch.save(model.state_dict(), path)

            test_cycle(model, hparams, val_loader)

if __name__ == '__main__':
    main()