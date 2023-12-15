import torch
import torch.nn as nn

import packed_models

def train_cycle(model, hparams, train_loader, val_loader):
    model = model.to(hparams.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=hparams.lr,
        momentum=hparams.momentum,
        weight_decay=hparams.weight_decay
    )

    running_lr = hparams.lr

    for epoch in range(hparams.epochs):
        if epoch in hparams.milestones:
            running_lr *= hparams.gamma_lr
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=running_lr,
                momentum=hparams.momentum,
                weight_decay=hparams.weight_decay
            )

        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(hparams.device)
            if isinstance(model, packed_models.PackedResNet18):
                labels_one_hot = torch.zeros(
                    (labels.size(0), 10),
                    dtype=torch.float32
                )
                labels_one_hot[torch.arange(labels.size(0)), labels] = 1
                packed_labels_one_hot = labels_one_hot.repeat(1, 4)
                labels = packed_labels_one_hot
            labels = labels.to(hparams.device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(hparams.device)
                labels = labels.to(hparams.device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()

        accuracy = 100 * correct / total
        print(f'Epoch: {epoch+1}/{hparams.epochs} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}%', flush=True)