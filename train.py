import torch
import torch.nn as nn

def train_cycle(model, hparams, train_loader, val_loader, verbose):
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
        if verbose:
            print(f'Epoch: {epoch+1}/{hparams.epochs} | Loss: {loss.item():.4f} | Accuracy: {accuracy:.2f}%')