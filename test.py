import torch
import torchmetrics

def test_cycle(model, hparams, val_loader):
    """
    Calculates Acc, Cross-entropy, ECE, AUPR, AUC and FPR95
    """

    acc = torchmetrics.Accuracy(task='multiclass', num_classes=10).to(hparams.device)
    ece = torchmetrics.CalibrationError(task='multiclass', num_classes=10).to(hparams.device)
    aupr = torchmetrics.classification.MulticlassAveragePrecision(num_classes=10, thresholds=5).to(hparams.device)
    auc = torchmetrics.classification.MulticlassAUROC(num_classes=10, thresholds=5).to(hparams.device)
    fpr95 = 0
    model = model.to(hparams.device)
    model.eval()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(hparams.device)
            labels = labels.to(hparams.device)

            outputs = model(images)
            outputs = torch.softmax(outputs, dim=1)
            acc.update(outputs, labels)
            ece.update(outputs, labels)
            aupr.update(outputs, labels)
            auc.update(outputs, labels)

            neg_one_hot_labels = torch.ones_like(outputs)
            neg_one_hot_labels.scatter_(1, labels.unsqueeze(1), 0)
            outputs_greater_than_95th_percentile = outputs > 0.05
            false_positives = torch.sum(neg_one_hot_labels * outputs_greater_than_95th_percentile)
            fpr95 += false_positives / len(val_loader.dataset)

    total_acc = acc.compute()
    total_ece = ece.compute()
    total_aupr = aupr.compute()
    total_auc = auc.compute()
    total_fpr95 = fpr95

    return total_acc, total_ece, total_aupr, total_auc, total_fpr95

def test_cycle_DE(models, hparams, val_loader):
    """
    Same as test_cycle, but for Deep Ensembles
    """

    acc = torchmetrics.Accuracy(task='multiclass', num_classes=10).to(hparams.device)
    ece = torchmetrics.CalibrationError(task='multiclass', num_classes=10).to(hparams.device)
    aupr = torchmetrics.classification.MulticlassAveragePrecision(num_classes=10, thresholds=5).to(hparams.device)
    auc = torchmetrics.classification.MulticlassAUROC(num_classes=10, thresholds=5).to(hparams.device)
    fpr95 = 0

    for model in models:
        model = model.to(hparams.device)
        model.eval()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(hparams.device)
            labels = labels.to(hparams.device)

            outputs = torch.stack([model(images) for model in models])
            outputs = torch.softmax(outputs, dim=2)
            outputs = torch.mean(outputs, dim=0)
            acc.update(outputs, labels)
            ece.update(outputs, labels)
            aupr.update(outputs, labels)
            auc.update(outputs, labels)

            neg_one_hot_labels = torch.ones_like(outputs)
            neg_one_hot_labels.scatter_(1, labels.unsqueeze(1), 0)
            outputs_greater_than_95th_percentile = outputs > 0.95
            false_positives = torch.sum(neg_one_hot_labels * outputs_greater_than_95th_percentile)
            fpr95 += false_positives / len(val_loader.dataset)

    total_acc = acc.compute()
    total_ece = ece.compute()
    total_aupr = aupr.compute()
    total_auc = auc.compute()
    total_fpr95 = fpr95

    return total_acc, total_ece, total_aupr, total_auc, total_fpr95

def test_cycle_PE(model, hparams, val_loader, num_models=4):
    """
    Same as test_cycle, but for Packed-Ensembles
    """

    acc = torchmetrics.Accuracy(task='multiclass', num_classes=10).to(hparams.device)
    ece = torchmetrics.CalibrationError(task='multiclass', num_classes=10).to(hparams.device)
    aupr = torchmetrics.classification.MulticlassAveragePrecision(num_classes=10, thresholds=5).to(hparams.device)
    auc = torchmetrics.classification.MulticlassAUROC(num_classes=10, thresholds=5).to(hparams.device)
    fpr95 = 0

    model = model.to(hparams.device)
    model.eval()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(hparams.device)
            labels = labels.to(hparams.device)

            outputs = model(images)

            # Output is of shape (batch_size, num_models * num_classes)
            # We need to reshape it to (batch_size, num_models, num_classes)
            outputs = outputs.reshape(images.shape[0], num_models, 10)
            outputs = torch.softmax(outputs, dim=2)
            outputs = torch.mean(outputs, dim=1)

            acc.update(outputs, labels)
            ece.update(outputs, labels)
            aupr.update(outputs, labels)
            auc.update(outputs, labels)

            neg_one_hot_labels = torch.ones_like(outputs)
            neg_one_hot_labels.scatter_(1, labels.unsqueeze(1), 0)
            outputs_greater_than_95th_percentile = outputs > 0.95
            false_positives = torch.sum(neg_one_hot_labels * outputs_greater_than_95th_percentile)
            fpr95 += false_positives / len(val_loader.dataset)

    total_acc = acc.compute()
    total_ece = ece.compute()
    total_aupr = aupr.compute()
    total_auc = auc.compute()
    total_fpr95 = fpr95

    return total_acc, total_ece, total_aupr, total_auc, total_fpr95