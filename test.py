import torch
import torchmetrics

def test_cycle(model, hparams, val_loader):
    """
    Calculates Acc, Cross-entropy, ECE, AUPR, AUC and FPR95
    """

    acc = torchmetrics.Accuracy(task='multiclass', num_classes=10).to(hparams.device)
    ece = torchmetrics.CalibrationError(task='multiclass', num_classes=10).to(hparams.device)
    aupr = torchmetrics.classification.MulticlassAveragePrecision(num_classes=10).to(hparams.device)
    auc = torchmetrics.AUROC(task='multiclass', num_classes=10).to(hparams.device)
    # fpr95 = torchmetrics.FalsePositiveRate(num_thresholds=1000, pos_label=1, compute_on_step=False).to(hparams.device)
    model = model.to(hparams.device)
    model.eval()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(hparams.device)
            labels = labels.to(hparams.device)

            outputs = model(images)
            acc.update(outputs, labels)
            ece.update(outputs, labels)
            aupr.update(outputs, labels)
            auc.update(outputs, labels)
            # fpr95.update(outputs, labels)

    acc = acc.compute()
    ece = ece.compute()
    aupr = aupr.compute()
    auc = auc.compute()
    # fpr95 = fpr95.compute()

    return acc, ece, aupr, auc#, fpr95