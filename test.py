import torchmetrics

def test_cycle(model, hparams, val_loader):
    """
    Calculates Acc, Cross-entropy, ECE, AUPR, AUC and FPR95
    """

    acc = torchmetrics.Accuracy(task='multiclass', num_classes=10)
    ece = torchmetrics.CalibrationError(task='multiclass', num_classes=10)
    aupr = torchmetrics.AUROC(task='multiclass', num_classes=10)
    auc = torchmetrics.AUROC(task='multiclass', num_classes=10)
    # fpr95 = torchmetrics.FalsePositiveRate(num_thresholds=1000, pos_label=1, compute_on_step=False)

    for i, (img, label) in enumerate(val_loader):
        img = img.to(hparams.device)
        label = label.to(hparams.device)

        output = model(img)
        acc.update(output, label)
        ece.update(output, label)
        aupr.update(output, label)
        auc.update(output, label)
        # fpr95.update(output, label)

    return acc.compute(), ece.compute(), aupr.compute(), auc.compute()#, fpr95.compute()