import logging

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

import utils.plotconfusionmatrix

log = logging.getLogger(__name__)

def eval(model, device, loader, plot=True):

    ## Launch predictions on test and calculate metrics
    acc_ = 0.0
    y_ = []
    preds_ = []

    model.eval()

    for batch in loader:

        batch = batch.to(device)
        pred_ = model(batch).max(1)[1]
        acc_ += pred_.eq(batch.y).sum().item()

        y_.append(batch.y)
        preds_.append(pred_)

    # TODO: OJO A QUE ESTO NO COJA LA LONGITUD EN BATCHES
    log.info("CHECK CHECK CHECK: {0}".format(len(loader)))
    acc_ /= len(loader)

    prec_, rec_, fscore_, _ = precision_recall_fscore_support(y_, preds_, average='binary')

    log.info("Metrics")
    log.info("Accuracy: {0}".format(acc_))
    log.info("Precision: {0}".format(prec_))
    log.info("Recall: {0}".format(rec_))
    log.info("F-score: {0}".format(fscore_))

    if (plot):
        conf_matrix_ = confusion_matrix(y_, preds_)

        ## Plot non-normalized confusion matrix
        utils.plotconfusionmatrix.plot_confusion_matrix(conf_matrix_, classes=np.unique(y_),
                            title='Confusion matrix, without normalization')